# Training SO-ARM100 Reach on vast.ai (Phase 4)

This is the manual/GPU half of Phase 4 (`SOARM_MJLAB_ROADMAP.md`): a real
training run — thousands of parallel envs, full `max_iterations` — on a
rented GPU, tracked via W&B, then played back locally in sim on the
MacBook, published to the Hugging Face Hub once it clears the promotion
bar, and eventually (Phase 5) run on the real arm. `soarm_mjlab` is fully
self-contained (the SO-ARM100 MJCF + meshes are vendored in-repo, not
referenced from the `soarm-ws` submodule tree), so the rented box only
ever needs `soarm_mjlab` itself — not the monorepo.

## Prerequisites

- A [vast.ai](https://vast.ai) account with billing set up (credits or a
  card on file).
- A W&B account. This Mac is already logged in (`~/.netrc` has a
  `api.wandb.ai` entry); the rented box needs its own `wandb login` — get
  the same API key from <https://wandb.ai/authorize>.
- A Hugging Face account, only needed on whichever machine runs
  `scripts/push_to_hub.py` (the Mac, in the flow below — not the rented
  box). Authenticate once with `uv run hf auth login` (paste a token from
  <https://huggingface.co/settings/tokens>).
- Nothing else — `soarm_mjlab` is public, no deploy key or token needed to
  clone it.

## 1. Rent an instance

On the [vast.ai console](https://cloud.vast.ai/create/), search offers and
pick:

- **GPU**: this task has no vision/camera observations and a tiny MLP
  policy (128×128×64) — an RTX 3090/4090 or A4000/A5000 is the right tier.
  Skip A100/H100: neither the physics workload (no contacts besides the
  gripper touching the ground, no rendering) nor the network size come
  close to needing that compute, you'd just be paying for it unused. As of
  mid-2026, RTX 4090 interruptible instances run roughly **$0.30–0.60/hr**
  on vast.ai (varies by host/region — check the live marketplace, don't
  assume the number here is current).
- **VRAM**: 16GB is comfortable, 24GB gives real headroom. At `num_envs=4096`
  the env state + PPO rollout buffer (`4096 × 24 steps × 31 obs floats` ≈
  12MB) is trivial — most of what you're paying VRAM for is torch/CUDA
  context overhead (~1–2GB) plus margin to push `num_envs` to 8192+ later
  without any OOM risk, not because this task needs it.
- **vCPUs**: 4–8 is plenty. Physics and PPO both run on the GPU via
  `mujoco_warp`/torch; the CPU is just orchestration, not a bottleneck the
  way it would be for a vision task doing CPU-side data loading.
- **RAM**: 16–32GB — comfortable, no large replay buffers or datasets to
  hold in memory.
- **Image/template**: any template with CUDA 12.8+ and a recent Ubuntu
  (e.g. vast.ai's own "PyTorch" template, or a plain
  `nvidia/cuda:12.8.1-devel-ubuntu22.04`) works — `uv sync --extra cu128`
  installs torch itself, the template just needs a driver new enough for
  CUDA 12.8.
- **Disk**: 30GB is comfortable (repo + vendored meshes ≈ 20MB, the rest is
  torch/CUDA wheels and checkpoints).
- **Interruptible vs. on-demand**: interruptible is cheaper but can be
  preempted. Fine for this task since checkpoints save every
  `save_interval` iterations (100, see `rl_cfg.py`) and training resumes
  from the last one with `--agent.resume` — but if you'd rather not deal
  with resuming, pick on-demand.

None of the above is benchmarked against actual `mujoco_warp` throughput on
these cards for this task — our own runs so far were all CPU-only at tiny
scale (4–64 envs), which doesn't extrapolate to GPU behavior at 4096 (GPU
physics scales roughly for-free across envs up to the point you saturate
SMs/bandwidth, CPU doesn't). Watch the printed `Steps per second` for the
first ~30s of the real run (step 5 below) and adjust `--env.scene.num-envs`
up or down from there rather than trusting the sizing above blindly.

Launch it and wait for the instance to show **running** in the Instances
tab.

## 2. Connect

Copy the SSH command from the instance's card in the vast.ai console (or
use their web terminal). Something like:

```bash
ssh -p <PORT> root@<HOST>
```

## 3. One-time setup

Run the setup script from the repo (installs `uv`, clones `soarm_mjlab`,
syncs the `cu128` extra):

```bash
curl -LsSf https://raw.githubusercontent.com/thanhndv212/soarm_mjlab/main/scripts/setup_remote.sh | bash
```

Or do it by hand if you'd rather see each step:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/thanhndv212/soarm_mjlab.git
cd soarm_mjlab
uv sync --locked --extra cu128 --group dev
```

## 4. Authenticate W&B

```bash
uv run wandb login
# paste the API key from https://wandb.ai/authorize
```

## 5. Launch training (in tmux — don't skip this)

SSH sessions drop; a multi-hour training run must not die with them.

```bash
tmux new -s train
cd soarm_mjlab
uv run python scripts/train.py SoArm100-Reach --env.scene.num-envs=4096
```

Leave `--agent.logger` alone — the default is `wandb`. `max_iterations`
(1500, from `rl_cfg.py`) and `num_envs` (4096 above) are the "real"
hyperparameters the roadmap calls for; scale `num_envs` down if the rented
GPU reports OOM (watch the first few seconds of output).

Detach with **Ctrl-b d** — training keeps running. Reattach any time with
`tmux attach -t train`. Check progress non-invasively with
`tmux capture-pane -t train -p | tail -30`.

## 6. Monitor

The very first lines of `train.py`'s output include a W&B run URL — open
it in a browser on the Mac for the reward curve and all
`Episode_Reward/*` / `Episode_Termination/*` / `Metrics/ee_pose/*` scalars
in real time. Tensorboard logs also land in `logs/rsl_rl/so_arm100_reach/`
if you'd rather tunnel:

```bash
# On the Mac:
ssh -p <PORT> -L 6006:localhost:6006 root@<HOST>
# On the rented box, in a second tmux window:
uv run tensorboard --logdir logs/rsl_rl/so_arm100_reach --host 0.0.0.0
```

## 7. Decide the promotion bar *before* looking at the finished curve

Per the roadmap: pick the bar now, not after seeing the number, so it
doesn't quietly move to match whatever the run produced. A reasonable
starting bar for Reach, using metrics already logged by
`UniformPoseCommand`/`distance_to_target`:

- `Metrics/ee_pose/episode_success` (fraction of episodes that ever got
  within `success_threshold` for `success_steps` consecutive steps) ≥ 0.90
  over the last ~100 logged episodes.
- `Episode_Termination/joint_limit_violated` ≈ 0.
- `Metrics/ee_pose/position_error` (mean, near the end of training) small
  relative to the target box size (see `command_cfg.position_range`).

If the run doesn't clear this, that's a Phase 4 finding (reward shaping,
more iterations, curriculum), not a reason to lower the bar after the fact.

## 8. Retrieve the checkpoint

Both the `.pt` checkpoint (via rsl_rl's own `WandbLogWriter`) and the
`.onnx` export (via `ReachOnPolicyRunner.save`) are uploaded to the W&B run
automatically — no manual copy needed. From the Mac:

```bash
uv run python scripts/play.py SoArm100-Reach \
    --wandb-run-path <entity>/<project>/<run_id>
```

(`<entity>/<project>/<run_id>` is in the W&B run URL.) If you'd rather
pull the raw files:

```bash
scp -P <PORT> root@<HOST>:soarm_mjlab/logs/rsl_rl/so_arm100_reach/*/model_*.pt .
```

## 9. Play it back locally

`scripts/play.py` auto-picks a native MuJoCo window on the MacBook
(`NativeMujocoViewer`, since there's a real display) — same command as
step 8 above. Use `--video` to save an mp4 instead of opening a window, or
`--no-terminations` to watch full rollouts without early episode cutoffs.

## 10. Publish the checkpoint to the Hugging Face Hub

Only once it clears the promotion bar from step 7 — this is a deliberate,
manual publish, not something that happens automatically on every
checkpoint (unlike the W&B upload). From the Mac, pointed at the same W&B
run:

```bash
uv run python scripts/push_to_hub.py \
    --repo-id <your-hf-username>/soarm100-reach \
    --wandb-run-path <entity>/<project>/<run_id>
```

This downloads the ONNX export + configs from the W&B run (if not already
cached locally from step 8), generates a model card with the training
provenance (iteration count, `num_envs`, git commit, W&B run), and pushes
`policy.onnx` + `model.pt` + `env.yaml`/`agent.yaml` + `README.md` to a new
or existing HF model repo. Add `--private` to keep it unlisted. If you
already have the run directory locally, `--run-dir <path>` skips the W&B
download entirely.

## 11. Shut the instance down

vast.ai bills while an instance is **running**, and *stopped* instances
still bill for disk. Once you have the checkpoint and are done iterating,
**destroy** the instance from the console (or `vastai destroy instance
<id>` if you've installed their CLI) rather than just stopping it.

## Troubleshooting

- **CUDA/driver mismatch at `uv sync`**: pick a different offer — vast.ai
  lists each host's driver version; it needs to support CUDA 12.8.
- **OOM during `env` construction or the first PPO update**: lower
  `--env.scene.num-envs`.
- **Training stops when the SSH connection drops**: you skipped the tmux
  step — always launch inside `tmux`/`screen`.
- **`wandb: ERROR ... 401`**: the API key wasn't picked up — rerun
  `uv run wandb login` inside the repo's venv (`uv run`, not a bare
  `wandb` on the system Python).
- **No display / rendering errors on the rented box**: not expected for a
  plain training run — `ManagerBasedRlEnv` only opens a render context
  when `--video` is passed, and `mjlab` defaults `MUJOCO_GL=egl` on Linux
  for that case automatically.
- **`push_to_hub.py` fails with a 401/permission error**: not logged in (or
  the token lacks write access) on the machine running it — rerun
  `uv run hf auth login` there.

Sources on current vast.ai RTX 4090 pricing (checked 2026-07; re-check the
live marketplace before renting — spot prices move):
[vast.ai/pricing/gpu/RTX-4090](https://vast.ai/pricing/gpu/RTX-4090),
[vast.ai/pricing](https://vast.ai/pricing).
