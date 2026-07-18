# soarm_mjlab

RL training of SO-ARM100 in MuJoCo via [mjlab](https://github.com/mujocolab/mjlab),
deployed through `soarm_sdk.RobotInterface` — the same interface real-hardware
control already uses, so a trained policy doesn't care whether it's driving
simulation or a physical arm.

Status: Phases 0–3 done (repo scaffolding, the "Reach" sample task, a
CI-safe test pyramid, CI/CD). Phase 4 (a real training run) is next — see
`SOARM_MJLAB_ROADMAP.md` in the `soarm-ws` root for the full phased plan,
and `docs/vast_ai_training.md` for a step-by-step guide to running it on a
rented GPU.

## Install

Uses [uv](https://docs.astral.sh/uv/) — not plain pip — because `mjlab` gates
`torch` behind mutually-exclusive `cpu`/`cu128` extras routed to different
package indices (CPU vs CUDA wheels), which `uv`'s `[tool.uv.sources]` handles
natively; pip has no equivalent short of hand-juggling `--extra-index-url`.
This is the one package in `soarm-ws` that installs this way — every other
package there is plain `pip install -e .`, justified because none of them
have a GPU/CPU dependency-variant problem.

```bash
make sync-cpu   # dev machine without a GPU (or: uv sync --extra cpu --group dev)
make sync       # GPU training box, CUDA 12.8 (or: uv sync --extra cu128 --group dev)
```

`uv.lock` is committed — it pins the full resolved dependency tree (not just
`mjlab`/`mujoco-warp` directly), which is the actual reproducibility
guarantee for RL checkpoints. CI runs `uv sync --locked`, which fails the
build if the lockfile is stale relative to `pyproject.toml`.

## Training

```bash
uv run python scripts/list_envs.py                   # see registered tasks
uv run python scripts/train.py SoArm100-Reach --env.scene.num-envs=4096
uv run python scripts/play.py SoArm100-Reach --wandb-run-path <entity>/<project>/<run_id>
uv run python scripts/push_to_hub.py --repo-id <user>/soarm100-reach --wandb-run-path <entity>/<project>/<run_id>
```

No local GPU needed for this: train on a rented GPU box (tracked live via
W&B, the default logger), play the checkpoint back in sim on any machine,
then publish a promoted checkpoint (ONNX + config + model card) to the
Hugging Face Hub — see `docs/vast_ai_training.md` for the full walkthrough
(instance selection, cost, monitoring, retrieving the checkpoint, the
promotion bar to clear before publishing).

## Common tasks

```bash
make lint       # ruff check
make test-cpu   # pytest, forcing CPU (mirrors mjlab's own tests/conftest.py convention)
make test       # pytest on whatever device is available
make check      # lint + test-cpu
```

## Why not the reference architecture's C++ deployment stack?

`unitree_rl_mjlab` (the reference this package's task/config layout is
modeled on) deploys trained policies through a C++ FSM + onnxruntime,
because a legged robot's balance controller needs a hard real-time loop.
A 6-DOF arm driven over serial at ~50 Hz already runs its control loop in
Python elsewhere in this workspace (`m5teleop/teleop.py`), so deployment
here is a Python script loading a checkpoint and driving
`soarm_sdk.RobotInterface` directly — no second language, no train/deploy
consistency problem to solve.
