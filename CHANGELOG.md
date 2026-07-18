# Changelog

## [Unreleased]

### Added

- Phase 0 scaffolding: package layout, `pyproject.toml` (setuptools, flat
  layout, matching `soarm_lerobot`'s convention), pinned `mjlab==1.5.0` /
  `mujoco-warp==3.10.0.2` (verified installable together on macOS arm64),
  CI (lint + package-import smoke test), `LICENSE`, this changelog.
- Switched install/dependency tooling from plain pip to `uv`, matching
  mjlab's own tooling: `cpu`/`cu128` extras forwarding to `mjlab[cpu]`/
  `mjlab[cu128]` (mutually exclusive via `[tool.uv.conflicts]`), `torch`
  routed to the matching PyTorch package index per extra via
  `[tool.uv.sources]` (Linux only — darwin has one torch wheel, no
  CPU/CUDA choice), `dev` dependencies moved from
  `[project.optional-dependencies]` to PEP 735 `[dependency-groups]`,
  `uv.lock` committed for full-tree reproducibility, `Makefile` with
  `sync`/`sync-cpu`/`lint`/`test`/`test-cpu`/`check` targets, CI switched
  to `astral-sh/setup-uv` + `uv sync --locked` (fails the build on
  lockfile drift). Verified locally on macOS arm64 (`sync-cpu`, lint,
  test, import all pass); the Linux CUDA/CPU index-routing path is
  exercised for the first time by CI itself (`ubuntu-latest`), not
  verified on this dev machine.
- Phase 1: the "Reach" sample task, end to end — SO-ARM100 asset (MJCF
  vendored from the SO-ARM100 submodule's SO101 revision, STS3215 actuator
  gains, home keyframe, gripper-only collision), `UniformPoseCommand` with a
  target-position box derived from joint limits + forward kinematics (not
  hand-picked), reward/observation/termination MDP terms, PPO runner config,
  and `scripts/train.py`/`play.py`/`list_envs.py`. Verified:
  `python scripts/train.py SoArm100-Reach --env.scene.num-envs=4
  --agent.max-iterations=2 --gpu-ids None` runs to completion with no
  exceptions and a non-NaN reward curve (macOS arm64, CPU).
- Small `_mjlab_compat` shim (applied on `import soarm_mjlab.tasks`) working
  around a `mujoco-warp==3.10.0.2` regression: entities whose joint limits
  are never touched by a domain-randomization event crash on any reset with
  `num_envs > 1` (reproduces identically on mjlab's own bundled
  `Mjlab-Lift-Cube-Yam` task). Already reported and fixed upstream as
  [mujocolab/mjlab#1091](https://github.com/mujocolab/mjlab/pull/1091)
  (draft, opened independently the same day) — this shim is a narrower
  stopgap for our pinned versions until that merges and releases; safe to
  delete once it does.
- Phase 2: the test pyramid's four CI-safe layers — `test_mdp_reach.py`
  (unit tests for the reward/observation/termination functions, synthetic
  tensors, no MuJoCo), `test_asset_so_arm100.py` (MJCF compiles, actuators
  line up with joints 1:1, collision config applies to the right geoms,
  every Reach `# Set per-robot` placeholder is filled), `test_env_reach.py`
  (real `ManagerBasedRlEnv`, `num_envs=2`, reset + step with random actions,
  no NaN/Inf), `test_train_smoke.py` (`scripts/train.py` as a real
  subprocess — also exercises tyro CLI parsing — asserting a checkpoint +
  ONNX export land on disk). 24 tests, ~17s locally; no CI workflow changes
  needed, the existing `fast` job already runs all of them.
- Phase 3: `.github/workflows/ci.yml` gets a second job, `train-smoke` —
  runs a longer-but-still-small PPO slice (`num-envs=16`,
  `max-iterations=20`, ~15s locally) than the `fast` job's own 2-iteration
  test, uploading the checkpoint + tensorboard log as a build artifact.
  Triggered only on `schedule` (nightly) or `workflow_dispatch`, never on
  `push`/`pull_request`, so it structurally can't block a merge. Runs on
  `ubuntu-latest` — no GPU runner configured for this repo yet. Checkpoint
  traceability (fully-resolved `env.yaml`/`agent.yaml` + git commit
  hash/diff in the log directory) was already wired up in `scripts/train.py`
  since Phase 1; nothing new needed there.
