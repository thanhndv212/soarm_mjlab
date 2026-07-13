# soarm_mjlab

RL training of SO-ARM100 in MuJoCo via [mjlab](https://github.com/mujocolab/mjlab),
deployed through `soarm_sdk.RobotInterface` — the same interface real-hardware
control already uses, so a trained policy doesn't care whether it's driving
simulation or a physical arm.

Status: scaffolding only. See `SOARM_MJLAB_ROADMAP.md` in the `soarm-ws` root
for the phased plan (repo setup → sample "Reach" task → testing/CI → real
training run → sim2real deployment).

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
