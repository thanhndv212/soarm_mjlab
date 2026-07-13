# soarm_mjlab

RL training of SO-ARM100 in MuJoCo via [mjlab](https://github.com/mujocolab/mjlab),
deployed through `soarm_sdk.RobotInterface` — the same interface real-hardware
control already uses, so a trained policy doesn't care whether it's driving
simulation or a physical arm.

Status: scaffolding only. See `SOARM_MJLAB_ROADMAP.md` in the `soarm-ws` root
for the phased plan (repo setup → sample "Reach" task → testing/CI → real
training run → sim2real deployment).

## Install

```bash
pip install -e .
pip install -e ".[dev]"   # + pytest, ruff
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
