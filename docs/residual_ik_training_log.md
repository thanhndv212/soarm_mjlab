# Residual IK Training Log

Tracking doc for the **residual RL over IK base controller** experiment —
the first attempt to break the ~0.04 m position-error plateau that the
v1–v11 campaign (`reach_training_debug_log_v1_v11.md`) could not, by
reformulating the action space rather than tuning hyperparameters harder.

## Hypothesis

The v11 run proved the plateau is a **problem-formulation ceiling**, not an
optimization ceiling: 56× more data (7.6 B env steps) reproduced the same
~0.03–0.04 m position error. The root cause is that the policy must learn
the full joint-angle → EE-pose mapping from scratch, with a coarse action
scale (±1.0 rad) that was tuned for *reachability* (v5) but works *against*
fine final-approach positioning.

**Residual RL** replaces this with:

```
q_target = q_current + dq_base_ik + dq_residual
```

- `dq_base_ik` — a damped-least-squares (DLS) IK step toward the commanded
  target pose, computed every step via `mujoco_warp.jac` (GPU-vectorized,
  same Jacobian machinery mjlab's own `DifferentialIKAction` uses). This
  is the "base controller" — it analytically solves "get roughly there."
- `dq_residual` — the policy's output, scaled by a small residual scale
  (±0.1 rad for arm joints, ±0.05 for gripper). The policy only learns the
  *correction* — compensating for IK imperfections, joint-limit edge
  cases, and dynamic effects the analytic controller doesn't model.

This shrinks the learning problem from "learn full reaching" to "learn
small corrections on top of an IK base," which is a much smaller function
to fit.

## What changed (code)

| File | Change |
|---|---|
| `mdp/actions.py` (new) | `ResidualIKAction` / `ResidualIKActionCfg` — GPU-vectorized DLS IK base + joint-space residual. Reuses `mujoco_warp.jac` for the Jacobian, same DLS normal-equations solve as mjlab's `DifferentialIKAction`. |
| `reach_env_cfg.py` | Action term swapped from `JointPositionActionCfg` to `ResidualIKActionCfg`. Base cfg has per-robot placeholders for `frame_name`, `command_name`. |
| `config/so_arm100/env_cfgs.py` | Fills per-robot params: `frame_type="site"`, `frame_name="gripperframe"`, `command_name="ee_pose"`, `residual_scale` per joint. Workspace computation reverted to full joint-limit range (the base IK controller can reach the full workspace, not just `home ± action_scale`). |
| `config/so_arm100/rl_cfg.py` | `init_std` 0.3 → 0.5 (the residual is already scaled down, so the raw policy output needs more initial exploration range). `entropy_coef` kept at 0.001. |
| `so_arm100_constants.py` | Added `SO_ARM100_RESIDUAL_SCALE` dict (0.1 rad arm, 0.05 gripper). |
| `mdp/__init__.py` | Exports `actions` module. |

## Configuration (v12 — first residual IK run)

- **Action**: `ResidualIKAction` with DLS base controller
  - `damping=0.05`, `max_dq=0.5` rad/step
  - `position_weight=1.0`, `orientation_weight=0.0` (position-only base —
    success is gated on position only, per v9's finding that orientation
    drift is cosmetically ugly but harmless)
  - `residual_scale`: arm joints 0.1 rad, gripper 0.05 rad
- **Workspace**: full joint-limit range (not action-scale-limited) — the
  base IK controller can reach the full workspace
- **PPO**: same as v9/v11 except `init_std=0.5` (up from 0.3)
- **Reward**: unchanged from v9 (distance_to_target + shaped + success_bonus
  + action_rate_l2 + joint_pos_limits, same weights and curriculum)

## Run log

### v12 — First residual IK run

- **Date**: 2026-07-23
- **Instance**: vast.ai RTX 3090 (resumed)
- **Command**: `uv run python scripts/train.py SoArm100-Reach --env.scene.num-envs=4096`
- **W&B run**: _(to be filled after launch)_
- **Result**: _(to be filled after run completes)_

| Metric | v9 (best non-residual) | v12 (residual IK) |
|---|---|---|
| `episode_success` | ~30% (avg), 75% (peak) | _TBD_ |
| `position_error` | 0.03–0.04 m | _TBD_ |
| `orientation_error` | 0.7 rad | _TBD_ |
| `mean_reward` | +59 | _TBD_ |

### Notes

- _(observations during the run will be logged here)_

---

## Open questions / next steps (after v12)

- If v12 clears the ≥90% bar: promote, publish to HF Hub, proceed to Phase 5.
- If v12 improves but doesn't clear the bar:
  - Try `orientation_weight > 0` on the base controller (let IK also steer
    orientation, not just position)
  - Try two-scale residual (coarse + fine, like the two-scale reward shaping
    open lead from v9)
  - Try reducing `max_dq` (slower base controller → more work for the policy
    → more learning, but also more control authority for the residual)
  - Try adding observation history (frame stacking, open lead #3 from v9)
- If v12 is worse than v9:
  - Check whether the base IK controller is actually moving the arm (a
    Jacobian singularity or frame-resolution bug would make `dq_base=0`
    and the policy is back to doing everything, but with a 10× smaller
    action scale)
  - Check whether `orientation_weight=0` on the base controller causes the
    arm to reach the target position but with a wildly wrong orientation
    that somehow interferes with position accuracy