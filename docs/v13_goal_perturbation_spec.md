# v13 Spec: Task-Space Goal-Perturbation Residual (Option B)

**Status:** Implementation in progress
**Goal:** Fix the v12 degradation — PPO actively unlearning a near-optimal IK base
controller — by reformulating the residual from **additive joint-space** (v12,
Option A) to **goal-perturbation task-space** (v13, Option B).

## The problem with v12 (Option A)

v12's `ResidualIKAction` uses:

```
q_target = q_current + dq_base_ik + dq_residual   (joint-space additive)
```

where `dq_base_ik` is a DLS IK step toward the target pose and `dq_residual` is
the policy's per-joint output (±0.1 rad). Near the target, `dq_base ≈ 0` (error
is small) but `dq_residual` is not small relative to it → the residual kicks the
arm off target → the base controller corrects back → the residual kicks again →
**oscillation/shake**. PPO's only reward-improving direction is "shrink the
residual" (reducing the shake), which degrades positioning — the local optimum
where the residual is adversarial by construction.

## The fix: Option B (goal perturbation)

```
commanded_pose = target_pose + residual_pose_delta   (task-space)
dq = DLS_IK(commanded_pose, current_pose)             (IK tracks the perturbed goal)
q_target = q_current + dq                             (NO separate joint residual)
```

The residual **shifts the goal the IK aims at**; the IK just tracks it. The
residual no longer *fights* the controller — it **steers** it. Near the target:
the residual moves the equilibrium point, the IK settles there smoothly. **No
shake by construction.**

### Why this also fixes the degradation

- **Cleaner reward gradient:** the residual's effect on `distance_to_target` is
  direct — shifting the commanded goal moves the EE toward the true target. The
  reward signal is no longer buried under the base controller's reactive
  correction.
- **`action_rate_l2` becomes well-posed:** it penalizes *goal jitter* (genuinely
  undesirable), not the residual fighting the base.
- **The residual's job is a smooth, learnable bias:** compensate for IK's
  systematic reaching error. A small, well-defined function.

### Critical wiring: two targets

- `true_target` — the command manager's `ee_pose` (fixed per episode). The
  **reward** measures distance to this.
- `commanded_pose = true_target + residual` — what the IK aims at. The **action
  term** uses this internally.
- The reward functions already read from the command manager (true target), so
  **no reward changes needed** — the two-target wiring is entirely inside the
  action term.

## Implementation changes

### 1. `mdp/actions.py` — `ResidualIKActionCfg` + `ResidualIKAction`

**Config additions:**
- `residual_mode: Literal["joint", "goal"] = "joint"` — `"joint"` = v12
  (additive, per-joint), `"goal"` = v13 (goal perturbation, task-space).
- `residual_pos_scale: float = 0.02` — position delta scale (meters) in goal mode.
- `residual_rot_scale: float = 0.05` — orientation delta scale (rad) in goal mode.

**`__init__` changes:**
- In `"goal"` mode: `action_dim = 6` (3 pos + 3 rot), regardless of num_joints.
  `_raw_actions` buffer is `(num_envs, 6)`.
- Store `_residual_pos_scale` and `_residual_rot_scale` as floats.
- `_residual_scale` (used by ONNX metadata) becomes
  `[pos_scale, rot_scale]` in goal mode.

**`apply_actions` changes (goal mode branch):**
1. Get target pose (world frame) — same as v12.
2. `residual_pos = action[:, :3] * residual_pos_scale`
3. `residual_rot = action[:, 3:6] * residual_rot_scale`
4. `commanded_pos_w = target_pos_w + residual_pos`
5. `commanded_quat_w = quat_box_plus(target_quat_w, residual_rot)` — SO(3)
   box-plus composes the target quaternion with the residual rotation vector.
6. Compute pose error: current EE → `commanded_pose` (NOT `target_pose`).
7. DLS solve → `dq_base` (same machinery, error is to commanded pose).
8. `q_target = q_current + dq_base` — **NO joint residual added**.
9. Clamp, frozen joints, set_joint_position_target — same as v12.

**New import:** `quat_box_plus` from `mjlab.utils.lab_api.math`.

### 2. `so_arm100_constants.py`

- `SO_ARM100_RESIDUAL_POS_SCALE: float = 0.02` — 2 cm max position perturbation.
- `SO_ARM100_RESIDUAL_ROT_SCALE: float = 0.05` — ~2.9° max orientation perturbation.

### 3. `config/so_arm100/env_cfgs.py`

- `joint_pos_action.residual_mode = "goal"`
- `joint_pos_action.residual_pos_scale = SO_ARM100_RESIDUAL_POS_SCALE`
- `joint_pos_action.residual_rot_scale = SO_ARM100_RESIDUAL_ROT_SCALE`
- Remove the per-joint `residual_scale` dict (not used in goal mode).

### 4. `rl/runner.py` — `_get_reach_metadata`

- In goal mode: `action_scale` = `[pos_scale, rot_scale]`,
  `action_space` = `"residual_ik_goal"`.

### 5. Tests

- `test_asset_so_arm100.py`: update assertions for goal mode (action_dim=6,
  residual_mode="goal", no per-joint residual_scale dict).
- `test_env_reach.py`: uses `action_dim` dynamically — no change needed.

## What stays the same

- The DLS IK base controller (Jacobian, DLS normal equations, damping, max_dq).
- Frame resolution, Jacobian computation (`mujoco_warp.jac`).
- The reward stack (distance_to_target, shaped, success_bonus, action_rate_l2,
  joint_pos_limits, curriculum) — all unchanged.
- PPO hyperparameters (init_std=0.5, lr=1e-4, entropy_coef=0.001, etc.).
- The `frozen_joints` play/deploy feature.
- The workspace computation (full joint-limit range — the base IK controller
  can reach the full workspace).

## v13 training plan

- **Instance:** resume stopped vast.ai instance `45629304` (RTX 3090).
- **Command:** `uv run python scripts/train.py SoArm100-Reach --env.scene.num-envs=4096`
- **Duration:** 1500 iterations, ~14 min.
- **Success criteria:**
  - No shake near target (visual inspection via play viewer after training).
  - `episode_success` does NOT degrade from the iter-1–3 baseline (93–98%) —
    i.e., training preserves or improves the base controller's performance.
  - `position_error` stays at or below 0.018 m (the untrained base level).
  - Stretch: `episode_success` ≥ 90% (clearing the promotion bar).
- **W&B:** new run in project `mjlab`, entity `thanhndv212-thanh-nguyen`.

## Risk / fallback

If Option B also degrades (unlikely given the mechanism, but possible):
- The `residual_mode="joint"` config is still available — revert to v12 with
  `residual_mode="joint"` and proceed with the v13 ablation plan (action_rate_l2,
  dq_base/dq_residual logging, etc.) from the debug log.
- Option B can't express dynamic corrections (oscillating to break friction) —
  irrelevant for non-contact reach, would matter for insertion/contact.