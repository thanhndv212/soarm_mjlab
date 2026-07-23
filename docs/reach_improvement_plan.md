# Reach Training — Improvement Plan Spec

**Status:** Proposed (not yet scheduled)
**Scope:** Problem-formulation reforms for the SO-ARM100 Reach task, upstream of PPO hyperparameter tuning.
**Author context:** Derived from the v1–v12 training diagnosis (see `docs/reach_training_debug_log.md`).

---

## 0. The diagnosis this plan responds to

The ~0.04 m position-error plateau is **not** an optimization ceiling — it is a
**problem-formulation ceiling**. Evidence:

- The plateau reproduces at **56× the data** (v11: `num_envs=230,000`, 7.6 B env
  steps) with the *same* ~0.03–0.04 m error. More samples does not move it.
- The plateau reproduces across **different hyperparameters** (v5/v6/v7) — same
  floor regardless of LR / action-scale / reward-weight choices.
- The single biggest win so far (v8, **+2.5× success**) was an **environment-
  definition fix** (reachable-workspace sampling), *not* a training fix.

Conclusion: pushing the optimizer harder against the current formulation has
diminishing returns. The leverage is in **reformulating the problem so the
ceiling is higher** — action space, observation space, curriculum, and sim
fidelity — not in tuning the optimizer against the current one.

This spec collects the investable reforms, ranked by expected leverage, all
upstream of "tune PPO harder."

---

## 1. Residual RL over a differential-IK base controller — *highest leverage*

### Reformulation
Instead of learning the full joint-angle → EE-pose mapping, implement a
**Jacobian / differential-IK proportional controller** as the base and learn
only a **residual** on top:

```
action = IK_proportional(target_pose) + policy_residual
```

- The base controller analytically solves "get roughly there" — the hard part
  (6-DOF inverse kinematics via DLS Jacobian).
- The policy learns only the correction — the easy part (compensating for IK
  imperfections, joint limits, dynamic effects).

This shrinks the learning problem by orders of magnitude: the policy fits a
small correction function instead of the entire reaching manifold.

### Why investable now
- `m5teleop/ik_solver.py` already implements differential IK via `pink` +
  `pinocchio`. The infrastructure exists in this workspace — reuse, not build.
- `soarm_mjlab` already has a working `ResidualIKAction` (v12): DLS IK base
  (`mujoco_warp.jac`) + joint-space residual, `q_target = q_current + dq_base_ik
  + dq_residual`. The base controller *untrained* already hits 94–98% success /
  0.017–0.018 m — the reformulation works; the open problem is why PPO degrades
  it (see `docs/reach_training_debug_log.md` v12 "Open questions / v13 ablation plan").

### Why it fits the diagnosis
The plateau is "the policy gets to ~0.04 m but not closer." A proportional IK
base gets to ~0.04 m *by construction*; the residual only needs the last-mile
correction, a much smaller function to fit.

### Implementation status (as of v12)

**Implemented:**
- `ResidualIKAction` / `ResidualIKActionCfg` in `mdp/actions.py` — GPU-vectorized
  DLS IK base (`mujoco_warp.jac`) + **joint-space** residual,
  `q_target = q_current + dq_base_ik + dq_residual`. Reuses mjlab's
  `DifferentialIKAction` Jacobian/DLS machinery.
- Wired into `reach_env_cfg.py` (action term swapped from `JointPositionActionCfg`);
  per-robot params (`frame_name="gripperframe"`, `residual_scale`) in
  `config/so_arm100/env_cfgs.py`; `init_std` 0.3 → 0.5 in `rl_cfg.py`;
  `SO_ARM100_RESIDUAL_SCALE` added to `so_arm100_constants.py`.
- ONNX export bug fixed in `runner.py` (commit `f55d8b9`):
  `mjlab.rl.exporter_utils.get_base_metadata` hard-asserts `JointPositionAction`;
  reimplemented as `_get_reach_metadata` reading `ResidualIKAction._residual_scale`.
- Diagnostic play-viewer additions (EE pose frames, reachable-workspace box,
  frozen closed gripper via `frozen_joints`) to support v13 debugging.

**v12 result** (W&B [`0vzygq3k`](https://wandb.ai/thanhndv212-thanh-nguyen/mjlab/runs/0vzygq3k),
4096 envs, 1500 iters, ~14 min):

| Point in training | `episode_success` | `position_error` |
|---|---|---|
| Iteration 1–3 (base controller dominates) | **93.8–97.5%** | **0.017–0.018 m** |
| Iteration 1300–1500 (final, last-200 avg) | 38.3% | 0.049 m |

- **Hypothesis NOT confirmed:** the *untrained* base controller alone already
  beats v9's best trained checkpoint (30% avg / 75% peak, 0.03–0.04 m). Training
  actively *degrades* a near-optimal starting point — `action_std` drops
  0.50 → 0.25–0.28 while `position_error` worsens. The problem reframed from
  "learn a good residual" to "**why does PPO degrade the base controller**."
- Full detail in `docs/reach_training_debug_log.md` (v12 section).

### Open sub-questions / v13 ablation plan (from v12)

1. **Reduce/disable `action_rate_l2`** (weight −0.01) — strongest clue:
   `action_std` drops 0.50 → 0.25–0.28 while `position_error` worsens; the policy
   learns to move *less*, not more accurately. Isolates hypothesis #1 (PPO
   optimizes the wrong signal — the residual's marginal effect on
   `distance_to_target` is small/noisy vs the `action_rate_l2` penalty).
2. **Log `dq_base` and `dq_residual` magnitudes separately** (obs-manager
   metrics) — directly confirms/refutes hypothesis #2 (base/residual fighting,
   residual shrinking to zero). Do *before* another full run.
3. **Try `orientation_weight > 0`** on the base controller — tests hypothesis #3
   (orientation competition; `orientation_error` ~1.2 rad vs v9's 0.7 rad).
4. **Try smaller `max_dq`** (0.1–0.2, down from 0.5) — tests hypothesis #4 (base
   too aggressive, residual can't compensate).
5. **If none explain it:** compare `ResidualIKAction.dq_base` vs
   `DifferentialIKAction.compute_dq` on the same pose error to rule out an
   implementation bug.

---

## 2. Action space: task-space (Cartesian) commands

### Reformulation
The policy commands **EE velocity/position in world frame**, not joint angles.
An IK layer (same machinery as #1) maps to joints. The policy learns in the
space where the task lives — "move toward the target" is a simple direction in
Cartesian space, a complex nonlinear mapping in joint space.

### Current problem
The action scale (±1.0 rad/joint) was tuned for *reachability* (v5), but that
same coarse scale works *against* fine final-approach positioning — the policy
reaches the neighborhood but cannot make sub-centimeter corrections because each
action unit swings a joint by a full radian. Task-space actions decouple
"reachability" (IK handles it) from "precision" (Cartesian commands are
naturally fine-grained).

### Pairs with #1
Residual RL in task space = "learn a small Cartesian correction on top of an IK
base." This is the standard formulation for manipulation RL that works in
practice.

### Status
**Not started.** The v12 `ResidualIKAction` is a *joint-space* residual on top of
an IK base — it does **not** yet command EE pose/velocity in task space (the §2
reformulation). The task-space variant is a distinct, unbuilt reformulation that
pairs with the §1 base controller; it remains the higher-leverage half of
priority #1 once the v12 degradation is understood.

---

## 3. Observation: action / proprioception history (frame stacking)

### Reformulation
Give the policy its **last N actions and observations** (e.g. a 3–5 step window),
not just the current step. This lets the policy infer velocity and acceleration
implicitly, without relying on the noisy `joint_vel` observation (±1.5 rad/s
noise — flagged as an open lead in the v1–v11 log).

### Why cheap and high-impact
A frame-stacking observation is a small change in the obs config. It gives the
policy temporal context for free — PPO with only current-step observations is
partially blind to its own dynamics. **Lowest-effort item on this list.**

### Status
**Not started.** Config-level change only; no code landed yet.

---

## 4. Curriculum on target *distance*, not just success threshold

### Reformulation
The current curriculum (v9) only tightens the *success threshold*
(0.05 → 0.04 → 0.03 m). Add a **target-distance curriculum**: start episodes
with targets close to the current EE pose, expand the sampling range as
`episode_success` improves. This shapes the actual *difficulty* of the task, not
just the pass/fail bar.

### Why it fits
v8 fixed the workspace to be *reachable*, but the full reachable box is still a
wide range of difficulties (a target 10 cm away is easy; 40 cm away requires
full joint travel). A distance curriculum lets the policy learn easy reaches
first and generalize outward — the standard manipulation-RL curriculum pattern.

### Status
**Not started.** The v9 curriculum only tightens the success threshold; the
target-distance dimension is not yet implemented.

---

## 5. Actuator fidelity for sim2real (invest now, pays in Phase 5)

### Reformulation
The current MJCF uses the STS3215 default-class gains from the upstream model.
Real STS3215 servos have **latency** (~serial bus at 1 Mbaud, ~20 ms round-trip),
**torque-speed curves** (torque drops at high speed), **backlash**
(3D-printed links), and **thermal derating**. If the sim doesn't model these, a
policy that achieves 90% in sim may collapse on hardware — and worse, the sim
policy may *exploit* sim artifacts (instant torque response, no backlash) that
don't exist on the real arm.

### Why invest now, not in Phase 5
System identification (measuring real servo step response and matching the sim
actuator model) is upstream work that also *improves sim training* — a more
accurate sim produces a policy that generalizes better, not just one that
deploys better. Doing this before the next training campaign means the next
campaign trains in a sim closer to reality.

### Concrete first step
Command a single real servo to a step input, log the position response at 50 Hz,
fit a first-order lag + deadband model, add it to the MJCF actuator. ~half a
day of work, reusable forever.

### Status
**Not started.** No system-ID measurements taken; MJCF still uses upstream
STS3215 default-class gains.

---

## 6. Demonstration-seeded initialization

### Reformulation
Use `m5teleop` to collect 20–50 reach trajectories (operator drives the arm to
targets), then **behavioral-clone** to initialize the policy before RL
fine-tuning. The policy starts from "roughly knows how to reach" instead of
random, and RL refines it.

### Why investable
The workspace already has the teleop + recording infrastructure
(`m5teleop --record`, `soarm_lerobot.TeleopRecorder`). Demonstrations don't need
to be optimal — just better than random init. This is the standard "RL from
demonstrations" pattern and is particularly effective for the "plateau that more
data doesn't break" failure mode: demonstrations break the plateau by changing
the *starting point*, not the exploration budget.

### Status
**Not started.** No demonstrations collected; BC init not wired into the training
entry point.

---

## Priority table

| Priority | Investment | Status | Effort | Expected impact |
|---|---|---|---|---|
| 1 | Residual RL over IK base controller (#1 + #2) | **#1 implemented — v12 ran; hypothesis NOT confirmed** (see §1) | 2–3 days | Likely the single biggest reformulation — turns "learn full reaching" into "learn small corrections" |
| 2 | Observation history / frame stacking (#3) | Not started | hours | Cheap; addresses the noisy-`joint_vel` problem structurally |
| 3 | Target-distance curriculum (#4) | Not started | half a day | Shapes difficulty, not just the pass/fail bar |
| 4 | Actuator fidelity / system ID (#5) | Not started | half a day | Improves both sim training quality and future sim2real |
| 5 | Demonstration-seeded init (#6) | Not started | 1 day (collect + BC) | Changes the starting point, not the exploration budget |

**Common thread:** the plateau is a *problem-formulation ceiling*, not an
*optimization ceiling*. v11 proved that (56× data, same plateau). The
highest-leverage investments reformulate the problem so the ceiling is higher,
rather than pushing harder against the current one.

---

## Suggested execution order

1. **Resolve the v12 degradation first** (cheap, diagnostic) — see
   `docs/reach_training_debug_log.md` v12 "Open questions / v13 ablation plan":
   reduce/disable
   `action_rate_l2`, log `dq_base`/`dq_residual` separately, try
   `orientation_weight > 0`, try smaller `max_dq`. This either fixes #1 or
   rules out the residual formulation's implementation bugs before building
   more on top of it.
2. **Cheap wins in parallel** — #3 (frame stacking) and #4 (distance curriculum)
   are config-level and can land alongside the v13 ablation.
3. **Then #1 + #2 combined** as the main reformulation campaign (residual RL in
   task space).
4. **#5 (actuator fidelity)** before the next full campaign so it trains in a
   more realistic sim.
5. **#6 (demo seeding)** as a later lever if the reformulated campaign still
   plateaus.

---

## Open decision for the user

Spec out **#1 (residual RL over the IK base)** as a concrete implementation plan
first, or start with the **cheap wins (#3 frame stacking, #4 distance
curriculum)** to bank easy gains while the v13 ablation runs?
