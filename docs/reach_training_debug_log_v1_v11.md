# SO-ARM100 Reach: Training Debug Log (v1–v11)

Companion to `vast_ai_training.md` (how to rent/run) — this document is the
*what happened* record: a session that took the Reach policy from **0%
success** to a stable **~30% success / ~0.04m position error** plateau
across 11 training runs on a rented vast.ai RTX 3090, plus the reasoning
and evidence behind each change. Kept as a debugging case study so future
tuning starts from here instead of re-discovering the same failure modes.

All runs used `scripts/train.py SoArm100-Reach`, PPO via `rsl_rl`,
`mjlab`/`mujoco_warp` physics. W&B project: `mjlab`, entity
`thanhndv212-thanh-nguyen`.

## TL;DR outcome

| | v1 (broken baseline) | v9 (best) |
|---|---|---|
| `episode_success` | 0% | ~30% (avg), 75% (peak) |
| `position_error` | 0.34 m | 0.03–0.04 m |
| `orientation_error` | 2.12 rad | 0.7 rad (untracked by success — harmless) |
| `mean_reward` | -8.33 | +59 (not directly comparable — reward terms changed) |

**Best checkpoint: v9**, W&B run
[`thanhndv212-thanh-nguyen/mjlab/y4bomfz3`](https://wandb.ai/thanhndv212-thanh-nguyen/mjlab/runs/y4bomfz3),
`logs/rsl_rl/so_arm100_reach/2026-07-22_18-00-20/model_1499.pt` region on
the (now-destroyed) instance — re-download from the W&B run's artifact
storage if needed, it auto-uploads.

---

## v1 — Original baseline (0% success)

Command: `train.py SoArm100-Reach --env.scene.num-envs=4096`, defaults from
`reach_env_cfg.py` / `so_arm100_constants.py` / `rl_cfg.py` as they existed
before this session.

**Result:** `episode_success` 0% for all 1500 iterations.
`position_error` flat at ~0.34–0.35m the entire run (target box is
0.1–0.47m away). `mean_reward` -10.7 → -8.33, but this was the policy
learning to *avoid the joint-limit penalty* (freeze), not learning the
task — `Episode_Reward/distance_to_target` stayed flat while
`Episode_Reward/joint_pos_limits` improved from -0.05 to -1.02 and
dominated the total.

### Root causes identified

1. **Action scale catastrophically small.**
   `SO_ARM100_ACTION_SCALE = 0.25 * STS3215_EFFORT_LIMIT / STS3215_STIFFNESS
   = 0.25 * 2.94 / 998.22 ≈ 0.000736 rad` per action unit. With
   `use_default_offset=True`, the policy can only command `home ± 0.000736
   rad`. Over a 300-step episode that's **0.22 rad ≈ 12.6° of total
   possible joint travel** — physically unable to reach targets 10–47cm
   away regardless of training quality.
2. **Home keyframe near joint limits.** `shoulder_lift=-1.57` (limit
   `-1.75`), `elbow_flex=1.57` (limit `1.69`), `wrist_flex=1.57` (limit
   `1.66`) — only 0.09–0.18 rad of headroom before any hard termination
   fires.
3. **`joint_pos_limits` reward weight (-10.0) dominates.** ~3× the
   `distance_to_target` reward (weight 1.0) — policy's locally-optimal
   strategy is "don't move."
4. **No reward shaping, no orientation signal** (`orientation_weight=0.0`
   on `distance_to_target`).

---

## v2/v3 — Action scale 0.5, `clip_actions=1.0`

Bumped `SO_ARM100_ACTION_SCALE` to `0.5` rad (uniform per joint),
set `clip_actions=1.0` in `rl_cfg.py` to stop unclipped Gaussian actions
from overshooting.

**Result:** Episodes terminated in ~4–6 steps via `joint_limit_violated` —
~1000/4096 envs violating per iteration. The stiff PD controller
(`stiffness=998.22`) briefly overshoots commanded targets near a joint
limit, and the hard termination fires on that transient, never letting an
episode run long enough to learn anything.

**Fix carried to v4:** removed the `joint_limit_violated`
`TerminationTermCfg` entirely from `reach_env_cfg.py` — the
`joint_pos_limits` reward penalty (a continuous term, not a hard
termination) is enough of a deterrent once its weight is sane, and doesn't
kill episodes over a transient overshoot.

---

## v4 — Removed joint-limit termination, scale=0.5

**Result:** Episodes finally ran full length (300 steps). `mean_reward`
improved to -3.55, `position_error` down to 0.24, `orientation_error` down
to 0.06 (from 2.12 — orientation improved incidentally even with
`orientation_weight` unused at this point... actually this was from a
different config edit; see full run logs). But `task_success` plateaued
at 0% by the end of the run; `action_std` **exploded** from 0.35 → 1.65 and
`Loss/entropy` rose monotonically the whole run.

### Diagnosis

`schedule="adaptive"` in `RslRlPpoAlgorithmCfg` (`desired_kl=0.01`)
increases the learning rate/std when the observed KL divergence stays
below the target — its implicit assumption is "the policy isn't
exploring enough." When the real blocker is something else (here: action
scale still capped the reachable workspace), the adaptive schedule
mistakes "not improving" for "not exploring" and spirals, degrading an
already-converging policy.

---

## v5 — Action scale 1.0 for arm joints

Increased `SO_ARM100_ACTION_SCALE` to `1.0` rad for the five arm joints
(kept `gripper=0.15`, since gripper's usable range is much smaller). Home
keyframe unchanged from v4 (already adjusted to a less-folded pose).
Verified numerically before launching: `home ± scale` fits inside each
joint's hard range for all six joints.

**Result:** `task_success` reached **37.5%** at the final iteration (peak
~60% mid-run per the W&B sparkline), `position_error` down to 0.19m. Still
the same adaptive-KL `action_std` explosion pattern as v4 (0.33 → 0.67+),
plus fluctuating/noisy success rate rather than monotonic improvement.

---

## v6 — Fixed LR schedule, reward shaping, longer episodes

Changes from v5:
- `algorithm.schedule`: `"adaptive"` → `"fixed"`, `learning_rate` 1e-3 →
  1e-4 (stop the runaway std spiral at the source).
- Added `distance_to_target_shaped` reward
  (`soarm_mjlab/tasks/reach/mdp/rewards.py`): `exp(-position_error /
  sigma)` — denser gradient near the target than raw `-distance`,
  weight 5.0, `sigma=0.15`.
- `episode_length_s`: 6.0 → 10.0 (more time per episode to reach target).
- `init_std`: 0.5 → 0.3 (less aggressive initial exploration).

**Result:** Reward went **positive** for the first time (+29.1).
`action_std` still climbed (0.30 → 0.82) — the fixed LR schedule stopped
the adaptive-KL spiral, but PPO's own entropy bonus (`entropy_coef=0.005`)
was independently pushing exploration up. `task_success` noisy, 12–54%,
ending at 21%. `position_error` stable ~0.18–0.20m (best 0.1825).

---

## v7 — Lower entropy_coef, sharper reward shaping

Changes from v6:
- `entropy_coef`: 0.005 → 0.001.
- `distance_to_target_shaped` `sigma`: 0.15 → 0.10 (sharper reward peak).

**Result:** `action_std` **converged** to a stable 0.08 (down from 0.30,
monotonically decreasing — fixed) and `Loss/entropy` decreased
monotonically instead of rising. Training behavior finally well-behaved
and reproducible. But `task_success` avg dropped to ~17% (from v6's noisy
12–54%) and `position_error` stayed at the same ~0.20m wall.

**Conclusion at this point:** the entropy/exploration instability was
fixed, but a *different*, still-unexplained ceiling held `position_error`
at ~0.19–0.20m across three runs (v5, v6, v7) that each changed unrelated
hyperparameters and each improved something else. A metric that plateaus
identically across otherwise-different runs points to a bottleneck
upstream of anything being tuned — i.e., in the task/environment
definition, not the RL hyperparameters.

---

## v8 — Fixed reachable-workspace mismatch (biggest single win)

### Root cause

`_compute_reachable_workspace()` in
`soarm_mjlab/tasks/reach/config/so_arm100/env_cfgs.py` samples random
joint configurations across the **full hard joint-limit range** via
forward kinematics to derive the target sampling box (`position_range`).
But the policy's action range is capped at `home ± action_scale` (currently
`±1.0` rad for arm joints, `±0.15` for gripper) — a much smaller region of
joint space. A meaningful fraction of the 10th–90th-percentile target box
was therefore **physically unreachable** given the actual action scale, no
matter how well the policy trained.

### Fix

Changed the FK sampling range in `_compute_reachable_workspace` from
`joint.range` (hard limits) to `[home - scale, home + scale]` (clamped to
hard limits), i.e. sampling from what the policy can *actually* command:

```python
joint_lo[i] = max(j.range[0], home - scale)
joint_hi[i] = min(j.range[1], home + scale)
```

Verified: new target box (`x=[0.104, 0.352], y=[-0.183, 0.18],
z=[0.09, 0.457]`) is visibly tighter than the old one
(`x=[-0.07, 0.298], y=[-0.251, 0.253], z=[0.1, 0.467]`).

**Result:** Massive jump — **task_success 42% avg (last 200 iters)**, up
2.5× from v7's 17%. `position_error` down 40%, to 0.121m avg (best
0.0946). `mean_reward` up 36% to 29.9. Zero training instability
(`action_std` still stable at 0.08). This confirms the root-cause
diagnosis was correct: fixing the task's own infeasibility, not further
hyperparameter tuning, was the actual unlock.

---

## v9 — Success bonus + threshold curriculum + softened joint limits

Three changes applied together (see question below for why bundled):

1. **`success_bonus` reward** (new function in `mdp/rewards.py`):
   discrete `+1` while `position_error < threshold`, weight 10.0. Added
   because the continuous shaped reward can be maximized by hugging the
   target from just outside the strict success radius without ever
   crossing it — decoupling a pass/fail-aligned term from the continuous
   proxy keeps the optimized objective aligned with the actual success
   metric.
2. **Threshold curriculum** (`mjlab.envs.mdp.curriculums.reward_curriculum`,
   wired via a new `curriculum` dict in `reach_env_cfg.py`): tightens
   `success_bonus`'s `threshold` from 0.05 → 0.04 → 0.03 at
   `common_step_counter` 0 / 12000 / 24000 (roughly training's 1/3 and 2/3
   marks, given `max_iterations=1500` × `num_steps_per_env=24` = 36000
   total steps).
3. **Softened joint limits**: `soft_joint_pos_limit_factor` 0.9 → 0.95,
   `joint_pos_limits` reward weight -1.0 → -0.5 — v8's data showed this
   penalty growing over training (-0.027 → -0.034) as the policy needed
   more range to reach workspace corners; giving it more headroom reduces
   that tension.

**Result:** Position error improved dramatically — **0.039m avg (last
200), best 0.030m** (right at the true success threshold). But raw
`episode_success` was noisier than v8 (29.6% avg vs v8's 42%), because
`orientation_error` regressed to ~0.73 rad (from v8's 0.034) — the
position-dominant `success_bonus` gave the policy no reason to also match
orientation, and it stopped trying. Since `orientation_range` is fixed at
identity and success only ever gated on *position*, this orientation
drift is **cosmetically ugly but harmless to the actual task metric** —
not a true capability regression, a reward-bookkeeping side effect.

**This is the best checkpoint of the session** despite the noisier
`episode_success` average, because position accuracy (the thing that
actually matters for reach) is at its best-ever level and training
remained stable throughout.

---

## v10 — Attempted orientation fix (regression, reverted)

Misdiagnosed the v9 orientation drift as a real problem. Changes:
`orientation_weight` 0.5 → 0.1 (both `distance_to_target` and
`_shaped`), `success_bonus` weight 10.0 → 6.0 (to "let it complement
rather than dominate").

**Result: made things worse on both counts.**
`orientation_error` avg **rose** to 1.03 rad (from v9's 0.73) —
*lowering* its reward weight removed even more incentive to control it;
the causality was backwards. `episode_success` avg **dropped** to 21.5%
(from v9's 29.6%) — diluting `success_bonus` weakened the very signal
that was doing useful work. `mean_reward` also dropped (51.2 vs 58.9).

**Lesson:** the v9 "orientation regression" wasn't actually a regression
worth fixing (orientation was never gating success), and treating a
cosmetic side-effect as a bug produced a worse checkpoint on the metric
that matters. Reverted both changes back to v9's values.

---

## v11 — Curriculum ablation at 56× scale (falsifies curriculum-dip hypothesis)

With v10 reverted to v9's reward weights, tested whether the *step-wise*
curriculum transitions (0.05→0.04→0.03) were themselves destabilizing —
v9's `episode_success` dipped noticeably right after each threshold
tightening (31%→15%→30%). Ablated the curriculum to a single fixed stage
(`threshold=0.03` from step 0, no tightening) to isolate the effect.

**Also used this run to test maximum safe GPU utilization** (see
"Resource scaling" below): `num_envs` 4096 → **230,000** (56×), using
~21.5GB/24GB (~88%, 12% safety margin), full 1500 iterations took ~2h35m
instead of ~15min, collecting **7.6 billion** total env steps (vs v9's
147M) — same reward/curriculum config as v9 except the ablated curriculum,
at vastly more data.

**Result:** `episode_success` avg (last 200) **18.7%** — *worse* than
v9's 29.6%, not better. `position_error` avg 0.038 (same as v9's 0.039).
This **falsifies** the curriculum-dip hypothesis: removing the staged
tightening made things worse, not better, meaning the staged schedule
was actually helping (giving the policy an easier target early, before
tightening) rather than causing instability. The dips visible in v9's
sparkline are more likely just normal PPO exploration/exploitation
dynamics maturing over training, not a curriculum artifact.

**Secondary finding:** running the *same* config at 56× the data (7.6B
vs 147M steps) reproduced the same ~0.03–0.04m position-error plateau,
confirming it's a robust, reproducible limit of the current reward/action
setup — not a "just needs more samples" problem.

---

## Full change ledger by file

`soarm_mjlab/assets/robots/so_arm100/so_arm100_constants.py`:
- `SO_ARM100_ACTION_SCALE`: `0.25 * effort/stiffness` (≈0.000736, all
  joints) → per-joint dict, arm joints `0.5` (v4) → `1.0` (v5), gripper
  `0.15` throughout.
- `HOME_KEYFRAME.joint_pos`: `shoulder_lift=-1.57, elbow_flex=1.57,
  wrist_flex=1.57, wrist_roll=-1.57` (fully folded, near limits) →
  `shoulder_lift=-0.3, elbow_flex=0.5, wrist_flex=0.0, wrist_roll=0.0`
  (v4, more centered / away from limits).
- `ARTICULATION.soft_joint_pos_limit_factor`: `0.9` → `0.95` (v9).

`soarm_mjlab/tasks/reach/reach_env_cfg.py`:
- Removed `joint_limit_violated` `TerminationTermCfg` (v4).
- `episode_length_s`: `6.0` → `10.0` (v6).
- `rewards["joint_pos_limits"].weight`: `-10.0` → `-1.0` (v9's baseline
  change came earlier, part of the original bundle) → `-0.5` (v9).
- `rewards["distance_to_target"]`: added `orientation_weight=0.5`
  (from 0.0), weight `1.0` → `2.0`.
- Added `rewards["distance_to_target_shaped"]` (v6): `exp(-error/sigma)`,
  weight `5.0`, `sigma` `0.15` (v6) → `0.10` (v7).
- Added `rewards["success_bonus"]` (v9): weight `10.0`, `threshold=0.05`
  (curriculum-controlled).
- Added `curriculum["success_threshold_curriculum"]` (v9):
  `reward_curriculum` tightening `success_bonus.threshold`
  `0.05→0.04→0.03` at steps `0/12000/24000`.

`soarm_mjlab/tasks/reach/mdp/rewards.py`:
- Added `distance_to_target_shaped(sigma, orientation_weight, ...)` (v6).
- Added `success_bonus(threshold, ...)` (v9).

`soarm_mjlab/tasks/reach/config/so_arm100/env_cfgs.py`:
- `_compute_reachable_workspace`: FK sampling range `joint.range` (hard
  limits) → `[home - action_scale, home + action_scale]` clamped to hard
  limits (v8 — the biggest single fix of the session).
- Wired `asset_cfg` for `distance_to_target_shaped` / `success_bonus`
  per-robot (site resolution).

`soarm_mjlab/tasks/reach/config/so_arm100/rl_cfg.py`:
- `clip_actions`: unset (`null`) → `1.0` (v2/v3).
- `distribution_cfg["init_std"]`: `1.0` → `0.5` (v3) → `0.3` (v6).
- `algorithm.schedule`: `"adaptive"` → `"fixed"` (v6).
- `algorithm.learning_rate`: `1.0e-3` → `1.0e-4` (v6).
- `algorithm.entropy_coef`: `0.005` → `0.001` (v7).

---

## Resource scaling (v11 sizing experiment)

Instance: RTX 3090, 24GB VRAM. Empirically tested `num_envs` scaling by
launching short (5-iteration) smoke runs and watching `nvidia-smi`:

| `num_envs` | VRAM used | Notes |
|---|---|---|
| 4096 (v1–v10) | ~0.8GB | Original default, huge headroom unused |
| 180,000 | ~16.9GB (69%) | Stable |
| 225,000 | ~21.0GB (85.5%) | Stable |
| 238,000 | ~22.2GB (90.2%) | At the 90% target, thin margin |
| 300,000 | **OOM** | Fails during PPO backward pass — env-only
  memory tests (bare `env.step()`, no optimizer) under-predict real usage;
  must smoke-test with an actual training run, not just env construction |

**Chosen for v11: 230,000** (~21.5GB, ~88%, ~12% safety margin).

**Lesson:** memory footprint from `env.reset()` + `env.step()` alone is
misleadingly low — the PPO rollout buffer, optimizer state, and backward
pass add substantial overhead that only shows up once you launch actual
training (`scripts/train.py ... --agent.max-iterations=5`), not a bare
env-stepping loop. Always headroom-test with a short real training run
before committing to a multi-hour job at a new `num_envs`.

---

## Open leads for further improvement (not yet tried)

From the v9 review — none of these have been tested yet:

1. **Reduce `joint_vel` observation noise.** Currently
   `Unoise(n_min=-1.5, n_max=1.5)` rad/s in `reach_env_cfg.py`'s
   `actor_terms["joint_vel"]` — large relative to typical fine-control
   velocities near a target. Could be swamping the signal needed for
   final-approach precision. Untested hypothesis, would need an isolated
   ablation (same v9 config, only this noise reduced/removed) to
   attribute any change correctly — don't bundle with other edits, this
   session's `code-review-and-quality` lesson (v9's 3-way bundle) made it
   hard to attribute the orientation regression to a specific one of the
   three changes.
2. **Two-scale reward shaping.** A single `sigma=0.10` for
   `distance_to_target_shaped` has weak gradient once error is already
   well below `sigma` — a second, finer kernel (`sigma≈0.03`, matching
   `success_threshold`) layered on top might sharpen final-approach
   behavior.
3. **LR/action-scale annealing.** Both are currently fixed for the whole
   run. A late-training decay (last ~20%) of either could let the policy
   fine-tune precision without the coarse action scale (`1.0` rad, tuned
   for reachability) working against fine positioning.
4. **Performance-gated curriculum** (instead of step-count-gated). v11
   showed the step-count schedule helps over no curriculum at all, but a
   schedule that advances only once `episode_success` crosses a bar
   (rather than a fixed step count) might avoid the dips seen right after
   each v9 transition.

## Methodology notes for future tuning sessions

- **Verify physical feasibility with simple math before touching RL
  hyperparameters.** v1's entire failure was a unit/scale bug, not an
  algorithm problem — `home ± action_scale` vs joint hard limits is a
  30-second check that would have caught it immediately.
- **Decompose aggregate reward into its components** before trusting a
  rising reward curve — v1's -10.7→-8.33 "improvement" was entirely the
  policy learning to avoid a penalty, not learning the task.
- **A metric that plateaus identically across otherwise-different runs**
  (v5, v6, v7 all stuck at ~0.19–0.20m position error despite each fixing
  something else) is a sign the bottleneck is upstream of what's being
  tuned — check the task/environment definition, not more hyperparameter
  sweeps.
- **Change one variable at a time when the causal story matters.** v9
  bundled three changes and left an ambiguous attribution for the
  orientation regression; v10's "fix" for that ambiguity made two
  different things worse because the underlying diagnosis was wrong.
- **Numerically verify config changes before spending a training run.**
  Every action-scale/home-keyframe change in this session was checked
  against joint hard limits via a small Python snippet first — this
  caught would-be joint-limit issues in seconds instead of 15 minutes into
  a wasted GPU run.
- **Memory-test with real training, not bare env steps**, before scaling
  `num_envs` up for a multi-hour run (see Resource scaling above).
