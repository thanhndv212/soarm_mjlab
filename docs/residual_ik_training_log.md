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

### v12 — First residual IK run: base controller alone beats the final trained policy

- **Date**: 2026-07-23
- **Instance**: vast.ai RTX 3090 (fresh rental, id `45629304` — the original
  instance's host GPU was preempted indefinitely; see "Instance notes" below)
- **Command**: `uv run python scripts/train.py SoArm100-Reach --env.scene.num-envs=4096`
- **W&B run**: [`thanhndv212-thanh-nguyen/mjlab/0vzygq3k`](https://wandb.ai/thanhndv212-thanh-nguyen/mjlab/runs/0vzygq3k)
- **Duration**: 1500/1500 iterations, ~14 minutes

**Headline result: the *untrained* base IK controller (iteration 1) already
outperforms the *fully trained* policy (iteration 1500) by a wide margin.**
Training makes the policy *worse*, monotonically, for most of the run:

| Point in training | `episode_success` | `position_error` | `orientation_error` |
|---|---|---|---|
| Iteration 0 (random init, residual ≈ noise) | 19.8% | 0.177 m | 0.85 rad |
| **Iteration 1–3** (base controller dominates, policy barely updated) | **93.8–97.5%** | **0.017–0.018 m** | 1.23–1.36 rad |
| Iteration 20 (peak) | 97.9% | 0.021 m | — |
| Iteration 100 | 44.7% | 0.047 m | — |
| Iteration 500 | 16.9% | 0.061 m | — |
| **Iteration 1300–1500 (final, last-200 avg)** | **38.3%** | **0.049 m** | **1.24 rad** |

Full run: `mean_reward` last-200 avg = 45.4 (peaked much higher early,
declined and partially recovered).

**This is worse than v9** (30% avg / 75% peak success, 0.03–0.04m error) on
every axis except being *faster to reach a similar plateau* (minutes not
hours) — and dramatically worse than what the base controller alone
achieves with zero training. The hypothesis that "learning a residual is a
smaller/easier problem" is **not confirmed by this run** — instead, the
policy actively unlearns a near-optimal starting point.

### Diagnosis: why does training make it worse?

The pre-launch smoke test already showed this same signature
(`--env.scene.num-envs=16 --agent.max-iterations=3`: `episode_success: 1.0`,
`position_error: 0.013–0.016m` by iteration 2, before any real learning) —
at the time this looked like confirmation of the hypothesis, but in
hindsight it's the same "base controller alone is already excellent"
result, just not yet showing the *degradation* that only appears with
enough iterations for the policy to actually move.

Leading hypotheses, in order of suspicion, **none yet tested**:

1. **PPO is optimizing the wrong signal.** The reward stack
   (`distance_to_target` + `_shaped` + `success_bonus`) was tuned in v6–v9
   for a policy that does *all* the positioning work itself. With the base
   IK controller already closing most of the position error, the
   *residual*'s marginal effect on `distance_to_target` is small and noisy
   relative to `action_rate_l2`'s penalty and PPO's own entropy-driven
   exploration — the optimizer may be climbing a much flatter, noisier
   reward surface than v9's, and finding directions that help
   `action_rate_l2`/entropy at the expense of the (already small) position
   contribution.
2. **The residual and base controller fight each other.** `dq_base` is
   recomputed fresh every step from the *current* pose error — if the
   residual perturbs the arm off the base controller's implicit trajectory,
   the base controller's next-step correction may partially undo the
   residual's intended effect, creating an oscillation/dither pattern that
   PPO's `action_rate_l2` penalty then suppresses by driving the residual
   toward zero *and* slightly off-target (consistent with `action_std`
   dropping from 0.50 to 0.25–0.28 over the run while `position_error`
   worsens — the policy is learning to move *less*, not more accurately).
3. **`orientation_weight=0.0` on the base controller may be actively
   harmful, not neutral.** `orientation_error` sits at ~1.2 rad throughout
   training (worse than v9's 0.7 rad) — the base controller does nothing to
   control it, and the reward's `orientation_weight=0.5` on
   `distance_to_target`/`_shaped` gives the *residual* an incentive to fix
   orientation that may compete with maintaining position accuracy, unlike
   v9 where the same policy controlled both jointly and could balance them.
4. **`max_dq=0.5` rad/step (uncapped by residual_scale) may make the base
   controller too aggressive**, causing it to overshoot/oscillate near the
   target in a way v9's smooth learned policy didn't, and the small
   residual (±0.1 rad) can't fully compensate.

### Comparison table

| Metric | v9 (best non-residual, final) | v12 (residual IK, iter 1–3) | v12 (residual IK, final, last-200 avg) |
|---|---|---|---|
| `episode_success` | ~30% (avg), 75% (peak) | **93.8–97.5%** | 38.3% |
| `position_error` | 0.03–0.04 m | **0.017–0.018 m** | 0.049 m |
| `orientation_error` | 0.7 rad | 1.23–1.36 rad | 1.24 rad |
| `mean_reward` | +59 | +5–7 (early, not comparable) | +45.4 |

The most important row is the middle one: **the untrained base controller
already beats v9's best trained checkpoint on position accuracy.** The
actual problem to solve now is "why does PPO training degrade this,"
not "how do we get the policy to learn a good residual" — the base
controller already provides one for free.

### Instance notes (relevant to future runs)

- The original instance (`45541933`, interruptible) could not resume after
  being preempted — `vastai start instance` queued with "could take
  anywhere from hours to weeks." Destroyed it and rented a fresh one instead
  of waiting; `soarm_mjlab` being self-contained (vendored MJCF/meshes, no
  monorepo dependency) makes this a ~2 minute clone+sync, not a big loss.
- The first replacement instance (`45627045`) had very low `disk_bw`
  despite a fast-looking network link — `uv sync --extra cu128` was stuck
  unpacking CUDA/torch wheels at ~1-5MB/s for 20+ minutes. Destroyed it and
  rented one with `disk_bw > 6GB/s` (`37930677` → contract `45629304`);
  the same `uv sync` completed in under a minute. See
  `docs/vast_ai_training.md`'s updated "Rent an instance" section.
- Found and fixed a real bug during the pre-launch smoke test: the ONNX
  export was silently failing every checkpoint save
  (`mjlab.rl.exporter_utils.get_base_metadata` hard-asserts
  `JointPositionAction`, which `ResidualIKAction` isn't). Fixed in
  `runner.py` (commit `f55d8b9`) before launching the real run — verified
  the fix produces a valid `.onnx` file on this same GPU.
- Automated W&B auth for future instance switches: `scripts/setup_remote.sh`
  now accepts `WANDB_API_KEY` and runs `wandb login` non-interactively
  (same commit). Used it for this instance's setup.

---

## Open questions / next steps (after v12)

v12 did **not** clear the ≥90% bar, and did **not** beat v9 on the trained
policy — it was in fact worse than v9's final checkpoint, despite the
untrained base controller alone being far better than either. This
reframes the problem: v13+ needs to explain and fix the *degradation*
during training, not just tune the residual formulation harder.

**Priority order for v13** (cheapest/most-diagnostic first):

1. **Reduce `action_rate_l2` weight, or disable it, for one ablation run.**
   `action_std` dropping from 0.50 → 0.25–0.28 over the run while
   `position_error` gets *worse* is the strongest single clue — the policy
   is being pushed toward smaller actions by something, and
   `action_rate_l2` (weight -0.01, penalizing action *changes*, not
   magnitude — but a shrinking-residual local optimum could still be
   partly reward-driven) is the most direct reward-shaping suspect. Cheap,
   isolates hypothesis #1 from the diagnosis above.
2. **Log `dq_base` and `dq_residual` magnitudes separately** (add them as
   extra observation-manager metrics, not just to reward) to see directly
   whether the residual is shrinking toward zero over training, and whether
   `dq_base` itself is oscillating (hypothesis #2 — base controller and
   residual fighting each other). This is the single most direct way to
   confirm or rule out #1 vs #2 above, and should be done *before* another
   full 1500-iteration run, since it's cheap to check on the existing W&B
   run's saved checkpoints or a short replay.
3. **Try `orientation_weight > 0` on the base controller** (let IK also
   steer orientation, not just position) — directly tests hypothesis #3.
   `orientation_error` sitting at ~1.2 rad throughout (worse than v9's
   0.7 rad) despite the reward's `orientation_weight=0.5` term suggests the
   residual alone can't reliably control orientation while also handling
   position corrections; giving the base controller some orientation
   authority may resolve the competition.
4. **Try a much smaller `max_dq`** (e.g. 0.1–0.2 rad/step, down from 0.5) —
   tests hypothesis #4. A gentler base controller leaves the residual more
   room to matter without fighting large corrective steps.
5. **If none of the above explains it**: verify the Jacobian/frame
   resolution isn't subtly wrong by comparing `ResidualIKAction`'s `dq_base`
   against `DifferentialIKAction`'s `compute_dq` on the same pose error, to
   rule out an implementation bug rather than a formulation problem.

**Not recommended as a next step**: re-running v12's exact config at
different `num_envs`/iteration counts. v11 already established (in the
non-residual formulation) that scaling data doesn't fix a formulation-level
plateau; there's no reason to expect a different outcome here without
addressing one of the hypotheses above first.