"""PPO runner for the Reach task, with ONNX export on checkpoint save.

Mirrors mjlab's own per-task runner subclasses (e.g.
``mjlab.tasks.manipulation.rl.ManipulationOnPolicyRunner``) so a trained
checkpoint is always paired with an ONNX export — what Phase 5's
``deploy/reach_policy_runner.py`` will load. The raw ``.pt`` checkpoint is
already uploaded to W&B automatically by rsl_rl's own ``WandbLogWriter``
whenever ``--agent.logger wandb`` (the default) is active; this runner adds
the ONNX export to that same upload so both artifacts are retrievable from
a W&B run — e.g. by ``scripts/play.py --wandb-run-path`` on a machine that
never ran the training job itself (see docs/vast_ai_training.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, list_to_csv_str
from mjlab.rl.runner import MjlabOnPolicyRunner
from soarm_mjlab.tasks.reach.mdp.actions import ResidualIKAction

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


def _get_reach_metadata(
    env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float]:
    """Build ONNX export metadata for the Reach task's ``joint_pos`` action term.

    Same fields as ``mjlab.rl.exporter_utils.get_base_metadata``, but that
    function hard-asserts the action term is ``JointPositionAction`` to read
    its ``_scale`` for the ``action_scale`` metadata field — Reach uses
    ``ResidualIKAction`` (Phase 4's residual-RL reformulation) instead, so
    this reimplements the same metadata shape with an ``action_scale`` drawn
    from ``ResidualIKAction._residual_scale``.

    Note for a Phase 5 deploy consumer: ``action_scale`` here documents the
    raw-policy-output -> joint-radians scale for the *residual* only. The
    actual joint target also requires the DLS IK base-controller step
    (``dq_base``, driven by the commanded target pose), which is not encoded
    in this metadata — a deploy script must run the same base-controller
    logic as ``ResidualIKAction.apply_actions``, not just apply
    ``action_scale`` to the raw policy output.
    """
    robot = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, ResidualIKAction)

    joint_name_to_ctrl_id = {}
    for actuator in robot.spec.actuators:
        joint_name = actuator.target.split("/")[-1]
        joint_name_to_ctrl_id[joint_name] = actuator.id
    ctrl_ids_natural = [
        joint_name_to_ctrl_id[jname]
        for jname in robot.joint_names
        if jname in joint_name_to_ctrl_id
    ]
    joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
    joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]

    observation_term_scale: list = []
    observation_term_flatten_history_dim: list = []
    observation_term_history_length: list = []
    observation_term_clip: list = []
    observation_names = env.observation_manager.active_terms["actor"]

    for active_term in observation_names:
        cfg = env.observation_manager.get_term_cfg("actor", active_term)

        if cfg.scale is None:
            observation_term_scale.append(1.0)
        else:
            raw_scale = cfg.scale
            scale = (
                raw_scale.cpu().tolist()
                if isinstance(raw_scale, torch.Tensor)
                else raw_scale
            )
            observation_term_scale.append(scale)

        raw_clip = cfg.clip
        if raw_clip is None:
            observation_term_clip.append([float("-inf"), float("inf")])
        else:
            observation_term_clip.append(list(raw_clip))

        observation_term_flatten_history_dim.append(cfg.flatten_history_dim)
        observation_term_history_length.append(cfg.history_length)

    residual_scale = joint_action._residual_scale
    if isinstance(residual_scale, torch.Tensor):
        action_scale = residual_scale[0].cpu().tolist()
    elif isinstance(residual_scale, list):
        # Goal mode: [pos_scale, rot_scale]
        action_scale = residual_scale
    else:
        action_scale = residual_scale

    action_space = (
        "residual_ik_goal"
        if joint_action.cfg.residual_mode == "goal"
        else "residual_ik"
    )

    return {
        "run_path": run_path,
        "joint_names": list(robot.joint_names),
        "joint_stiffness": joint_stiffness.tolist(),
        "joint_damping": joint_damping.tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "command_names": list(env.command_manager.active_terms),
        "observation_names": observation_names,
        "observation_terms_scale": observation_term_scale,
        "observation_terms_flatten_history_dim": observation_term_flatten_history_dim,
        "observation_terms_history_length": observation_term_history_length,
        "observation_terms_clip": observation_term_clip,
        "action_scale": action_scale,
        "action_space": action_space,
    }


class ReachOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        super().save(path, infos)
        policy_dir, filename, onnx_path = self._get_export_paths(path)
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
            has_active_run = wandb is not None and wandb.run is not None
            run_path = wandb.run.name if has_active_run else "local"
            metadata = _get_reach_metadata(self.env.unwrapped, run_path=run_path)
            attach_metadata_to_onnx(str(onnx_path), metadata)
            if has_active_run:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")


# Re-exported for callers that only need CSV formatting without the full
# metadata dict (mirrors mjlab.rl.exporter_utils's own module surface).
__all__ = ["ReachOnPolicyRunner", "list_to_csv_str"]