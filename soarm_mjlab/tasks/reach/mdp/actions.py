"""Residual IK action: DLS IK base controller + learned residual.

Reformulates the Reach action space from "policy commands joint positions
directly" to "policy commands a small correction on top of an analytic IK
base controller." The base controller computes a damped-least-squares (DLS)
IK step toward the commanded target pose every step (GPU-vectorized via
``mujoco_warp.jac``).

Two residual modes are supported:

**``residual_mode="joint"`` (v12, additive joint-space):**

    q_target = q_current + dq_base_ik + dq_residual

The policy outputs a per-joint residual added on top of the IK step. Near
the target, ``dq_base ≈ 0`` while ``dq_residual`` is not small relative to
it → the residual fights the base controller → oscillation/shake. PPO's
only improving direction is "shrink the residual," which degrades
positioning.

**``residual_mode="goal"`` (v13, goal-perturbation task-space):**

    commanded_pose = target_pose + residual_pose_delta
    dq = DLS_IK(commanded_pose, current_pose)
    q_target = q_current + dq

The policy outputs a 6-D pose delta (3 position + 3 rotation) that
**perturbs the goal the IK aims at**. The IK just tracks the perturbed
goal — the residual no longer fights the controller, it steers it. Near
the target: the residual moves the equilibrium point, the IK settles
there smoothly. No shake by construction.

The Jacobian/DLS machinery is identical to mjlab's own
``DifferentialIKAction`` (``mjlab.envs.mdp.actions.differential_ik``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_box_plus,
    quat_from_matrix,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.string import resolve_expr

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ResidualIKActionCfg(ActionTermCfg):
    """Configuration for the residual IK action space.

    The action dimension equals the number of controlled joints — the
    policy outputs a per-joint residual (dimensionless, scaled by
    ``residual_scale`` to radians).
    """

    actuator_names: tuple[str, ...] | list[str]
    """Actuator name expressions to resolve the controlled joints."""

    frame_type: Literal["body", "site", "geom"] = "site"
    """Element type of the end-effector frame."""

    frame_name: str
    """Name of the EE frame element on the entity. Set per-robot."""

    command_name: str = "ee_pose"
    """Name of the command term that provides the target pose (base frame,
    7-D: [pos(3), quat(4)]). The IK base controller steers toward this
    target every step."""

    # -- Base IK controller params (same semantics as DifferentialIKAction) --

    damping: float = 0.05
    """Damping coefficient (lambda) for the DLS pseudoinverse."""

    max_dq: float = 0.5
    """Maximum joint displacement per IK solve (rad/step)."""

    position_weight: float = 1.0
    """Weight for the position residual in the DLS system."""

    orientation_weight: float = 0.0
    """Weight for the orientation residual (0 = position-only base)."""

    joint_limit_weight: float = 0.0
    """Weight for soft joint-limit residuals (0 = disabled)."""

    posture_weight: float = 0.0
    """Weight for posture regularization toward ``posture_target`` (0 = disabled)."""

    posture_target: dict[str, float] | None = None
    """Target joint positions for posture regularization. Only used when
    ``posture_weight > 0``."""

    # -- Residual params --

    residual_mode: Literal["joint", "goal"] = "joint"
    """How the policy residual composes with the IK base controller.

    ``"joint"`` (v12): additive joint-space — the policy outputs a
    per-joint residual added to the IK step
    (``q = q_cur + dq_base + dq_residual``). Near the target the
    residual fights the base controller → oscillation.

    ``"goal"`` (v13): goal-perturbation task-space — the policy outputs
    a 6-D pose delta (3 position + 3 rotation) that perturbs the IK
    goal (``commanded = target + delta; q = q_cur + DLS_IK(commanded)``).
    The residual steers the IK instead of fighting it → no shake.
    """

    residual_scale: float | dict[str, float] = 0.1
    """Per-joint scale (rad) for the policy's residual output in ``"joint"``
    mode. Float or dict mapping joint names to scales. Much smaller than the
    old ``JointPositionAction`` scale (1.0 rad) because the base controller
    handles coarse motion — the residual is only for fine corrections.
    Ignored in ``"goal"`` mode."""

    residual_pos_scale: float = 0.02
    """Position delta scale (meters) for the policy's residual output in
    ``"goal"`` mode. The policy commands ``action[:, :3] * residual_pos_scale``
    as a world-frame position perturbation on the IK target. Ignored in
    ``"joint"`` mode."""

    residual_rot_scale: float = 0.05
    """Orientation delta scale (rad) for the policy's residual output in
    ``"goal"`` mode. The policy commands ``action[:, 3:6] * residual_rot_scale``
    as a rotation-vector perturbation on the IK target's orientation (applied
    via SO(3) box-plus). Ignored in ``"joint"`` mode."""

    frozen_joints: dict[str, float] | None = None
    """Optional joint-name -> fixed target (rad) overrides, applied after
    the base IK + residual computation every step. Not used during
    training (the Reach task only scores the EE pose, not the gripper, so
    training leaves this ``None`` and the gripper follows whatever the base
    controller + residual happen to produce for it). Set for play/deploy —
    e.g. ``{"gripper": <closed_rad>}`` — so watching a rollout isn't
    confused by an untrained, semantically-meaningless gripper motion."""

    def build(self, env: ManagerBasedRlEnv) -> ResidualIKAction:
        return ResidualIKAction(self, env)


class ResidualIKAction(ActionTerm):
    """DLS IK base controller + learned residual (joint-space or goal-perturbation).

    Every step:
    1. Reads the target pose from the command manager (base frame) and
       converts it to world frame via the robot's root pose.
    2. Computes the pose error between the current EE frame and the target
       (or the perturbed target in ``"goal"`` mode).
    3. Solves a damped-least-squares IK step (``dq_base``) using the
       GPU-vectorized Jacobian from ``mujoco_warp.jac``.
    4. In ``"joint"`` mode: adds the policy's per-joint residual
       (``dq_residual = action * residual_scale``) and sets
       ``q_target = q_current + dq_base + dq_residual``.
       In ``"goal"`` mode: the policy's 6-D pose delta perturbs the IK goal
       (``commanded = target + delta``), the IK step tracks the perturbed
       goal, and ``q_target = q_current + dq_base`` (no joint residual).
    5. Clamps to soft joint limits and applies frozen-joint overrides.
    """

    cfg: ResidualIKActionCfg
    _entity: Entity

    def __init__(self, cfg: ResidualIKActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)

        # -- Joint setup (same as DifferentialIKAction) --
        joint_ids, joint_names = self._entity.find_joints_by_actuator_names(
            cfg.actuator_names
        )
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._num_joints = len(joint_ids)
        self._joint_names = joint_names
        self._joint_dof_ids = self._entity.indexing.joint_v_adr[self._joint_ids]

        # -- Frame resolution (same as DifferentialIKAction) --
        self._frame_type = cfg.frame_type
        self._resolve_frame(cfg.frame_name)

        # -- Action dim: num_joints (joint mode) or 6 (goal mode: 3 pos + 3 rot)
        if cfg.residual_mode == "goal":
            self._action_dim = 6
        else:
            self._action_dim = self._num_joints
        self._raw_actions = torch.zeros(
            self.num_envs, self._action_dim, device=self.device
        )

        # -- Joint limits --
        limits = self._entity.data.soft_joint_pos_limits
        self._joint_lower = limits[:, self._joint_ids, 0]
        self._joint_upper = limits[:, self._joint_ids, 1]

        # -- Posture regularization target (same as DifferentialIKAction) --
        q_target = self._entity.data.default_joint_pos[:, self._joint_ids].clone()
        if cfg.posture_target is not None:
            overrides = resolve_expr(cfg.posture_target, tuple(joint_names))
            for j, val in enumerate(overrides):
                if val is not None:
                    q_target[:, j] = val
        self._posture_target = q_target

        # -- Residual scale --
        if cfg.residual_mode == "goal":
            # Goal mode: pose-space scales (pos in meters, rot in rad).
            # _residual_scale stores [pos_scale, rot_scale] for ONNX metadata.
            self._residual_scale = [
                float(cfg.residual_pos_scale),
                float(cfg.residual_rot_scale),
            ]
            self._residual_pos_scale = float(cfg.residual_pos_scale)
            self._residual_rot_scale = float(cfg.residual_rot_scale)
        elif isinstance(cfg.residual_scale, (float, int)):
            self._residual_scale = float(cfg.residual_scale)
        elif isinstance(cfg.residual_scale, dict):
            scale = torch.ones(
                self.num_envs, self._num_joints, device=self.device
            )
            index_list, _, value_list = resolve_matching_names_values(
                cfg.residual_scale, self._joint_names
            )
            scale[:, index_list] = torch.tensor(value_list, device=self.device)
            self._residual_scale = scale
        else:
            raise ValueError(
                f"Unsupported residual_scale type: {type(cfg.residual_scale)}. "
                "Supported: float or dict (joint mode)."
            )

        # -- Frozen-joint overrides (play/deploy only, see cfg docstring) --
        self._frozen_mask: torch.Tensor | None = None
        self._frozen_target: torch.Tensor | None = None
        if cfg.frozen_joints is not None:
            name_to_local = {name: i for i, name in enumerate(joint_names)}
            mask = torch.zeros(self._num_joints, dtype=torch.bool, device=self.device)
            target = torch.zeros(self._num_joints, device=self.device)
            for name, value in cfg.frozen_joints.items():
                if name not in name_to_local:
                    raise ValueError(
                        f"frozen_joints key {name!r} is not one of the "
                        f"actuated joints controlled by this action term: "
                        f"{joint_names}"
                    )
                idx = name_to_local[name]
                mask[idx] = True
                target[idx] = value
            self._frozen_mask = mask
            self._frozen_target = target

        # -- Jacobian buffers (same as DifferentialIKAction) --
        nworld = self.num_envs
        nv = self._env.sim.mj_model.nv
        with wp.ScopedDevice(self._env.sim.wp_device):
            self._jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
            self._jacr_wp = wp.zeros((nworld, 3, nv), dtype=float)
            self._point_wp = wp.zeros(nworld, dtype=wp.vec3)
            self._body_wp = wp.zeros(nworld, dtype=wp.int32)
            self._body_wp.fill_(self._body_id)

        self._jacp_torch = wp.to_torch(self._jacp_wp)
        self._jacr_torch = wp.to_torch(self._jacr_wp)
        self._point_torch = wp.to_torch(self._point_wp).view(nworld, 3)

    # -- ActionTerm interface --

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        # 1. Get command target (base frame) → world frame.
        command = self._env.command_manager.get_command(self.cfg.command_name)
        # command: (num_envs, 7) = [pos_b(3), quat_b(4)]
        root_pos_w = self._entity.data.root_link_pos_w
        root_quat_w = self._entity.data.root_link_quat_w
        target_pos_w, target_quat_w = combine_frame_transforms(
            root_pos_w, root_quat_w, command[:, :3], command[:, 3:7]
        )

        # 2. Current EE frame pose (world frame).
        frame_pos, frame_quat = self._get_frame_pose()

        # 3. Determine the pose the IK base controller will aim at.
        if self.cfg.residual_mode == "goal":
            # Goal-perturbation: the policy's 6-D output perturbs the IK
            # target. The IK tracks the perturbed goal; the residual steers
            # the IK instead of fighting it (no adversarial closed loop).
            # The reward still measures distance to the *true* target (the
            # command manager's ee_pose), not this perturbed goal — the
            # two-target wiring is entirely inside this action term.
            residual_pos = self._raw_actions[:, :3] * self._residual_pos_scale
            residual_rot = self._raw_actions[:, 3:6] * self._residual_rot_scale
            ik_target_pos = target_pos_w + residual_pos
            ik_target_quat = quat_box_plus(target_quat_w, residual_rot)
        else:
            # Joint-space additive: IK aims at the true target; the
            # residual is added in joint space after the IK step.
            ik_target_pos = target_pos_w
            ik_target_quat = target_quat_w

        # 4. Pose error → IK residual.
        pos_error, rot_error = compute_pose_error(
            frame_pos, frame_quat, ik_target_pos, ik_target_quat
        )

        # 5. Jacobian (GPU-vectorized via mujoco_warp).
        self._point_torch[:] = frame_pos
        self._compute_jacobian()
        jacp = self._jacp_torch[:, :, self._joint_dof_ids]
        jacr = self._jacr_torch[:, :, self._joint_dof_ids]

        # 6. DLS normal equations: (J^T W J + λ²I) dq = J^T W dx.
        #    Same formulation as DifferentialIKAction.compute_dq.
        w_pos = self.cfg.position_weight
        w_ori = self.cfg.orientation_weight
        w_lim = self.cfg.joint_limit_weight
        w_post = self.cfg.posture_weight
        lam = max(self.cfg.damping, 1e-6)

        wp2, wo2 = w_pos * w_pos, w_ori * w_ori
        JTJ = wp2 * torch.einsum("bti,btj->bij", jacp, jacp) + wo2 * torch.einsum(
            "bti,btj->bij", jacr, jacr
        )
        JTdx = wp2 * torch.einsum(
            "bti,bt->bi", jacp, pos_error
        ) + wo2 * torch.einsum("bti,bt->bi", jacr, rot_error)

        # Joint-limit penalty (diagonal contribution).
        q = self._entity.data.joint_pos[:, self._joint_ids]
        r_limit = (self._joint_upper - q).clamp(max=0) + (
            self._joint_lower - q
        ).clamp(min=0)
        violated = (r_limit != 0).float()
        wl2 = w_lim * w_lim
        JTJ.diagonal(dim1=-2, dim2=-1).add_(wl2 * violated)
        JTdx.add_(wl2 * violated * r_limit)

        # Posture regularization (identity Jacobian contribution).
        if w_post > 0:
            r_posture = self._posture_target - q
            wpost2 = w_post * w_post
            JTJ.diagonal(dim1=-2, dim2=-1).add_(wpost2)
            JTdx.add_(wpost2 * r_posture)

        # Damping.
        JTJ.diagonal(dim1=-2, dim2=-1).add_(lam * lam)

        dq_base = torch.linalg.solve(JTJ, JTdx)
        dq_base = dq_base.clamp(-self.cfg.max_dq, self.cfg.max_dq)

        # 7. Final joint target.
        q_current = self._entity.data.joint_pos[:, self._joint_ids]
        if self.cfg.residual_mode == "goal":
            # Goal mode: IK tracks the perturbed goal — no joint residual.
            q_target = q_current + dq_base
        else:
            # Joint mode: add the per-joint policy residual.
            dq_residual = self._raw_actions * self._residual_scale
            q_target = q_current + dq_base + dq_residual
        q_target = q_target.clamp(self._joint_lower, self._joint_upper)

        # 8. Frozen-joint overrides (play/deploy only — see cfg docstring).
        if self._frozen_mask is not None:
            assert self._frozen_target is not None
            q_target = torch.where(self._frozen_mask, self._frozen_target, q_target)

        self._entity.set_joint_position_target(q_target, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0

    # -- Private (same as DifferentialIKAction) --

    def _resolve_frame(self, frame_name: str) -> None:
        if self._frame_type == "body":
            ids, _ = self._entity.find_bodies(frame_name)
            local_id = ids[0]
            self._frame_id = int(self._entity.indexing.body_ids[local_id].item())
            self._body_id = self._frame_id
        elif self._frame_type == "site":
            ids, _ = self._entity.find_sites(frame_name)
            local_id = ids[0]
            self._frame_id = int(self._entity.indexing.site_ids[local_id].item())
            self._body_id = int(self._env.sim.mj_model.site_bodyid[self._frame_id])
        elif self._frame_type == "geom":
            ids, _ = self._entity.find_geoms(frame_name)
            local_id = ids[0]
            self._frame_id = int(self._entity.indexing.geom_ids[local_id].item())
            self._body_id = int(self._env.sim.mj_model.geom_bodyid[self._frame_id])
        else:
            raise ValueError(f"Unknown frame_type: {self._frame_type}")

    def _get_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._env.sim.data
        if self._frame_type == "body":
            pos = data.xpos[:, self._frame_id]
            quat = data.xquat[:, self._frame_id]
        elif self._frame_type == "site":
            pos = data.site_xpos[:, self._frame_id]
            xmat = data.site_xmat[:, self._frame_id]
            quat = quat_from_matrix(xmat)
        else:
            assert self._frame_type == "geom"
            pos = data.geom_xpos[:, self._frame_id]
            xmat = data.geom_xmat[:, self._frame_id]
            quat = quat_from_matrix(xmat)
        return pos, quat

    def _compute_jacobian(self) -> None:
        with wp.ScopedDevice(self._env.sim.wp_device):
            mjwarp.jac(
                self._env.sim.wp_model,
                self._env.sim.wp_data,
                self._jacp_wp,
                self._jacr_wp,
                self._point_wp,
                self._body_wp,
            )