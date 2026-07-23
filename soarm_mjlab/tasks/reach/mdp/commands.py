"""UniformPoseCommand: a random end-effector target pose in the robot base frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_euler_xyz,
    sample_uniform,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class UniformPoseCommand(CommandTerm):
    """Samples a target end-effector pose uniformly within a reachable box.

    The target is expressed (and resampled) in the robot base frame, so it is
    independent of ``env_origins`` — converted to world frame only for debug
    visualization.
    """

    cfg: UniformPoseCommandCfg

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        # Unlike ObservationTermCfg/RewardTermCfg/TerminationTermCfg params,
        # CommandTermCfg fields aren't auto-resolved by a manager — do it here
        # so site_names -> site_ids actually narrows to the configured site.
        cfg.asset_cfg.resolve(env.scene)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0  # Identity quaternion by default.
        self.success_streak = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.episode_success = torch.zeros(self.num_envs, device=self.device)

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Current end-effector pose in the robot base frame."""
        asset_cfg = self.cfg.asset_cfg
        ee_pos_w = self.robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
        ee_quat_w = self.robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
        root_pos_w = self.robot.data.root_link_pos_w
        root_quat_w = self.robot.data.root_link_quat_w
        return subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    def _update_metrics(self) -> None:
        from mjlab.utils.lab_api.math import quat_error_magnitude

        ee_pos_b, ee_quat_b = self._ee_pose_b()
        position_error = torch.norm(self.pose_command_b[:, :3] - ee_pos_b, dim=-1)
        orientation_error = quat_error_magnitude(self.pose_command_b[:, 3:7], ee_quat_b)

        within_threshold = position_error < self.cfg.success_threshold
        if self.cfg.orientation_threshold > 0.0:
            within_threshold = within_threshold & (
                orientation_error < self.cfg.orientation_threshold
            )
        self.success_streak = torch.where(
            within_threshold,
            self.success_streak + 1,
            torch.zeros_like(self.success_streak),
        )
        self.episode_success = torch.maximum(
            self.episode_success,
            (self.success_streak >= self.cfg.success_steps).float(),
        )

        self.metrics["position_error"] = position_error
        self.metrics["orientation_error"] = orientation_error
        self.metrics["episode_success"] = self.episode_success

    def compute_success(self) -> torch.Tensor:
        return self.success_streak >= self.cfg.success_steps

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self.success_streak[env_ids] = 0

        r = self.cfg.position_range
        lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
        upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
        self.pose_command_b[env_ids, :3] = sample_uniform(
            lower, upper, (n, 3), device=self.device
        )

        o = self.cfg.orientation_range
        roll = sample_uniform(o.roll[0], o.roll[1], (n,), device=self.device)
        pitch = sample_uniform(o.pitch[0], o.pitch[1], (n,), device=self.device)
        yaw = sample_uniform(o.yaw[0], o.yaw[1], (n,), device=self.device)
        self.pose_command_b[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        root_pos_w = self.robot.data.root_link_pos_w
        root_quat_w = self.robot.data.root_link_quat_w
        target_pos_w, target_quat_w = combine_frame_transforms(
            root_pos_w,
            root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:7],
        )
        target_rotm = matrix_from_quat(target_quat_w).cpu().numpy()

        asset_cfg = self.cfg.asset_cfg
        ee_pos_w = self.robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
        ee_quat_w = self.robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
        ee_rotm = matrix_from_quat(ee_quat_w).cpu().numpy()

        if self.cfg.viz.show_workspace:
            self._draw_workspace_box(visualizer, env_indices, root_pos_w, root_quat_w)

        for batch in env_indices:
            color = (
                self.cfg.viz.success_color
                if bool(self.episode_success[batch])
                else self.cfg.viz.target_color
            )
            visualizer.add_sphere(
                center=target_pos_w[batch].cpu().numpy(),
                radius=0.02,
                color=color,
                label=f"reach_target_{batch}",
            )
            # Target EE pose (dashed-style default axis colors, larger scale
            # so it's visually distinct from the "current" frame below).
            visualizer.add_frame(
                position=target_pos_w[batch].cpu().numpy(),
                rotation_matrix=target_rotm[batch],
                scale=0.08,
                label=f"reach_target_frame_{batch}",
            )
            # Current EE pose — where the reach is actually happening. This
            # is the frame to watch during play to see how well the policy
            # (+ IK base controller, for ResidualIKAction) tracks position
            # *and* orientation, not just the position-only success sphere.
            visualizer.add_frame(
                position=ee_pos_w[batch].cpu().numpy(),
                rotation_matrix=ee_rotm[batch],
                scale=0.12,
                label=f"reach_ee_frame_{batch}",
            )

    def _draw_workspace_box(
        self,
        visualizer: DebugVisualizer,
        env_indices,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> None:
        """Draw the target-position sampling range (``position_range``) as a
        translucent box, in the robot base frame — i.e. the actual reachable
        space targets are drawn from, not a general kinematic workspace.
        """
        r = self.cfg.position_range
        center_b = torch.tensor(
            [
                (r.x[0] + r.x[1]) / 2,
                (r.y[0] + r.y[1]) / 2,
                (r.z[0] + r.z[1]) / 2,
            ],
            device=self.device,
        )
        half_extent = (
            (r.x[1] - r.x[0]) / 2,
            (r.y[1] - r.y[0]) / 2,
            (r.z[1] - r.z[0]) / 2,
        )
        for batch in env_indices:
            center_w, _ = combine_frame_transforms(
                root_pos_w[batch : batch + 1],
                root_quat_w[batch : batch + 1],
                center_b.unsqueeze(0),
            )
            root_rotm = matrix_from_quat(root_quat_w[batch]).cpu().numpy()
            visualizer.add_box(
                center=center_w[0].cpu().numpy(),
                size=half_extent,
                mat=root_rotm,
                color=self.cfg.viz.workspace_color,
                label=f"reach_workspace_{batch}",
            )


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    entity_name: str
    asset_cfg: SceneEntityCfg = field(default_factory=lambda: _DEFAULT_ASSET_CFG)
    success_threshold: float = 0.03
    """Position error (m) below which the target counts as reached."""
    success_steps: int = 10
    """Consecutive steps the error must stay below threshold to count as success."""
    orientation_threshold: float = 0.0
    """Orientation error (rad) below which the target counts as reached.
    0 = position-only success (v1–v13 behavior). >0 = orientation-gated
    success — the base IK controller (orientation_weight=0) provably fails,
    giving the residual a real task (fix orientation) and preventing reward
    saturation."""

    @dataclass
    class PositionRangeCfg:
        """Target position sampling range, in the robot base frame (m)."""

        x: tuple[float, float] = (0.1, 0.3)
        y: tuple[float, float] = (-0.2, 0.2)
        z: tuple[float, float] = (0.1, 0.4)

    position_range: PositionRangeCfg = field(default_factory=PositionRangeCfg)

    @dataclass
    class OrientationRangeCfg:
        """Target orientation sampling range, in the robot base frame (rad)."""

        roll: tuple[float, float] = (0.0, 0.0)
        pitch: tuple[float, float] = (0.0, 0.0)
        yaw: tuple[float, float] = (0.0, 0.0)

    orientation_range: OrientationRangeCfg = field(default_factory=OrientationRangeCfg)

    @dataclass
    class VizCfg:
        target_color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.5)
        success_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5)
        show_workspace: bool = False
        """Draw the target-position sampling range (``position_range``) as
        a translucent box. Off by default (adds visual clutter during
        training-time debug_vis); enable for play to see the reachable
        space the targets are drawn from."""
        workspace_color: tuple[float, float, float, float] = (0.2, 0.4, 1.0, 0.25)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> UniformPoseCommand:
        return UniformPoseCommand(self, env)
