"""Reach-task observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import compute_pose_error, subtract_frame_transforms

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def ee_pose_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Position + orientation error from the end effector to the commanded pose.

    Both the command and the current end-effector pose live in the robot base
    frame, so no world-frame conversion is needed. Returns (B, 6): position
    error (3) followed by an axis-angle orientation error (3).
    """
    robot: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_link_pos_w, robot.data.root_link_quat_w, ee_pos_w, ee_quat_w
    )

    pos_error, rot_error = compute_pose_error(
        ee_pos_b,
        ee_quat_b,
        command[:, :3],
        command[:, 3:7],
        rot_error_type="axis_angle",
    )
    return torch.cat([pos_error, rot_error], dim=-1)
