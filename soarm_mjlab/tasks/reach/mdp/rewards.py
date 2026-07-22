"""Reach-task rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude, subtract_frame_transforms

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def distance_to_target(
    env: ManagerBasedRlEnv,
    command_name: str,
    orientation_weight: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Negative pose error to the commanded end-effector target.

    Position error (m) dominates; ``orientation_weight`` is 0 by default —
    the smallest reward set that produces a non-degenerate policy, per the
    roadmap. Raise it once orientation targets need to be scored, since a
    0-weighted orientation command gives the policy no incentive to satisfy it.
    """
    robot: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_link_pos_w, robot.data.root_link_quat_w, ee_pos_w, ee_quat_w
    )

    position_error = torch.norm(command[:, :3] - ee_pos_b, dim=-1)
    penalty = position_error
    if orientation_weight > 0.0:
        orientation_error = quat_error_magnitude(command[:, 3:7], ee_quat_b)
        penalty = penalty + orientation_weight * orientation_error
    return -penalty

def distance_to_target_shaped(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 0.25,
    orientation_weight: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Exponentially shaped distance reward: exp(-error / sigma).

    Gives a denser gradient near the target than raw -distance.
    Returns values in [0, 1]: 1.0 at target, decaying exponentially.
    """
    robot: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_link_pos_w, robot.data.root_link_quat_w, ee_pos_w, ee_quat_w
    )

    position_error = torch.norm(command[:, :3] - ee_pos_b, dim=-1)
    shaped = torch.exp(-position_error / sigma)
    if orientation_weight > 0.0:
        orientation_error = quat_error_magnitude(command[:, 3:7], ee_quat_b)
        shaped = shaped + orientation_weight * torch.exp(-orientation_error / sigma)
    return shaped
