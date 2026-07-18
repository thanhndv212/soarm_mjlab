"""Reach-task terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from soarm_mjlab.tasks.reach.mdp.commands import UniformPoseCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def task_success(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """True once the end effector has held the target within threshold long enough."""
    command = cast(UniformPoseCommand, env.command_manager.get_term(command_name))
    return command.compute_success()


def joint_limit_violated(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """True if any selected joint is outside its hard position limits."""
    robot: Entity = env.scene[asset_cfg.name]
    pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.joint_pos_limits[:, asset_cfg.joint_ids]
    violated = (pos < limits[..., 0]) | (pos > limits[..., 1])
    return violated.any(dim=-1)
