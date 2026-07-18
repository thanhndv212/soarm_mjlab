"""Layer 1 unit tests: Reach mdp/*.py functions called with synthetic tensors.

No MuJoCo, no ManagerBasedRlEnv — just the reward/observation/termination
math, exercised against small hand-built fakes for ``env``/``robot``/
``command_manager``. See SOARM_MJLAB_ROADMAP.md Phase 2, test pyramid
layer 1.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from soarm_mjlab.tasks.reach.mdp.observations import ee_pose_error
from soarm_mjlab.tasks.reach.mdp.rewards import distance_to_target
from soarm_mjlab.tasks.reach.mdp.terminations import joint_limit_violated, task_success


class FakeCommandManager:
    """Stands in for mjlab's CommandManager: raw tensors + term objects by name."""

    def __init__(self, commands: dict | None = None, terms: dict | None = None):
        self._commands = commands or {}
        self._terms = terms or {}

    def get_command(self, name: str) -> torch.Tensor:
        return self._commands[name]

    def get_term(self, name: str):
        return self._terms[name]


def _identity_quat(n: int) -> torch.Tensor:
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def _make_env(
    *,
    ee_pos_w: torch.Tensor,
    ee_quat_w: torch.Tensor,
    root_pos_w: torch.Tensor | None = None,
    root_quat_w: torch.Tensor | None = None,
    command: torch.Tensor | None = None,
) -> SimpleNamespace:
    n = ee_pos_w.shape[0]
    root_pos_w = root_pos_w if root_pos_w is not None else torch.zeros(n, 3)
    root_quat_w = root_quat_w if root_quat_w is not None else _identity_quat(n)
    robot_data = SimpleNamespace(
        site_pos_w=ee_pos_w.unsqueeze(
            1
        ),  # (B, 1, 3): one site selected by site_ids=[0].
        site_quat_w=ee_quat_w.unsqueeze(1),
        root_link_pos_w=root_pos_w,
        root_link_quat_w=root_quat_w,
    )
    robot = SimpleNamespace(data=robot_data)
    commands = {"ee_pose": command} if command is not None else {}
    return SimpleNamespace(
        scene={"robot": robot}, command_manager=FakeCommandManager(commands=commands)
    )


_ASSET_CFG = SceneEntityCfg("robot", site_ids=[0])


def test_distance_to_target_zero_error_gives_zero_reward():
    env = _make_env(
        ee_pos_w=torch.zeros(1, 3),
        ee_quat_w=_identity_quat(1),
        command=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
    )
    reward = distance_to_target(env, "ee_pose", asset_cfg=_ASSET_CFG)
    assert torch.allclose(reward, torch.zeros(1))


def test_distance_to_target_matches_known_l2_distance():
    ee_pos_w = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    command = torch.zeros(2, 7)
    command[:, 3] = 1.0  # Identity quat.
    command[1, :3] = torch.tensor([3.0, 4.0, 0.0])
    env = _make_env(ee_pos_w=ee_pos_w, ee_quat_w=_identity_quat(2), command=command)

    reward = distance_to_target(env, "ee_pose", asset_cfg=_ASSET_CFG)

    assert torch.allclose(reward, torch.tensor([-1.0, -5.0]), atol=1e-5)


def test_distance_to_target_ignores_orientation_by_default():
    """orientation_weight=0.0 is the documented default — a rotated target
    must not change the reward versus an aligned one."""
    ee_pos_w = torch.zeros(1, 3)
    aligned_command = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    rotated_quat = quat_from_euler_xyz(
        torch.zeros(1), torch.zeros(1), torch.tensor([math.pi / 2])
    )
    rotated_command = torch.cat([torch.zeros(1, 3), rotated_quat], dim=-1)

    env_aligned = _make_env(
        ee_pos_w=ee_pos_w, ee_quat_w=_identity_quat(1), command=aligned_command
    )
    env_rotated = _make_env(
        ee_pos_w=ee_pos_w, ee_quat_w=_identity_quat(1), command=rotated_command
    )

    reward_aligned = distance_to_target(env_aligned, "ee_pose", asset_cfg=_ASSET_CFG)
    reward_rotated = distance_to_target(env_rotated, "ee_pose", asset_cfg=_ASSET_CFG)

    assert torch.allclose(reward_aligned, reward_rotated)


def test_distance_to_target_scores_orientation_when_weighted():
    ee_pos_w = torch.zeros(1, 3)
    rotated_quat = quat_from_euler_xyz(
        torch.zeros(1), torch.zeros(1), torch.tensor([math.pi / 2])
    )
    command = torch.cat([torch.zeros(1, 3), rotated_quat], dim=-1)
    env = _make_env(ee_pos_w=ee_pos_w, ee_quat_w=_identity_quat(1), command=command)

    reward = distance_to_target(
        env, "ee_pose", orientation_weight=1.0, asset_cfg=_ASSET_CFG
    )

    # Zero position error, pi/2 orientation error, weight 1.0.
    assert torch.allclose(reward, torch.tensor([-math.pi / 2]), atol=1e-5)


def test_ee_pose_error_identity_frame_is_zero():
    env = _make_env(
        ee_pos_w=torch.zeros(1, 3),
        ee_quat_w=_identity_quat(1),
        command=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
    )
    error = ee_pose_error(env, "ee_pose", asset_cfg=_ASSET_CFG)
    assert error.shape == (1, 6)
    assert torch.allclose(error, torch.zeros(1, 6), atol=1e-6)


def test_ee_pose_error_position_component_matches_offset():
    ee_pos_w = torch.zeros(1, 3)
    command = torch.zeros(1, 7)
    command[:, 3] = 1.0
    command[0, :3] = torch.tensor([0.1, -0.2, 0.3])
    env = _make_env(ee_pos_w=ee_pos_w, ee_quat_w=_identity_quat(1), command=command)

    error = ee_pose_error(env, "ee_pose", asset_cfg=_ASSET_CFG)

    assert torch.allclose(error[:, :3], torch.tensor([[0.1, -0.2, 0.3]]), atol=1e-5)
    assert torch.allclose(error[:, 3:], torch.zeros(1, 3), atol=1e-6)


def test_ee_pose_error_orientation_component_matches_rotation_angle():
    ee_pos_w = torch.zeros(1, 3)
    rotated_quat = quat_from_euler_xyz(
        torch.zeros(1), torch.zeros(1), torch.tensor([math.pi / 2])
    )
    command = torch.cat([torch.zeros(1, 3), rotated_quat], dim=-1)
    env = _make_env(ee_pos_w=ee_pos_w, ee_quat_w=_identity_quat(1), command=command)

    error = ee_pose_error(env, "ee_pose", asset_cfg=_ASSET_CFG)

    assert torch.allclose(
        torch.norm(error[:, 3:], dim=-1), torch.tensor([math.pi / 2]), atol=1e-5
    )


def test_task_success_passes_through_command_term():
    class FakeCommandTerm:
        def compute_success(self) -> torch.Tensor:
            return torch.tensor([True, False])

    env = SimpleNamespace(
        command_manager=FakeCommandManager(terms={"ee_pose": FakeCommandTerm()})
    )
    result = task_success(env, "ee_pose")
    assert torch.equal(result, torch.tensor([True, False]))


def test_joint_limit_violated_detects_out_of_range():
    joint_pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Within limits.
            [2.0, 0.0, 0.0],  # First joint exceeds upper limit.
            [0.0, -2.0, 0.0],  # Second joint exceeds lower limit.
        ]
    )
    limits = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]).expand(3, 3, 2)
    robot = SimpleNamespace(
        data=SimpleNamespace(joint_pos=joint_pos, joint_pos_limits=limits)
    )
    env = SimpleNamespace(scene={"robot": robot})

    asset_cfg = SceneEntityCfg("robot", joint_ids=[0, 1, 2])
    result = joint_limit_violated(env, asset_cfg=asset_cfg)

    assert torch.equal(result, torch.tensor([False, True, True]))
