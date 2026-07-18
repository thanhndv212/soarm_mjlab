"""Layer 3: SoArm100-Reach environment smoke test.

Instantiate the real ManagerBasedRlEnv (num_envs=2), reset() + a few
step()s with random actions, assert no NaN/Inf anywhere and termination
shapes match num_envs. See SOARM_MJLAB_ROADMAP.md Phase 2, test pyramid
layer 3.
"""

from __future__ import annotations

import pytest
import torch

import soarm_mjlab.tasks  # noqa: F401 — populates the task registry.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from tests.conftest import get_test_device

NUM_ENVS = 2
NUM_STEPS = 10


@pytest.fixture(scope="module")
def device():
    return get_test_device()


@pytest.fixture(scope="module")
def env(device):
    cfg = load_env_cfg("SoArm100-Reach")
    cfg.scene.num_envs = NUM_ENVS
    environment = ManagerBasedRlEnv(cfg=cfg, device=device)
    yield environment
    environment.close()


def _assert_finite(name: str, value: torch.Tensor) -> None:
    assert torch.isfinite(value).all(), f"non-finite value in {name}: {value}"


def test_reset_produces_finite_observations(env):
    obs, _ = env.reset()
    assert set(obs.keys()) == {"actor", "critic"}
    for group, tensor in obs.items():
        assert tensor.shape[0] == NUM_ENVS
        _assert_finite(f"obs[{group}] after reset", tensor)


def test_step_loop_stays_finite_with_random_actions(env):
    env.reset()
    action_dim = env.action_manager.total_action_dim

    for _ in range(NUM_STEPS):
        actions = 2 * torch.rand(NUM_ENVS, action_dim, device=env.device) - 1
        obs, reward, terminated, truncated, _ = env.step(actions)

        for group, tensor in obs.items():
            _assert_finite(f"obs[{group}]", tensor)
        _assert_finite("reward", reward)

        assert reward.shape == (NUM_ENVS,)
        assert terminated.shape == (NUM_ENVS,)
        assert truncated.shape == (NUM_ENVS,)
        assert terminated.dtype == torch.bool
        assert truncated.dtype == torch.bool
