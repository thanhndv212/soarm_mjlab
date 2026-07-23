"""Layer 2: SO-ARM100 asset + Reach config validation.

MJCF compiles, actuators/joints/site line up, every "Set per-robot"
placeholder from reach_env_cfg.py is actually filled by the time
config/so_arm100/env_cfgs.py returns. No env construction — just spec/cfg
inspection. See SOARM_MJLAB_ROADMAP.md Phase 2, test pyramid layer 2.
"""

from __future__ import annotations

import re

import pytest

from mjlab.entity.entity import Entity
from mjlab.sensor import ContactSensorCfg
from soarm_mjlab.assets.robots.so_arm100.so_arm100_constants import (
    EE_SITE_NAME,
    JOINT_NAMES,
    HOME_KEYFRAME,
    get_so_arm100_robot_cfg,
    get_spec,
)
from soarm_mjlab.tasks.reach.config.so_arm100.env_cfgs import (
    _compute_reachable_workspace,
    so_arm100_reach_env_cfg,
)
from soarm_mjlab.tasks.reach.mdp import UniformPoseCommandCfg
from soarm_mjlab.tasks.reach.mdp.actions import ResidualIKActionCfg


def test_mjcf_compiles_with_actuators_stripped():
    spec = get_spec()
    model = spec.compile()
    assert model.nu == 0
    assert model.nq == len(JOINT_NAMES)


def test_joint_names_match_compiled_model():
    spec = get_spec()
    compiled_names = {j.name for j in spec.joints}
    assert compiled_names == set(JOINT_NAMES)


def test_ee_site_exists():
    spec = get_spec()
    assert EE_SITE_NAME in {s.name for s in spec.sites}


def test_collision_geoms_are_named_and_addressable():
    spec = get_spec()
    collision_geoms = [
        g for g in spec.geoms if g.classname and g.classname.name == "collision"
    ]
    assert collision_geoms, "expected at least one collision-class geom"
    pattern = re.compile(r".*_collision.*")
    for geom in collision_geoms:
        assert geom.name, (
            "collision geom left unnamed — CollisionCfg/sensor regexes can't match it"
        )
        assert pattern.match(geom.name)


def test_home_keyframe_joint_names_are_valid():
    named_joints = {k for k in HOME_KEYFRAME.joint_pos if k != ".*"}
    assert named_joints == set(JOINT_NAMES)


def test_entity_actuators_match_joints_one_to_one():
    entity = Entity(get_so_arm100_robot_cfg())
    model = entity.compile()

    assert model.nu == len(JOINT_NAMES)
    assert set(entity.actuator_names) == set(JOINT_NAMES)


def test_gripper_only_collision_applies_to_expected_geoms():
    entity = Entity(get_so_arm100_robot_cfg())
    model = entity.compile()

    ee_pattern = re.compile(r"(gripper|moving_jaw_so101_v1)_collision.*")
    for i in range(model.ngeom):
        name = model.geom(i).name
        if not name:
            continue
        expected_contype = 1 if ee_pattern.match(name) else 0
        assert model.geom_contype[i] == expected_contype, (
            f"geom '{name}' has contype={model.geom_contype[i]}, expected {expected_contype}"
        )


@pytest.mark.parametrize("play", [False, True])
def test_reach_env_cfg_placeholders_are_filled(play):
    cfg = so_arm100_reach_env_cfg(play=play)

    ee_asset_cfg = cfg.rewards["distance_to_target"].params["asset_cfg"]
    assert ee_asset_cfg.site_names == (EE_SITE_NAME,)

    for group in ("actor", "critic"):
        term_cfg = cfg.observations[group].terms["ee_pose_error"].params["asset_cfg"]
        assert term_cfg.site_names == (EE_SITE_NAME,)

    command_cfg = cfg.commands["ee_pose"]
    assert isinstance(command_cfg, UniformPoseCommandCfg)
    assert command_cfg.asset_cfg.site_names == (EE_SITE_NAME,)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, ResidualIKActionCfg)
    assert joint_pos_action.frame_name == EE_SITE_NAME
    # v13: goal-perturbation task-space residual (not per-joint).
    assert joint_pos_action.residual_mode == "goal"
    assert joint_pos_action.residual_pos_scale > 0
    assert joint_pos_action.residual_rot_scale > 0

    assert cfg.viewer.body_name != ""

    assert cfg.scene.sensors is not None
    (sensor,) = [s for s in cfg.scene.sensors if s.name == "ee_ground_collision"]
    assert isinstance(sensor, ContactSensorCfg)
    assert sensor.primary.pattern != ""

    if play:
        assert cfg.episode_length_s > 1e6
        assert cfg.observations["actor"].enable_corruption is False
        assert command_cfg.resampling_time_range == (3.0, 3.0)


def test_reachable_workspace_box_is_sane():
    x_range, y_range, z_range = _compute_reachable_workspace()

    for lo, hi in (x_range, y_range, z_range):
        assert lo < hi

    # Must clear the ground plane (see so_arm100_constants.HOME_KEYFRAME's
    # comment on why the base is mounted 3cm up).
    assert z_range[0] > 0.0


def test_reachable_workspace_is_deterministic():
    """Same seed => same box on a fresh (uncached) computation."""
    _compute_reachable_workspace.cache_clear()
    first = _compute_reachable_workspace()
    _compute_reachable_workspace.cache_clear()
    second = _compute_reachable_workspace()
    assert first == second
