"""SO-ARM100 per-robot Reach env config: fills the base cfg's placeholders."""

from __future__ import annotations

from functools import lru_cache

import mujoco
import numpy as np

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg
from soarm_mjlab.assets.robots import SO_ARM100_ACTION_SCALE, get_so_arm100_robot_cfg
from soarm_mjlab.assets.robots.so_arm100.so_arm100_constants import (
    EE_SITE_NAME,
    get_spec,
)
from soarm_mjlab.tasks.reach.mdp import UniformPoseCommandCfg
from soarm_mjlab.tasks.reach.reach_env_cfg import make_reach_env_cfg

# End-effector body whose contact with the ground plane counts as illegal
# (see reach_env_cfg's ee_ground_collision sensor/termination).
_EE_BODY_PATTERN = "(gripper|moving_jaw_so101_v1)"


@lru_cache(maxsize=1)
def _compute_reachable_workspace(
    seed: int = 0,
    num_samples: int = 20_000,
    min_ground_clearance: float = 0.05,
    percentile: float = 10.0,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Derive a reachable target-position box from joint limits + forward kinematics.

    Samples random joint configurations within the arm's hard limits, computes
    the end-effector site position for each via FK, discards samples below
    ``min_ground_clearance`` (table/ground penetration), and returns the
    ``[percentile, 100 - percentile]`` box per axis — a reliably reachable
    region rather than the full (mostly unreachable-in-practice) workspace
    envelope. Cached: this compiles a throwaway model, cheap but no reason to
    repeat it if called more than once in a process (e.g. train + play).
    """
    spec = get_spec()
    model = spec.compile()
    data = mujoco.MjData(model)
    site_id = model.site(EE_SITE_NAME).id

    joint_ranges = np.array([j.range for j in spec.joints])
    rng = np.random.default_rng(seed)
    samples = rng.uniform(
        joint_ranges[:, 0], joint_ranges[:, 1], size=(num_samples, len(spec.joints))
    )

    positions = np.zeros((num_samples, 3))
    for i in range(num_samples):
        data.qpos[:] = samples[i]
        mujoco.mj_kinematics(model, data)
        positions[i] = data.site_xpos[site_id]

    positions = positions[positions[:, 2] > min_ground_clearance]
    lo, hi = np.percentile(positions, [percentile, 100.0 - percentile], axis=0)
    return (
        (round(float(lo[0]), 3), round(float(hi[0]), 3)),
        (round(float(lo[1]), 3), round(float(hi[1]), 3)),
        (round(float(lo[2]), 3), round(float(hi[2]), 3)),
    )


def so_arm100_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_reach_env_cfg()

    cfg.scene.entities = {"robot": get_so_arm100_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = SO_ARM100_ACTION_SCALE

    ee_site_cfg = SceneEntityCfg("robot", site_names=(EE_SITE_NAME,))

    ee_pose_error_term = cfg.observations["actor"].terms["ee_pose_error"]
    ee_pose_error_term.params["asset_cfg"] = ee_site_cfg
    cfg.observations["critic"].terms["ee_pose_error"].params["asset_cfg"] = ee_site_cfg

    command_cfg = cfg.commands["ee_pose"]
    assert isinstance(command_cfg, UniformPoseCommandCfg)
    command_cfg.asset_cfg = ee_site_cfg
    x_range, y_range, z_range = _compute_reachable_workspace()
    command_cfg.position_range = UniformPoseCommandCfg.PositionRangeCfg(
        x=x_range, y=y_range, z=z_range
    )

    cfg.rewards["distance_to_target"].params["asset_cfg"] = ee_site_cfg

    assert cfg.scene.sensors is not None
    for sensor in cfg.scene.sensors:
        if sensor.name == "ee_ground_collision":
            assert isinstance(sensor, ContactSensorCfg)
            sensor.primary.pattern = _EE_BODY_PATTERN

    cfg.viewer.body_name = "wrist"

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        # Faster target resampling for more dynamic play.
        assert cfg.commands is not None
        ee_pose_cmd = cfg.commands["ee_pose"]
        assert isinstance(ee_pose_cmd, UniformPoseCommandCfg)
        ee_pose_cmd.resampling_time_range = (3.0, 3.0)

    return cfg
