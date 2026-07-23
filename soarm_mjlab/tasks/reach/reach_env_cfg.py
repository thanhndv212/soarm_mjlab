"""Base "Reach" task config: random end-effector target, single SO-ARM100."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation.mdp import illegal_contact
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from soarm_mjlab.tasks.reach import mdp
from soarm_mjlab.tasks.reach.mdp import UniformPoseCommandCfg


def make_reach_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create the base Reach task configuration."""

    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "target_pose": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"},
        ),
        "ee_pose_error": ObservationTermCfg(
            func=mdp.ee_pose_error,
            params={
                "command_name": "ee_pose",
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    critic_terms = {**actor_terms}

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Override per-robot.
            use_default_offset=True,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "ee_pose": UniformPoseCommandCfg(
            entity_name="robot",
            asset_cfg=SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            resampling_time_range=(4.0, 6.0),
            debug_vis=True,
            success_threshold=0.03,
            success_steps=10,
        )
    }

    events = {
        # For positioning the base of the robot at env_origins.
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    # Collision sensor for end-effector to ground contact.
    ee_ground_collision_cfg = ContactSensorCfg(
        name="ee_ground_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern="",  # Set per-robot.
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    rewards = {
        "distance_to_target": RewardTermCfg(
            func=mdp.distance_to_target,
            weight=2.0,
            params={
                "command_name": "ee_pose",
                "orientation_weight": 0.5,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
        "distance_to_target_shaped": RewardTermCfg(
            func=mdp.distance_to_target_shaped,
            weight=5.0,
            params={
                "command_name": "ee_pose",
                "sigma": 0.10,
                "orientation_weight": 0.5,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
        "success_bonus": RewardTermCfg(
            func=mdp.success_bonus,
            weight=10.0,
            params={
                "command_name": "ee_pose",
                "threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "task_success": TerminationTermCfg(
            func=mdp.task_success, params={"command_name": "ee_pose"}
        ),
        "ee_ground_collision": TerminationTermCfg(
            func=illegal_contact,
            params={"sensor_name": "ee_ground_collision", "force_threshold": 10.0},
        ),
    }

    curriculum: dict[str, CurriculumTermCfg] = {
        "success_threshold_curriculum": CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={
                "reward_name": "success_bonus",
                # common_step_counter increments once per env.step() call
                # (num_steps_per_env=24/iteration), so with max_iterations=1500
                # the counter tops out at 36000. Tighten threshold roughly at
                # the 1/3 and 2/3 marks of training.
                "stages": [
                    {"step": 0, "params": {"threshold": 0.05}},
                    {"step": 12000, "params": {"threshold": 0.04}},
                    {"step": 24000, "params": {"threshold": 0.03}},
                ],
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        curriculum=curriculum,
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=1.0,
            sensors=(ee_ground_collision_cfg,),
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=1.0,
            elevation=-20.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=30,
            njmax=200,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=10.0,
    )
