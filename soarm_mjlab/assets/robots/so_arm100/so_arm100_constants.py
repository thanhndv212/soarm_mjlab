"""SO-ARM100 constants: MJCF, actuator gains, home keyframe, collision config.

MJCF and meshes are vendored (not referenced across the submodule boundary,
same convention mjlab itself uses for every ``asset_zoo`` robot) from
``SO-ARM100/Simulation/SO101/so101_new_calib.xml`` — the SO-ARM100 repo only
ships a ready-to-compile MuJoCo model for the SO101 revision (SO100 there is
URDF-only). Both revisions use the same STS3215 servos and the joint set
matches ``soarm_sdk``'s ``configs/soarm100.yaml`` 1:1 (see the name-mapping
comment below), so it stands in for "SO-ARM100" here.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

SO_ARM100_XML: Path = Path(__file__).parent / "xmls" / "so_arm100.xml"
assert SO_ARM100_XML.exists()

# soarm_sdk's configs/soarm100.yaml joint_names, in servo-ID order, mapped to
# this MJCF's joint names (onshape-to-robot's own naming, unchanged so the
# vendored file stays a clean diff against upstream).
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",  # Rotation
    "shoulder_lift",  # Pitch
    "elbow_flex",  # Elbow
    "wrist_flex",  # Wrist_Pitch
    "wrist_roll",  # Wrist_Roll
    "gripper",  # Jaw
)

# End-effector reference frame already defined in the MJCF (site on the
# "gripper" body, positioned at the jaw tip) — no custom site needed.
EE_SITE_NAME = "gripperframe"


def get_spec() -> mujoco.MjSpec:
    """Load the SO-ARM100 spec with actuators stripped and collision geoms named.

    The source MJCF ships its own ``<position>`` actuators (TheRobotStudio's
    own sim defaults) and unnamed collision geoms (one per body, MuJoCo
    auto-names them empty-string). Both are replaced/fixed here rather than by
    hand-editing the vendored file, so it stays a clean diff against upstream:
    actuators are rebuilt from ``ARTICULATION`` below (this package's own gain
    source), and collision geoms get ``{body}_collision_{i}`` names so
    ``CollisionCfg``/``SceneEntityCfg`` regexes can address them.
    """
    spec = mujoco.MjSpec.from_file(str(SO_ARM100_XML))
    for act in list(spec.actuators):
        spec.delete(act)
    for body in spec.bodies:
        idx = 0
        for geom in body.geoms:
            if geom.classname and geom.classname.name == "collision":
                geom.name = f"{body.name}_collision_{idx}"
                idx += 1
    return spec


##
# Actuator config.
##

# STS3215 servo gains, taken from the MJCF's own ``sts3215`` default class
# (so101_new_calib.xml, actually compiled/used by TheRobotStudio's own
# scene.xml) rather than the sibling ``joints_properties.xml`` file in the
# same directory, which defines a *different* set of numbers for the same
# class name and is not included by any scene — it reads as a leftover
# alternative, not the config in active use.
#
# kp/kv are MuJoCo <position> gains (stiffness/damping below); forcerange is
# the effort limit; joint damping/frictionloss/armature are passive dynamics.
# The XML comment documents kp/kv as derived (not datasheet values) assuming
# a servo proportional gain of 16 — see so_arm100.xml's `sts3215` default
# class for the full derivation note.
STS3215_STIFFNESS = 998.22
STS3215_DAMPING = 2.731
STS3215_EFFORT_LIMIT = 2.94
STS3215_VISCOUS_DAMPING = 0.60
STS3215_FRICTIONLOSS = 0.052
STS3215_ARMATURE = 0.028

ARM_ACTUATORS = tuple(
    BuiltinPositionActuatorCfg(
        target_names_expr=(name,),
        stiffness=STS3215_STIFFNESS,
        damping=STS3215_DAMPING,
        effort_limit=STS3215_EFFORT_LIMIT,
        viscous_damping=STS3215_VISCOUS_DAMPING,
        frictionloss=STS3215_FRICTIONLOSS,
        armature=STS3215_ARMATURE,
    )
    for name in JOINT_NAMES
)

##
# Keyframe config.
##

# soarm_sdk's configs/soarm100.yaml home_position, in JOINT_NAMES order. Base
# raised 3cm off the ground plane (mirrors mjlab's own YAM mount-height
# convention) — at pos=(0,0,0), this home pose puts the gripper's collision
# geoms ~4-6cm above the floor, and reset_robot_joints' random offset (see
# reach_env_cfg.py) is comfortably enough to dip that under 0 and trigger an
# immediate ee_ground_collision termination on a sizeable fraction of resets.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.03),
    joint_pos={
        "shoulder_pan": 0.0,
        "shoulder_lift": -0.3,
        "elbow_flex": 0.5,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Only the gripper/jaw geoms collide with the environment (e.g. the ground
# plane) — enough for a reach task's illegal-contact termination without
# paying for self-collision across the rest of the arm. Mirrors mjlab's own
# GRIPPER_ONLY_COLLISION pattern for the YAM manipulator.
GRIPPER_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision.*",),
    contype={
        "(gripper|moving_jaw_so101_v1)_collision.*": 1,
        ".*_collision.*": 0,
    },
    conaffinity={
        "(gripper|moving_jaw_so101_v1)_collision.*": 1,
        ".*_collision.*": 0,
    },
    condim={".*_collision.*": 3},
    friction={".*_collision.*": (0.6,)},
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
    actuators=ARM_ACTUATORS,
    soft_joint_pos_limit_factor=0.95,
)


def get_so_arm100_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(GRIPPER_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


# Per-joint action scale (rad per action unit). The previous formula
# (0.25 * effort_limit / stiffness = 0.000736 rad) was so small the policy
# could only move joints 0.04 deg per step -- physically impossible to reach
# targets 10-47cm away. These values give the policy meaningful control
# authority while staying within joint limits from the home keyframe.
SO_ARM100_ACTION_SCALE: dict[str, float] = {
    "shoulder_pan": 1.0,
    "shoulder_lift": 1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_roll": 1.0,
    "gripper": 0.15,
}

# Per-joint residual scale (rad) for ResidualIKAction. Much smaller than
# SO_ARM100_ACTION_SCALE because the DLS IK base controller handles coarse
# motion toward the target — the residual is only for fine corrections
# (IK imperfections, joint-limit edge cases, dynamic effects). The policy
# output is clipped to [-1, 1] by rl_cfg's clip_actions, then scaled by
# these values, so the max per-step residual is ±0.1 rad (arm) / ±0.05
# (gripper) — enough for fine positioning, not enough to override the
# base controller's coarse motion.
SO_ARM100_RESIDUAL_SCALE: dict[str, float] = {
    "shoulder_pan": 0.1,
    "shoulder_lift": 0.1,
    "elbow_flex": 0.1,
    "wrist_flex": 0.1,
    "wrist_roll": 0.1,
    "gripper": 0.05,
}


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_so_arm100_robot_cfg())
    viewer.launch(robot.spec.compile())
