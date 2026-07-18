"""SO-ARM100 Reach-task registration (register_mjlab_task)."""

from mjlab.tasks.registry import register_mjlab_task
from soarm_mjlab.tasks.reach.rl import ReachOnPolicyRunner

from .env_cfgs import so_arm100_reach_env_cfg
from .rl_cfg import so_arm100_reach_ppo_runner_cfg

register_mjlab_task(
    task_id="SoArm100-Reach",
    env_cfg=so_arm100_reach_env_cfg(),
    play_env_cfg=so_arm100_reach_env_cfg(play=True),
    rl_cfg=so_arm100_reach_ppo_runner_cfg(),
    runner_cls=ReachOnPolicyRunner,
)
