"""Reach-task MDP terms: observations, rewards, terminations, commands.

Re-exports mjlab's generic MDP library (joint_pos_rel, action_rate_l2,
joint_pos_limits, time_out, reset_root_state_uniform, ...) alongside the
Reach-specific terms below, same convention as mjlab's own task packages
(e.g. ``mjlab.tasks.velocity.mdp``).
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
