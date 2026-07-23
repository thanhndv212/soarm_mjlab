"""RL tasks. Importing this package populates mjlab's task registry
(register_mjlab_task side effects), same convention as mjlab's own
``mjlab.tasks`` and the ``unitree_rl_mjlab`` reference this layout follows.
"""

from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)
