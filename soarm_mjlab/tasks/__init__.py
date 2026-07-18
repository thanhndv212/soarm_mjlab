"""RL tasks. Importing this package populates mjlab's task registry
(register_mjlab_task side effects), same convention as mjlab's own
``mjlab.tasks`` and the ``unitree_rl_mjlab`` reference this layout follows.

mjlab is an optional dependency (the ``cpu``/``cu128`` extras) — this is the
first place it's actually needed, so the ``_mjlab_compat`` patch (see that
module) is applied here rather than in the top-level ``soarm_mjlab``
package, which must stay importable without mjlab installed.
"""

import soarm_mjlab._mjlab_compat  # noqa: F401
from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)
