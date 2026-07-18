"""Phase 0 sanity check: the package installs and imports.

The Phase 2 test pyramid (mdp unit tests, config/asset validation, env
smoke test, training smoke test) is separate, not-yet-implemented work —
see SOARM_MJLAB_ROADMAP.md — even though the Phase 1 Reach task these tests
would exercise now exists.
"""

from __future__ import annotations

import soarm_mjlab


def test_package_imports() -> None:
    assert soarm_mjlab.__all__ == []
