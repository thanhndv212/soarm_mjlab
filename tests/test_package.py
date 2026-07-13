"""Phase 0 sanity check: the package installs and imports.

Real coverage (mdp unit tests, config/asset validation, env smoke test,
training smoke test — the Phase 2 test pyramid) lands with the Phase 1
Reach task; there's no task/asset code yet for those to exercise.
"""

from __future__ import annotations

import soarm_mjlab


def test_package_imports() -> None:
    assert soarm_mjlab.__all__ == []
