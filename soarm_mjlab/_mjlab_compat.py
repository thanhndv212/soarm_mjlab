"""Workaround for an mjlab/mujoco_warp compatibility bug: crashes with
num_envs > 1 whenever an entity's joint limits are never touched by a
domain-randomization event.

Root cause (confirmed upstream, not just guessed at): ``mujoco_warp==3.10.0.2``
(released 2026-07-13, the version this package is pinned to) changed how it
allocates per-world-shared model fields like ``jnt_range`` — from stride-0
broadcast views to real materialized size-1 arrays. mjlab's ``TorchArray``
(``mjlab/sim/sim_data.py``) detects "needs expanding to nworld" via
``stride(0) == 0``, so it silently stopped expanding these fields, and any
per-env indexing of the resulting buffers (e.g.
``EntityData.soft_joint_pos_limits`` inside
``mjlab.envs.mdp.reset_joints_by_offset``) raises ``IndexError: index i is
out of bounds for dimension 0 with size 1`` for any ``num_envs > 1``.

Reproduces identically on mjlab's own bundled ``Mjlab-Lift-Cube-Yam`` task
(confirmed 2026-07-14) — not specific to soarm_mjlab's Reach task. Already
reported and fixed upstream as of this writing:
https://github.com/mujocolab/mjlab/pull/1091 ("Fix TorchArray world-dim
expansion for mujoco_warp 3.10.0.2") — opened as a draft the same day,
independently, by someone else who hit the identical error. That PR is the
real fix (it corrects ``TorchArray`` generally, for every model/data field,
not just the three below); this module is a narrower stopgap for our own
pinned versions until it merges and mjlab cuts a release with it.

Safe to delete this module (and its import in ``soarm_mjlab/tasks/__init__.py``)
once that lands: the guard below is a no-op if the tensors are already
broadcast to ``nworld``.
"""

from __future__ import annotations

from mjlab.entity.entity import Entity

_JOINT_LIMIT_FIELDS = (
    "default_joint_pos_limits",
    "joint_pos_limits",
    "soft_joint_pos_limits",
)

_original_initialize = Entity.initialize


def _initialize_with_broadcast_fix(self: Entity, mj_model, model, data, device) -> None:
    _original_initialize(self, mj_model, model, data, device)
    nworld = data.nworld
    if nworld <= 1:
        return
    for name in _JOINT_LIMIT_FIELDS:
        tensor = getattr(self.data, name)
        if tensor.shape[0] == 1:
            setattr(self.data, name, tensor.expand(nworld, -1, -1).contiguous())


Entity.initialize = _initialize_with_broadcast_fix
