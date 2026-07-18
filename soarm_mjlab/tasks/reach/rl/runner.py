"""PPO runner for the Reach task, with ONNX export on checkpoint save.

Mirrors mjlab's own per-task runner subclasses (e.g.
``mjlab.tasks.manipulation.rl.ManipulationOnPolicyRunner``) so a trained
checkpoint is always paired with an ONNX export — what Phase 5's
``deploy/reach_policy_runner.py`` will load. No W&B upload here (unlike
mjlab's own runners): kept out until this project actually wires up W&B.
"""

from __future__ import annotations

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner


class ReachOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        super().save(path, infos)
        policy_dir, filename, onnx_path = self._get_export_paths(path)
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
            metadata = get_base_metadata(self.env.unwrapped, run_path="local")
            attach_metadata_to_onnx(str(onnx_path), metadata)
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")
