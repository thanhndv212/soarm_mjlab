"""PPO runner for the Reach task, with ONNX export on checkpoint save.

Mirrors mjlab's own per-task runner subclasses (e.g.
``mjlab.tasks.manipulation.rl.ManipulationOnPolicyRunner``) so a trained
checkpoint is always paired with an ONNX export — what Phase 5's
``deploy/reach_policy_runner.py`` will load. The raw ``.pt`` checkpoint is
already uploaded to W&B automatically by rsl_rl's own ``WandbLogWriter``
whenever ``--agent.logger wandb`` (the default) is active; this runner adds
the ONNX export to that same upload so both artifacts are retrievable from
a W&B run — e.g. by ``scripts/play.py --wandb-run-path`` on a machine that
never ran the training job itself (see docs/vast_ai_training.md).
"""

from __future__ import annotations

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class ReachOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        super().save(path, infos)
        policy_dir, filename, onnx_path = self._get_export_paths(path)
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
            has_active_run = wandb is not None and wandb.run is not None
            run_path = wandb.run.name if has_active_run else "local"
            metadata = get_base_metadata(self.env.unwrapped, run_path=run_path)
            attach_metadata_to_onnx(str(onnx_path), metadata)
            if has_active_run:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")
