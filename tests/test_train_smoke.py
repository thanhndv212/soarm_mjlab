"""Layer 4: training smoke test — exercises the full RSL-RL wrapper -> PPO
update path via the real scripts/train.py entry point.

Runs as a subprocess (not an in-process call) so it also exercises tyro CLI
parsing, exactly like a user invoking it. ``--gpu-ids None`` forces CPU
regardless of host (this dev machine and CI both lack a CUDA device, and
the default gpu_ids=[0] assumes one exists); ``--agent.logger tensorboard``
avoids a real W&B network call/run per invocation. See
SOARM_MJLAB_ROADMAP.md Phase 2, test pyramid layer 4.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def test_train_smoke(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "SoArm100-Reach",
            "--env.scene.num-envs=4",
            "--agent.max-iterations=2",
            "--gpu-ids",
            "None",
            "--agent.logger",
            "tensorboard",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"train.py exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "Traceback" not in result.stderr
    assert "nan" not in result.stdout.lower()
    assert "Learning iteration 1/2" in result.stdout

    run_dirs = list((tmp_path / "logs" / "rsl_rl" / "so_arm100_reach").glob("*"))
    assert len(run_dirs) == 1, "expected exactly one training run directory"
    run_dir = run_dirs[0]
    assert list(run_dir.glob("model_*.pt")), "no checkpoint written"
    assert list(run_dir.glob("*.onnx")), "no ONNX export written"
    assert (run_dir / "params" / "env.yaml").exists()
    assert (run_dir / "params" / "agent.yaml").exists()
