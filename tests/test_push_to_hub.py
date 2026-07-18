"""Unit tests for push_to_hub.py's pure logic — model-card generation and
checkpoint/run-dir resolution. No network, no HfApi/wandb calls (those are
the manual, promotion-gated part described in docs/vast_ai_training.md,
not something to exercise in CI).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "push_to_hub.py"

_spec = importlib.util.spec_from_file_location("push_to_hub", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
push_to_hub = importlib.util.module_from_spec(_spec)
# Register before exec: the dataclass decorator resolves annotations via
# sys.modules[cls.__module__], which must exist first.
sys.modules["push_to_hub"] = push_to_hub
_spec.loader.exec_module(push_to_hub)


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "2026-07-18_12-00-00"
    (run_dir / "params").mkdir(parents=True)
    (run_dir / "params" / "agent.yaml").write_text("max_iterations: 1500\n")
    (run_dir / "params" / "env.yaml").write_text("scene:\n  num_envs: 4096\n")
    (run_dir / "git").mkdir()
    (run_dir / "git" / "soarm_mjlab.diff").write_text(
        "--- git commit ---\nabc1234deadbeef\n\n\n--- git status ---\n...\n"
    )
    return run_dir


def test_build_model_card_includes_provenance(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    cfg = push_to_hub.PushConfig(repo_id="user/soarm100-reach", run_dir=str(run_dir))

    card = push_to_hub.build_model_card(run_dir, cfg)

    assert "1500" in card
    assert "4096" in card
    assert "abc1234deadbeef" in card
    assert "license: mit" in card


def test_build_model_card_includes_wandb_run_when_given(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    cfg = push_to_hub.PushConfig(
        repo_id="user/soarm100-reach", wandb_run_path="entity/project/abc123"
    )

    card = push_to_hub.build_model_card(run_dir, cfg)

    assert "entity/project/abc123" in card


def test_build_model_card_handles_missing_provenance_files(tmp_path):
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    cfg = push_to_hub.PushConfig(repo_id="user/soarm100-reach", run_dir=str(run_dir))

    card = push_to_hub.build_model_card(run_dir, cfg)

    assert "unknown" in card


def test_latest_checkpoint_picks_highest_iteration(tmp_path):
    for name in ("model_100.pt", "model_1500.pt", "model_900.pt"):
        (tmp_path / name).write_bytes(b"")

    result = push_to_hub._latest_checkpoint(tmp_path)

    assert result is not None
    assert result.name == "model_1500.pt"


def test_latest_checkpoint_returns_none_when_empty(tmp_path):
    assert push_to_hub._latest_checkpoint(tmp_path) is None


def test_resolve_run_dir_rejects_neither_source(tmp_path):
    with pytest.raises(ValueError):
        push_to_hub._resolve_run_dir(
            push_to_hub.PushConfig(repo_id="user/soarm100-reach")
        )


def test_resolve_run_dir_rejects_both_sources(tmp_path):
    with pytest.raises(ValueError):
        push_to_hub._resolve_run_dir(
            push_to_hub.PushConfig(
                repo_id="user/soarm100-reach",
                run_dir=str(tmp_path),
                wandb_run_path="entity/project/run",
            )
        )


def test_resolve_run_dir_rejects_missing_local_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        push_to_hub._resolve_run_dir(
            push_to_hub.PushConfig(
                repo_id="user/soarm100-reach", run_dir=str(tmp_path / "does_not_exist")
            )
        )
