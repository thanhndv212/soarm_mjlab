#!/usr/bin/env bash
# One-time setup for a rented GPU box (vast.ai or similar): installs uv,
# clones soarm_mjlab (self-contained — no soarm-ws/submodules needed, the
# SO-ARM100 MJCF + meshes are vendored in-repo), and syncs the cu128 extra.
#
# See docs/vast_ai_training.md for the full step-by-step guide this
# script is one step of.
#
# Usage (on the remote box):
#   curl -LsSf https://raw.githubusercontent.com/thanhndv212/soarm_mjlab/main/scripts/setup_remote.sh | bash

set -euo pipefail

REPO_URL="https://github.com/thanhndv212/soarm_mjlab.git"
REPO_DIR="soarm_mjlab"

if ! command -v uv &>/dev/null; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ -d "$REPO_DIR/.git" ]; then
  echo "==> $REPO_DIR already cloned, pulling latest"
  git -C "$REPO_DIR" pull
else
  echo "==> Cloning $REPO_URL"
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "==> Syncing cu128 extra (--locked: fails if uv.lock is stale)"
uv sync --locked --extra cu128 --group dev

cat <<'EOF'

==> Setup complete. Next steps:

  1. Authenticate W&B (paste the API key from https://wandb.ai/authorize):
       cd soarm_mjlab && uv run wandb login

  2. Start a tmux session so training survives an SSH disconnect:
       tmux new -s train

  3. Launch training (inside tmux):
       uv run python scripts/train.py SoArm100-Reach --env.scene.num-envs=4096

  4. Detach with Ctrl-b d; reattach later with: tmux attach -t train

See docs/vast_ai_training.md for monitoring, retrieving the checkpoint, and
shutting the instance down when done.
EOF
