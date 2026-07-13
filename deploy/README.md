# deploy

`reach_policy_runner.py` — loads a trained checkpoint (ONNX or torch) and
drives `soarm_sdk.RobotInterface` directly (sim or real, same interface).
No C++ stack — see the root README for why. Phase 5 — not implemented yet.
