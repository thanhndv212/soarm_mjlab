# Changelog

## [Unreleased]

### Added

- Phase 0 scaffolding: package layout, `pyproject.toml` (setuptools, flat
  layout, matching `soarm_lerobot`'s convention), pinned `mjlab==1.5.0` /
  `mujoco-warp==3.10.0.2` (verified installable together on macOS arm64),
  CI (lint + package-import smoke test), `LICENSE`, this changelog.
- No task, asset, or training code yet — see `SOARM_MJLAB_ROADMAP.md` in the
  `soarm-ws` root for the phased plan. Phase 1 (the "Reach" sample task)
  is next.
- Switched install/dependency tooling from plain pip to `uv`, matching
  mjlab's own tooling: `cpu`/`cu128` extras forwarding to `mjlab[cpu]`/
  `mjlab[cu128]` (mutually exclusive via `[tool.uv.conflicts]`), `torch`
  routed to the matching PyTorch package index per extra via
  `[tool.uv.sources]` (Linux only — darwin has one torch wheel, no
  CPU/CUDA choice), `dev` dependencies moved from
  `[project.optional-dependencies]` to PEP 735 `[dependency-groups]`,
  `uv.lock` committed for full-tree reproducibility, `Makefile` with
  `sync`/`sync-cpu`/`lint`/`test`/`test-cpu`/`check` targets, CI switched
  to `astral-sh/setup-uv` + `uv sync --locked` (fails the build on
  lockfile drift). Verified locally on macOS arm64 (`sync-cpu`, lint,
  test, import all pass); the Linux CUDA/CPU index-routing path is
  exercised for the first time by CI itself (`ubuntu-latest`), not
  verified on this dev machine.
