"""Shared test fixtures and utilities."""

import os

import torch


def get_test_device() -> str:
    """Get device for testing, preferring CUDA if available.

    Can be overridden with FORCE_CPU=1 environment variable to test
    CPU-only behavior on GPU machines. Matches mjlab's own tests/conftest.py
    convention (see the mjlab repo's ``tests/conftest.py``).
    """
    if os.environ.get("FORCE_CPU") == "1":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
