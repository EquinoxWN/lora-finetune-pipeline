"""Every cache variable points inside the repository's .tmp folder."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lora_finetune_pipeline import env


def test_configure_points_every_cache_inside_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in env.CACHE_VARS:
        monkeypatch.setenv(var, "C:/Users/someone/.cache")
    applied = env.configure(tmp_path)
    base = (tmp_path / ".tmp").resolve()
    for var in env.CACHE_VARS:
        assert Path(os.environ[var]).resolve().is_relative_to(base), var
        assert applied[var] == os.environ[var]
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["WANDB_DISABLED"] == "true"


def test_required_variables_are_covered() -> None:
    required = {
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "MPLCONFIGDIR",
    }
    assert required <= set(env.CACHE_VARS)


def test_ensure_inside_refuses_traversal(tmp_path: Path) -> None:
    assert env.ensure_inside(tmp_path / "a" / "b", tmp_path) == (tmp_path / "a" / "b").resolve()
    with pytest.raises(ValueError, match="outside"):
        env.ensure_inside(tmp_path / ".." / "elsewhere", tmp_path)
