"""Keep every cache, download and temp file inside the repository's .tmp folder."""

from __future__ import annotations

import os
from pathlib import Path

# Variable -> path under .tmp. Set before torch, transformers or huggingface_hub are imported.
CACHE_VARS: dict[str, str] = {
    "HF_HOME": "hf",
    "HF_HUB_CACHE": "hf/hub",
    "HF_XET_CACHE": "hf/xet",
    "HF_DATASETS_CACHE": "hf/datasets",
    "TRANSFORMERS_CACHE": "hf/hub",
    "TORCH_HOME": "torch",
    "TRITON_CACHE_DIR": "triton",
    "CUDA_CACHE_PATH": "cuda-cache",
    "MPLCONFIGDIR": "mpl",
    "TMP": "",
    "TEMP": "",
    "TMPDIR": "",
}
FLAGS: dict[str, str] = {
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def repo_root() -> Path:
    """The repository root (two levels above this package)."""
    return Path(__file__).resolve().parents[2]


def tmp_dir(root: Path | None = None) -> Path:
    """The git-ignored scratch folder."""
    return (root or repo_root()) / ".tmp"


def configure(root: Path | None = None, *, override: bool = True) -> dict[str, str]:
    """Point every known cache variable inside .tmp and return what was set."""
    base = tmp_dir(root)
    applied: dict[str, str] = {}
    for var, sub in CACHE_VARS.items():
        path = base / sub if sub else base
        path.mkdir(parents=True, exist_ok=True)
        if override or var not in os.environ:
            os.environ[var] = str(path)
        applied[var] = os.environ[var]
    for var, value in FLAGS.items():
        os.environ[var] = value
        applied[var] = value
    return applied


def ensure_inside(path: Path, root: Path) -> Path:
    """Resolve a path and refuse it unless it stays inside root."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{path} is outside {root}")
    return resolved
