"""Pinned inputs and validated run settings."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
MAX_CONFIG_BYTES = 64 * 1024
_T = TypeVar("_T")


@dataclass(frozen=True)
class RemoteFile:
    """A file fetched over HTTPS and accepted only if its size and SHA-256 match."""

    url: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        """Validate the pin."""
        if not self.url.startswith("https://"):
            raise ValueError(f"only HTTPS downloads are allowed: {self.url}")
        if not SHA256.match(self.sha256):
            raise ValueError(f"bad sha256 for {self.url}")
        if not 0 < self.size <= 50 * 1024 * 1024:
            raise ValueError(f"size out of range for {self.url}")


@dataclass(frozen=True)
class ModelPin:
    """A Hugging Face model repo pinned to one commit and one weights file hash."""

    repo_id: str
    revision: str
    weights_file: str
    weights_sha256: str
    licence: str

    def __post_init__(self) -> None:
        """Validate the pin."""
        if not REPO_ID.match(self.repo_id):
            raise ValueError(f"bad repo id: {self.repo_id}")
        if not SHA1.match(self.revision):
            raise ValueError("revision must be a full 40-character commit SHA")
        if not SHA256.match(self.weights_sha256):
            raise ValueError("bad weights sha256")


# Qwen2.5-0.5B-Instruct, Apache-2.0, commit of 2024-09-25; weights SHA-256 from the Hub's LFS pointer.
MODEL = ModelPin(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    weights_file="model.safetensors",
    weights_sha256="fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    licence="Apache-2.0",
)

# BANKING77 (Casanueva et al., 2020), CC BY 4.0, PolyAI-LDN/task-specific-datasets at commit 57ec275.
_DATA_BASE = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/"
)
DATASET_LICENCE = "CC-BY-4.0"
DATASET_FILES: dict[str, RemoteFile] = {
    "train": RemoteFile(
        _DATA_BASE + "train.csv",
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
        839073,
    ),
    "test": RemoteFile(
        _DATA_BASE + "test.csv",
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
        239961,
    ),
}


@dataclass(frozen=True)
class TrainConfig:
    """QLoRA hyperparameters."""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    learning_rate: float = 2e-4
    epochs: float = 1.0
    max_steps: int = 0
    batch_size: int = 16
    grad_accum: int = 1
    warmup_ratio: float = 0.05
    max_seq_len: int = 128
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        """Reject values that cannot train."""
        _check(1 <= self.lora_r <= 256, "lora_r must be in 1..256")
        _check(self.lora_alpha >= 1, "lora_alpha must be >= 1")
        _check(0.0 <= self.lora_dropout < 1.0, "lora_dropout must be in [0, 1)")
        _check(len(self.target_modules) > 0, "target_modules must not be empty")
        _check(0.0 < self.learning_rate < 1.0, "learning_rate must be in (0, 1)")
        _check(self.epochs > 0 or self.max_steps > 0, "epochs or max_steps must be positive")
        _check(self.max_steps >= 0, "max_steps must be >= 0")
        _check(1 <= self.batch_size <= 256, "batch_size must be in 1..256")
        _check(1 <= self.grad_accum <= 64, "grad_accum must be in 1..64")
        _check(0.0 <= self.warmup_ratio < 1.0, "warmup_ratio must be in [0, 1)")
        _check(16 <= self.max_seq_len <= 4096, "max_seq_len must be in 16..4096")
        _check(self.weight_decay >= 0.0, "weight_decay must be >= 0")


@dataclass(frozen=True)
class EvalConfig:
    """How the held-out split is scored."""

    batch_size: int = 16
    max_new_tokens: int = 16
    few_shot_per_class: int = 1
    limit_per_class: int = 0

    def __post_init__(self) -> None:
        """Reject values that cannot evaluate."""
        _check(1 <= self.batch_size <= 256, "batch_size must be in 1..256")
        _check(1 <= self.max_new_tokens <= 64, "max_new_tokens must be in 1..64")
        _check(0 <= self.few_shot_per_class <= 5, "few_shot_per_class must be in 0..5")
        _check(self.limit_per_class >= 0, "limit_per_class must be >= 0")


@dataclass(frozen=True)
class RunConfig:
    """One complete comparison run."""

    name: str
    seed: int = 1234
    val_per_class: int = 10
    vram_fraction: float = 0.9
    min_free_ram_mib: int = 1500
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        """Reject unsafe names and out-of-range limits."""
        _check(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", self.name) is not None, "bad run name")
        _check(0 <= self.seed < 2**32, "seed must fit in 32 bits")
        _check(1 <= self.val_per_class <= 50, "val_per_class must be in 1..50")
        _check(0.1 <= self.vram_fraction <= 1.0, "vram_fraction must be in [0.1, 1]")
        _check(self.min_free_ram_mib >= 0, "min_free_ram_mib must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready form."""
        return asdict(self)


def _check(ok: bool, message: str) -> None:
    """Raise ValueError unless ok."""
    if not ok:
        raise ValueError(message)


def _build(cls: type[_T], raw: Any, where: str) -> _T:
    """Build a dataclass from a dict, refusing unknown keys."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object")
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown keys in {where}: {sorted(unknown)}")
    values = dict(raw)
    if "target_modules" in values:
        values["target_modules"] = tuple(values["target_modules"])
    return cls(**values)


def parse_run_config(raw: Any) -> RunConfig:
    """Validate a decoded JSON config."""
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    top = dict(raw)
    train = _build(TrainConfig, top.pop("train", {}), "train")
    ev = _build(EvalConfig, top.pop("eval", {}), "eval")
    run = _build(RunConfig, top, "config")
    return RunConfig(**{**asdict(run), "train": train, "eval": ev})


def load_run_config(path: Path) -> RunConfig:
    """Read and validate a JSON config file."""
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError(f"{path} is larger than {MAX_CONFIG_BYTES} bytes")
    return parse_run_config(json.loads(path.read_text(encoding="utf-8")))
