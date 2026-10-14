"""Download the pinned base model and load it in 4-bit NF4 for QLoRA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .config import ModelPin
from .data import sha256_file

MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
]


class IntegrityError(RuntimeError):
    """Downloaded weights do not match the pinned hash."""


def fetch(pin: ModelPin, cache_dir: Path) -> Path:
    """Snapshot the pinned revision (only the files needed) and verify the weights' SHA-256."""
    from huggingface_hub import snapshot_download

    local = Path(
        snapshot_download(
            repo_id=pin.repo_id,
            revision=pin.revision,
            cache_dir=str(cache_dir),
            allow_patterns=[*MODEL_FILES, pin.weights_file],
        )
    )
    weights = local / pin.weights_file
    if sha256_file(weights) != pin.weights_sha256:
        raise IntegrityError(f"{weights} does not match the pinned SHA-256")
    return local


def load_tokenizer(path: Path) -> PreTrainedTokenizerBase:
    """Tokenizer with a pad token, padding on the left for generation."""
    tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_4bit(path: Path) -> Any:
    """The base model quantized to 4-bit NF4 with double quantization, bf16 compute, on GPU 0."""
    from transformers import BitsAndBytesConfig

    quant = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = True
    return model
