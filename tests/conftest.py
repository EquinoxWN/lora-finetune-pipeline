"""Tiny offline components: a word-level chat tokenizer, a random Qwen2 model and synthetic intents."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # unit tests must never need the GPU

import random
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from lora_finetune_pipeline import env
from lora_finetune_pipeline.data import Example
from lora_finetune_pipeline.prompts import TASK

CHAT = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }} {{ m['content'] }}<|im_end|> {% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}"
)
SPECIAL = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"]
# Three intents, each signalled by its own keyword; filler words are shared.
INTENTS = {"card_arrival": "card", "lost_pin": "pin", "top_up": "topup"}
FILLER = ["my", "the", "is", "where", "please", "help", "still", "not", "a", "with", "i", "need"]


def synthetic(n_per_class: int, seed: int) -> list[Example]:
    """Planted data: the keyword decides the intent."""
    rng = random.Random(seed)
    rows = []
    for label, word in INTENTS.items():
        for _ in range(n_per_class):
            words = [*rng.sample(FILLER, 4), word]
            rng.shuffle(words)
            rows.append(Example(" ".join(words), label))
    rng.shuffle(rows)
    return rows


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level tokenizer over the synthetic vocabulary with a ChatML-like template."""
    words = sorted(set(TASK.split()) | set(FILLER) | set(INTENTS) | set(INTENTS.values()))
    words += ["system", "user", "assistant", "Allowed", "intents:"] + [
        f"{label}," for label in INTENTS
    ]
    vocab = {tok: i for i, tok in enumerate(SPECIAL + sorted(set(words) - set(SPECIAL)))}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=core,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tok.chat_template = CHAT
    return tok


def tiny_model(vocab_size: int, seed: int = 0) -> Qwen2ForCausalLM:
    """A two-layer random Qwen2 small enough for CPU tests."""
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config)  # type: ignore[no-untyped-call]


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    """Fresh tiny tokenizer."""
    return tiny_tokenizer()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake repository root whose .tmp is inside pytest's temp folder."""
    monkeypatch.setattr(env, "repo_root", lambda: tmp_path)
    for var in [*env.CACHE_VARS, *env.FLAGS]:
        if var in os.environ:
            monkeypatch.setenv(var, os.environ[var])
        else:
            monkeypatch.delenv(var, raising=False)
    (tmp_path / ".tmp").mkdir()
    return tmp_path
