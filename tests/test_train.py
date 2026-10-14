"""Loss masking, LoRA attachment and the training loop on a tiny CPU model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from lora_finetune_pipeline.config import TrainConfig
from lora_finetune_pipeline.data import Example
from lora_finetune_pipeline.train import (
    IGNORE,
    attach_lora,
    build_features,
    collate,
    count_params,
    for_inference,
    mean_loss,
    total_steps,
    train,
)

from .conftest import synthetic, tiny_model

CPU = torch.device("cpu")
CFG = TrainConfig(
    lora_r=4,
    lora_alpha=8,
    learning_rate=5e-3,
    epochs=3,
    batch_size=8,
    max_seq_len=64,
    gradient_checkpointing=False,
    warmup_ratio=0.1,
    lora_dropout=0.0,
)


def test_only_the_answer_and_end_of_turn_carry_loss(tokenizer: PreTrainedTokenizerFast) -> None:
    features, truncated = build_features(
        tokenizer, [Example("where is my card", "card_arrival")], 64
    )
    f = features[0]
    supervised = [t for t in f.labels if t != IGNORE]
    assert tokenizer.convert_ids_to_tokens(supervised) == ["card_arrival", "<|im_end|>"]
    assert f.input_ids[-2:] == supervised
    assert truncated == 0


def test_long_prompts_are_truncated_but_keep_the_answer(tokenizer: PreTrainedTokenizerFast) -> None:
    features, truncated = build_features(tokenizer, [Example("my " * 100, "top_up")], 32)
    assert truncated == 1
    assert len(features[0].input_ids) == 32
    assert [t for t in features[0].labels if t != IGNORE] == features[0].input_ids[-2:]


def test_collate_pads_right_and_masks_padding(tokenizer: PreTrainedTokenizerFast) -> None:
    features, _ = build_features(
        tokenizer,
        [Example("card", "card_arrival"), Example("my pin is lost please", "lost_pin")],
        64,
    )
    pad = int(tokenizer.pad_token_id or 0)
    batch = collate(features, pad)
    short = len(features[0].input_ids)
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"][0, short:].sum() == 0
    assert (batch["labels"][0, short:] == IGNORE).all()
    assert (batch["input_ids"][0, short:] == pad).all()


def test_lora_trains_only_adapters(tokenizer: PreTrainedTokenizerFast) -> None:
    model = attach_lora(tiny_model(len(tokenizer)), CFG, quantized=False)
    trainable, total = count_params(model)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names
    assert all("lora_" in n for n in names)
    assert 0 < trainable < total * 0.2


def test_training_lowers_validation_loss_and_is_reproducible(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    train_rows, val_rows = synthetic(40, 1), synthetic(5, 2)
    pad = int(tokenizer.pad_token_id or 0)
    tr, _ = build_features(tokenizer, train_rows, 64)
    va, _ = build_features(tokenizer, val_rows, 64)

    def once() -> tuple[float, float, list[tuple[int, float]]]:
        model = attach_lora(tiny_model(len(tokenizer)), CFG, quantized=False)
        before = mean_loss(model, va, pad_id=pad, batch_size=8, device=CPU)
        result = train(
            model, tr, CFG, pad_id=pad, seed=3, device=CPU, log=lambda _: None, log_every=5
        )
        assert result.steps == total_steps(len(tr), CFG) == 45
        return before, mean_loss(model, va, pad_id=pad, batch_size=8, device=CPU), result.loss_curve

    before, after, curve = once()
    assert after < before * 0.8
    assert curve[-1][1] < curve[0][1]
    assert once() == (before, after, curve)


def test_max_steps_overrides_epochs() -> None:
    cfg = TrainConfig(max_steps=7, batch_size=4)
    assert total_steps(1000, cfg) == 7
    assert total_steps(10, TrainConfig(epochs=2.0, batch_size=4)) == 6


class SimulatedCrashError(Exception):
    """Simulated crash."""


def test_resume_from_checkpoint_matches_an_uninterrupted_run(
    tokenizer: PreTrainedTokenizerFast, tmp_path: Path
) -> None:
    pad = int(tokenizer.pad_token_id or 0)
    tr, _ = build_features(tokenizer, synthetic(20, 1), 64)
    cfg = TrainConfig(
        lora_r=4,
        lora_alpha=8,
        learning_rate=5e-3,
        max_steps=12,
        batch_size=4,
        max_seq_len=64,
        gradient_checkpointing=False,
        lora_dropout=0.1,
    )

    def model() -> torch.nn.Module:
        return attach_lora(tiny_model(len(tokenizer)), cfg, quantized=False)

    straight = train(
        model(), tr, cfg, pad_id=pad, seed=3, device=CPU, log=lambda _: None, log_every=2
    )

    def crash_at_step_10(message: str) -> None:
        if message.startswith("step 10/"):
            raise SimulatedCrashError

    ckpt = tmp_path / "checkpoint.pt"
    with pytest.raises(SimulatedCrashError):
        train(
            model(),
            tr,
            cfg,
            pad_id=pad,
            seed=3,
            device=CPU,
            log=crash_at_step_10,
            log_every=2,
            checkpoint=ckpt,
            checkpoint_every=4,
        )
    assert ckpt.exists()
    resumed_model = model()
    resumed = train(
        resumed_model,
        tr,
        cfg,
        pad_id=pad,
        seed=3,
        device=CPU,
        log=lambda _: None,
        log_every=2,
        checkpoint=ckpt,
        checkpoint_every=4,
    )
    assert resumed.resumed_at == 8
    assert resumed.loss_curve == straight.loss_curve
    assert resumed.examples_seen == straight.examples_seen
    assert not ckpt.exists()  # removed once training completes


def test_checkpoint_for_another_model_is_refused(
    tokenizer: PreTrainedTokenizerFast, tmp_path: Path
) -> None:
    pad = int(tokenizer.pad_token_id or 0)
    tr, _ = build_features(tokenizer, synthetic(5, 1), 64)
    cfg = TrainConfig(
        lora_r=4, lora_alpha=8, max_steps=6, batch_size=4, gradient_checkpointing=False
    )
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"step": 2, "adapter": {"other.weight": torch.zeros(1)}}, ckpt)
    with pytest.raises(ValueError, match="does not match"):
        train(
            attach_lora(tiny_model(len(tokenizer)), cfg, quantized=False),
            tr,
            cfg,
            pad_id=pad,
            seed=0,
            device=CPU,
            log=lambda _: None,
            checkpoint=ckpt,
        )


def test_for_inference_turns_the_kv_cache_back_on(tokenizer: PreTrainedTokenizerFast) -> None:
    cfg = TrainConfig(lora_r=4, lora_alpha=8, gradient_checkpointing=True)
    model = attach_lora(tiny_model(len(tokenizer)), cfg, quantized=False)
    base: Any = model.get_base_model()
    base.config.use_cache = False  # what transformers does when training starts
    assert base.is_gradient_checkpointing
    for_inference(model)
    assert base.config.use_cache is True
    assert not base.is_gradient_checkpointing
    assert not model.training
