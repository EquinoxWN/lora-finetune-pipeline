"""QLoRA training: masked chat features, LoRA attachment and a small, explicit training loop."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import TrainConfig
from .data import Example
from .prompts import fine_tuned

IGNORE = -100


@dataclass(frozen=True)
class Feature:
    """Token ids for one example; labels are IGNORE everywhere except the answer."""

    input_ids: list[int]
    labels: list[int]


@dataclass
class TrainResult:
    """What a training run did and how long it took."""

    steps: int = 0
    examples_seen: int = 0
    tokens_seen: int = 0
    seconds: float = 0.0
    loss_curve: list[tuple[int, float]] = field(default_factory=list)
    truncated: int = 0
    trainable_params: int = 0
    total_params: int = 0
    resumed_at: int = 0


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def render(tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]]) -> str:
    """The chat template applied up to the assistant's turn."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(text, str):
        raise TypeError("chat template did not return text")
    return text


def answer_suffix(tokenizer: PreTrainedTokenizerBase) -> str:
    """The token that ends an assistant turn."""
    eos = tokenizer.eos_token
    if not isinstance(eos, str) or not eos:
        raise ValueError("tokenizer has no eos token")
    return eos


def build_features(
    tokenizer: PreTrainedTokenizerBase, examples: Sequence[Example], max_seq_len: int
) -> tuple[list[Feature], int]:
    """Tokenise prompt + answer; only the answer and end-of-turn token carry loss."""
    suffix = answer_suffix(tokenizer)
    features: list[Feature] = []
    truncated = 0
    for e in examples:
        prompt_ids: list[int] = tokenizer(
            render(tokenizer, fine_tuned(e.text)), add_special_tokens=False
        )["input_ids"]
        answer_ids: list[int] = tokenizer(e.label + suffix, add_special_tokens=False)["input_ids"]
        room = max_seq_len - len(answer_ids)
        if room < 8:
            raise ValueError("max_seq_len too small for the answer")
        if len(prompt_ids) > room:
            truncated += 1
            prompt_ids = prompt_ids[:room]
        features.append(Feature(prompt_ids + answer_ids, [IGNORE] * len(prompt_ids) + answer_ids))
    return features, truncated


def collate(batch: Sequence[Feature], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch; padding is masked from attention and loss."""
    width = max(len(f.input_ids) for f in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, f in enumerate(batch):
        n = len(f.input_ids)
        ids[i, :n] = torch.tensor(f.input_ids)
        labels[i, :n] = torch.tensor(f.labels)
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def lora_config(cfg: TrainConfig) -> LoraConfig:
    """PEFT LoRA settings from the run config."""
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def attach_lora(model: PreTrainedModel, cfg: TrainConfig, *, quantized: bool) -> PeftModel:
    """Freeze the base model and add trainable LoRA adapters."""
    if quantized:
        model = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()  # type: ignore[no-untyped-call]
    peft_model = get_peft_model(model, lora_config(cfg))
    if not isinstance(peft_model, PeftModel):
        raise TypeError("get_peft_model did not return a PeftModel")
    return peft_model


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def total_steps(n_examples: int, cfg: TrainConfig) -> int:
    """Optimizer steps for the configured epochs, or max_steps if set."""
    per_epoch = math.ceil(n_examples / (cfg.batch_size * cfg.grad_accum))
    return cfg.max_steps if cfg.max_steps > 0 else max(1, math.ceil(per_epoch * cfg.epochs))


def train(
    model: torch.nn.Module,
    features: Sequence[Feature],
    cfg: TrainConfig,
    *,
    pad_id: int,
    seed: int,
    device: torch.device,
    log: Callable[[str], None] = print,
    log_every: int = 25,
    checkpoint: Path | None = None,
    checkpoint_every: int = 50,
) -> TrainResult:
    """AdamW with linear warmup and decay over the LoRA parameters; resumes from a checkpoint."""
    seed_everything(seed)
    result = TrainResult()
    result.trainable_params, result.total_params = count_params(model)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    if not params:
        raise ValueError("model has no trainable parameters")
    steps = total_steps(len(features), cfg)
    warmup = int(steps * cfg.warmup_ratio)
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: _lr_factor(s, warmup, steps))
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    order = _batches(len(features), cfg.batch_size, seed)
    first, elapsed = 1, 0.0
    if checkpoint is not None and checkpoint.exists():
        first, elapsed = _restore(checkpoint, named, optimizer, scheduler, result, steps)
        for _ in range((first - 1) * cfg.grad_accum):
            next(order)
        result.resumed_at = first - 1
        log(f"resumed from {checkpoint.name} after step {first - 1}/{steps}")
    model.train()
    start = time.perf_counter()
    running: list[float] = []
    for step in range(first, steps + 1):
        for _ in range(cfg.grad_accum):
            idx = next(order)
            batch = {
                k: v.to(device) for k, v in collate([features[i] for i in idx], pad_id).items()
            }
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out: Any = model(**batch)
            loss = out.loss / cfg.grad_accum
            loss.backward()
            running.append(float(out.loss.detach()))
            result.examples_seen += len(idx)
            result.tokens_seen += int(batch["attention_mask"].sum())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step % log_every == 0 or step == steps:
            mean = sum(running) / len(running)
            result.loss_curve.append((step, round(mean, 4)))
            running = []
            if checkpoint is not None and step % checkpoint_every == 0 and step < steps:
                _save(
                    checkpoint,
                    step,
                    elapsed + time.perf_counter() - start,
                    named,
                    optimizer,
                    scheduler,
                    result,
                )
            log(f"step {step}/{steps} loss {mean:.4f}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    result.seconds = elapsed + time.perf_counter() - start
    result.steps = steps
    if checkpoint is not None:
        checkpoint.unlink(missing_ok=True)
    model.eval()
    return result


def _save(
    path: Path,
    step: int,
    elapsed: float,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    result: TrainResult,
) -> None:
    """Write adapter weights, optimizer, schedule, RNG and progress atomically (tensors only)."""
    state = {
        "step": step,
        "elapsed": elapsed,
        "adapter": {n: p.detach().cpu() for n, p in named},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loss_curve": [list(x) for x in result.loss_curve],
        "examples_seen": result.examples_seen,
        "tokens_seen": result.tokens_seen,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    torch.save(state, tmp)
    tmp.replace(path)


def _restore(
    path: Path,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    result: TrainResult,
    steps: int,
) -> tuple[int, float]:
    """Load a checkpoint written by _save (weights_only: no pickled code) and return (next step, elapsed)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    names = {n for n, _ in named}
    if set(state["adapter"]) != names or not 0 < int(state["step"]) < steps:
        raise ValueError(f"{path} does not match this model and config")
    with torch.no_grad():
        for n, p in named:
            p.copy_(state["adapter"][n].to(p.device))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    result.loss_curve = [(int(a), float(b)) for a, b in state["loss_curve"]]
    result.examples_seen = int(state["examples_seen"])
    result.tokens_seen = int(state["tokens_seen"])
    return int(state["step"]) + 1, float(state["elapsed"])


def for_inference(model: Any) -> None:
    """Undo training-time settings: gradient checkpointing off and the KV cache back on."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if getattr(base, "is_gradient_checkpointing", False):
        base.gradient_checkpointing_disable()
    base.config.use_cache = True
    model.eval()


def _lr_factor(step: int, warmup: int, steps: int) -> float:
    """Linear warmup then linear decay to zero."""
    if warmup and step < warmup:
        return (step + 1) / warmup
    return max(0.0, (steps - step) / max(1, steps - warmup))


def _batches(n: int, batch_size: int, seed: int) -> Any:
    """Endless seeded shuffled batches of indices (reshuffled every epoch)."""
    rng = random.Random(seed)
    while True:
        order = list(range(n))
        rng.shuffle(order)
        for i in range(0, n, batch_size):
            yield order[i : i + batch_size]


@torch.no_grad()
def mean_loss(
    model: torch.nn.Module,
    features: Sequence[Feature],
    *,
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Mean per-token answer loss (the validation loss)."""
    model.eval()
    total, tokens = 0.0, 0
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for i in range(0, len(features), batch_size):
        batch = {k: v.to(device) for k, v in collate(features[i : i + batch_size], pad_id).items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out: Any = model(**batch)
        n = int((batch["labels"][:, 1:] != IGNORE).sum())
        total += float(out.loss) * n
        tokens += n
    return total / max(1, tokens)
