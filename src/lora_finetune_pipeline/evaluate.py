"""Batched greedy generation on the held-out split and its scores."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from .data import Example
from .metrics import Scores, score, top_confusions
from .prompts import Mode, messages_for, parse_label
from .train import render


@dataclass
class EvalResult:
    """Scores, cost and the raw answers for one prompting mode."""

    mode: str
    scores: Scores
    seconds: float
    prompt_tokens_mean: float
    prompt_tokens_total: int
    generated_tokens_total: int
    confusions: list[tuple[str, str, int]] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """JSON-ready form without the raw answers."""
        return {
            "mode": self.mode,
            **self.scores.to_dict(),
            "seconds": round(self.seconds, 2),
            "ms_per_query": round(1000 * self.seconds / self.scores.n, 2),
            "prompt_tokens_mean": round(self.prompt_tokens_mean, 1),
            "prompt_tokens_total": self.prompt_tokens_total,
            "generated_tokens_total": self.generated_tokens_total,
            "top_confusions": self.confusions,
        }


@torch.no_grad()
def generate(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[str], list[int], int]:
    """Greedy answers in input order, the prompt lengths, and the generated token count."""
    tokenizer.padding_side = "left"
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    outputs: list[str] = [""] * len(prompts)
    generated = 0
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    model.eval()
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        enc = tokenizer(
            [prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        new = out[:, enc["input_ids"].shape[1] :]
        for row, i in enumerate(idx):
            ids = [t for t in new[row].tolist() if t not in (pad_id, eos_id)]
            generated += len(ids)
            outputs[i] = str(tokenizer.decode(ids, skip_special_tokens=True))
        if progress is not None:
            progress(min(start + batch_size, len(order)), len(order))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return outputs, lengths, generated


def evaluate(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    mode: Mode,
    examples: Sequence[Example],
    labels: Sequence[str],
    shots: Sequence[Example],
    *,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    progress: Callable[[int, int], None] | None = None,
) -> EvalResult:
    """Prompt every held-out example in one mode and score the parsed answers."""
    prompts = [render(tokenizer, messages_for(mode, e.text, labels, shots)) for e in examples]
    start = time.perf_counter()
    outputs, lengths, generated = generate(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
        progress=progress,
    )
    seconds = time.perf_counter() - start
    preds = [parse_label(o, labels) for o in outputs]
    gold = [e.label for e in examples]
    return EvalResult(
        mode=mode,
        scores=score(preds, gold),
        seconds=seconds,
        prompt_tokens_mean=sum(lengths) / len(lengths),
        prompt_tokens_total=sum(lengths),
        generated_tokens_total=generated,
        confusions=top_confusions(preds, gold),
        outputs=outputs,
    )
