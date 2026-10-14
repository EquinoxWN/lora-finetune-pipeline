"""Chat prompts for zero-shot, few-shot and fine-tuned classification, and a strict output parser."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from .data import Example

Mode = Literal["zero-shot", "few-shot", "fine-tuned"]
Message = dict[str, str]

TASK = "Classify the bank customer's message into exactly one intent. Answer with the intent name only."


def label_list(labels: Sequence[str]) -> str:
    """The allowed intents, comma-separated."""
    return ", ".join(labels)


def zero_shot(text: str, labels: Sequence[str]) -> list[Message]:
    """Instruction plus the full intent list, no examples."""
    return [
        {"role": "system", "content": f"{TASK}\nAllowed intents: {label_list(labels)}"},
        {"role": "user", "content": text},
    ]


def few_shot(text: str, labels: Sequence[str], shots: Sequence[Example]) -> list[Message]:
    """Instruction, intent list, then worked examples as earlier chat turns."""
    messages = [{"role": "system", "content": f"{TASK}\nAllowed intents: {label_list(labels)}"}]
    for shot in shots:
        messages.append({"role": "user", "content": shot.text})
        messages.append({"role": "assistant", "content": shot.label})
    messages.append({"role": "user", "content": text})
    return messages


def fine_tuned(text: str) -> list[Message]:
    """Short prompt for the adapter: it has learned the intents, so no list is sent."""
    return [{"role": "system", "content": TASK}, {"role": "user", "content": text}]


def messages_for(
    mode: Mode, text: str, labels: Sequence[str], shots: Sequence[Example]
) -> list[Message]:
    """Dispatch to the prompt builder for a mode."""
    if mode == "zero-shot":
        return zero_shot(text, labels)
    if mode == "few-shot":
        return few_shot(text, labels, shots)
    return fine_tuned(text)


def pick_shots(train: Sequence[Example], per_class: int, seed: int) -> list[Example]:
    """per_class training examples of every label, in a seeded random order."""
    by_label: dict[str, list[Example]] = defaultdict(list)
    for e in train:
        by_label[e.label].append(e)
    rng = random.Random(seed)
    shots: list[Example] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        shots.extend(group[:per_class])
    rng.shuffle(shots)
    return shots


_STRIP = re.compile(r"^[\s\"'`*.:\-]+|[\s\"'`*.:\-]+$")


def parse_label(output: str, labels: Sequence[str]) -> str | None:
    """The intent named by a model's answer, or None if it is not exactly one allowed intent."""
    lines = output.strip().splitlines()
    if not lines:
        return None
    answer = _STRIP.sub("", lines[0])
    answer = re.sub(r"^(intent|answer)\s*:\s*", "", answer, flags=re.IGNORECASE)
    answer = _STRIP.sub("", answer).lower().replace(" ", "_").replace("-", "_")
    return answer if answer in set(labels) else None
