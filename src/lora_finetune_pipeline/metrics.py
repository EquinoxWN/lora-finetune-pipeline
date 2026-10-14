"""Classification metrics where an unparseable answer counts as wrong."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

INVALID = "<invalid>"


@dataclass(frozen=True)
class Scores:
    """Accuracy, macro-F1 over the gold labels and the share of answers that named no allowed intent."""

    n: int
    accuracy: float
    macro_f1: float
    invalid_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready form."""
        return asdict(self)


def score(predictions: Sequence[str | None], gold: Sequence[str]) -> Scores:
    """Score predictions against gold labels."""
    if len(predictions) != len(gold):
        raise ValueError("predictions and gold differ in length")
    if not gold:
        raise ValueError("nothing to score")
    preds = [p if p is not None else INVALID for p in predictions]
    correct = sum(p == g for p, g in zip(preds, gold, strict=True))
    return Scores(
        n=len(gold),
        accuracy=correct / len(gold),
        macro_f1=macro_f1(preds, gold),
        invalid_rate=sum(p == INVALID for p in preds) / len(gold),
    )


def macro_f1(preds: Sequence[str], gold: Sequence[str]) -> float:
    """Unweighted mean F1 over the labels that occur in gold."""
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for p, g in zip(preds, gold, strict=True):
        if p == g:
            tp[g] += 1
        else:
            fn[g] += 1
            fp[p] += 1
    f1s = []
    for label in sorted(set(gold)):
        denom = 2 * tp[label] + fp[label] + fn[label]
        f1s.append(2 * tp[label] / denom if denom else 0.0)
    return sum(f1s) / len(f1s)


def top_confusions(
    predictions: Sequence[str | None], gold: Sequence[str], k: int = 10
) -> list[tuple[str, str, int]]:
    """The k most frequent (gold, predicted) mistakes."""
    pairs: Counter[tuple[str, str]] = Counter(
        (g, p if p is not None else INVALID)
        for p, g in zip(predictions, gold, strict=True)
        if p != g
    )
    return [(g, p, n) for (g, p), n in pairs.most_common(k)]
