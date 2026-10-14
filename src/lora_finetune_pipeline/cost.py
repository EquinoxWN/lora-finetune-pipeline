"""Cost comparison derived only from measured time and energy."""

from __future__ import annotations

import math
from typing import Any


def per_query(seconds: float, energy_wh: float | None, n: int) -> dict[str, float | None]:
    """GPU milliseconds and milliwatt-hours per query."""
    if n <= 0:
        raise ValueError("n must be positive")
    return {
        "gpu_ms_per_query": 1000 * seconds / n,
        "mwh_per_query": None if energy_wh is None else 1000 * energy_wh / n,
    }


def break_even_queries(
    one_off: float, per_query_prompting: float, per_query_tuned: float
) -> int | None:
    """Queries after which a one-off training cost is repaid by cheaper queries (None if never)."""
    saving = per_query_prompting - per_query_tuned
    if one_off < 0:
        raise ValueError("one_off must be >= 0")
    if saving <= 0:
        return None
    return math.ceil(one_off / saving)


def compare(
    train_seconds: float,
    train_wh: float | None,
    prompting: dict[str, Any],
    tuned: dict[str, Any],
) -> dict[str, Any]:
    """Break-even of fine-tuning against one prompting mode, in GPU time and in energy."""
    p = per_query(prompting["seconds"], prompting.get("energy_wh"), prompting["n"])
    t = per_query(tuned["seconds"], tuned.get("energy_wh"), tuned["n"])
    out: dict[str, Any] = {
        "against": prompting["mode"],
        "gpu_ms_per_query": {
            "prompting": round(p["gpu_ms_per_query"] or 0, 2),
            "fine_tuned": round(t["gpu_ms_per_query"] or 0, 2),
        },
        "prompt_tokens_mean": {
            "prompting": prompting["prompt_tokens_mean"],
            "fine_tuned": tuned["prompt_tokens_mean"],
        },
        "break_even_queries_gpu_time": break_even_queries(
            train_seconds * 1000, p["gpu_ms_per_query"] or 0, t["gpu_ms_per_query"] or 0
        ),
    }
    if train_wh is not None and p["mwh_per_query"] is not None and t["mwh_per_query"] is not None:
        out["mwh_per_query"] = {
            "prompting": round(p["mwh_per_query"], 3),
            "fine_tuned": round(t["mwh_per_query"], 3),
        }
        out["break_even_queries_energy"] = break_even_queries(
            train_wh * 1000, p["mwh_per_query"], t["mwh_per_query"]
        )
    return out
