"""Plain-text tables from a recorded results file."""

from __future__ import annotations

from typing import Any


def _fmt(value: Any, digits: int = 1) -> str:
    """A number with fixed digits, or n/a."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def table(results: dict[str, Any]) -> str:
    """Quality, latency, tokens, VRAM and energy per mode, then training and break-even lines."""
    rows = [
        "| Mode | Accuracy | Macro-F1 | Invalid | Prompt tokens | ms/query | Peak VRAM MiB | Energy Wh |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode, r in results["results"].items():
        phase = results["phases"][mode]
        rows.append(
            f"| {mode} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['invalid_rate']:.4f} | "
            f"{_fmt(r['prompt_tokens_mean'])} | {_fmt(r['ms_per_query'])} | "
            f"{_fmt(phase['peak_reserved_mib'], 0)} | {_fmt(phase['energy_wh'], 2)} |"
        )
    t = results["train"]
    tp = results["phases"]["train"]
    lines = [
        f"Eval set: {results['eval_set']['n']:,} held-out messages, {results['eval_set']['labels']} intents",
        *rows,
        "",
        f"QLoRA: {t['steps']} steps, {t['examples_seen']:,} examples in {tp['seconds']:.1f} s, "
        f"peak VRAM {_fmt(tp['peak_reserved_mib'], 0)} MiB, energy {_fmt(tp['energy_wh'], 2)} Wh",
        f"  loss {t['loss_curve'][0][1]:.3f} -> {t['loss_curve'][-1][1]:.3f}; "
        f"validation loss {t['val_loss_before']:.3f} -> {t['val_loss_after']:.3f}; "
        f"{t['trainable_params']:,} trainable params, adapter {t['adapter_mib']:.1f} MiB",
    ]
    lines.extend(_break_even(c) for c in results["cost"])
    return "\n".join(lines)


def _break_even(c: dict[str, Any]) -> str:
    """One break-even line for a prompting mode."""
    line = (
        f"Break-even vs {c['against']}: {_fmt(c['break_even_queries_gpu_time'])} queries (GPU time)"
    )
    if "break_even_queries_energy" in c:
        line += f", {_fmt(c['break_even_queries_energy'])} queries (energy)"
    return line
