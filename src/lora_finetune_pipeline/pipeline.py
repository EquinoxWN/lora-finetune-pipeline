"""The full comparison: base model prompted zero-shot and few-shot vs a QLoRA adapter, on one held-out set."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import torch
from transformers import PreTrainedTokenizerBase

from . import config as cfgmod
from .config import RunConfig
from .cost import compare
from .data import Example, labels_of, stratified_subset
from .env import ensure_inside, tmp_dir
from .evaluate import EvalResult, evaluate
from .gpu import Measurement, Meter, gpu_info
from .prompts import Mode, messages_for, pick_shots
from .train import (
    attach_lora,
    build_features,
    for_inference,
    mean_loss,
    render,
    seed_everything,
    train,
)

Log = Callable[[str], None]


@dataclass
class Loaded:
    """A model ready for evaluation and training."""

    model: Any
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    quantized: bool
    info: dict[str, Any]


class Backend(Protocol):
    """Loads the base model (the real 4-bit Qwen, or a tiny model in tests)."""

    device: torch.device

    def load(self, log: Log) -> Loaded:
        """Load and return the base model."""
        ...

    def model_paths(self) -> list[Path]:
        """Folders holding downloaded model files (deleted after the run)."""
        ...


class QloraBackend:
    """The pinned Hugging Face model in 4-bit NF4 on GPU 0."""

    def __init__(self, pin: cfgmod.ModelPin, root: Path, vram_fraction: float) -> None:
        """Remember where to cache the model and how much VRAM to allow."""
        self.pin = pin
        self.cache = tmp_dir(root) / "hf" / "hub"
        self.vram_fraction = vram_fraction
        self.device = torch.device("cuda", 0)

    def load(self, log: Log) -> Loaded:
        """Download (verified) and load in 4-bit."""
        from .model import fetch, load_4bit, load_tokenizer

        torch.cuda.set_per_process_memory_fraction(self.vram_fraction, 0)
        path = fetch(self.pin, self.cache)
        log(f"weights verified: {self.pin.repo_id}@{self.pin.revision[:12]}")
        tokenizer = load_tokenizer(path)
        model = load_4bit(path)
        return Loaded(
            model,
            tokenizer,
            self.device,
            True,
            {"footprint_mib": round(model.get_memory_footprint() / 2**20, 1)},
        )

    def model_paths(self) -> list[Path]:
        """The Hugging Face cache folder."""
        return [self.cache.parent]


def versions() -> dict[str, str]:
    """Versions of Python, the libraries and the CUDA stack used for the run."""
    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
    }
    for dist in (
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "huggingface-hub",
        "safetensors",
        "tokenizers",
    ):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = "not installed"
    try:
        import pynvml

        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        out["nvidia_driver"] = driver.decode() if isinstance(driver, bytes) else str(driver)
    except Exception:  # no NVIDIA driver (CI, tests)
        out["nvidia_driver"] = "none"
    return out


def run(
    cfg: RunConfig,
    splits: dict[str, list[Example]],
    backend: Backend,
    run_dir: Path,
    *,
    log: Log = print,
    modes: Sequence[Mode] = ("zero-shot", "few-shot"),
    reuse: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the prompted base model, train the adapter, evaluate it, and return every number.

    Prompting modes found in ``reuse`` (results recorded by an earlier run on the same split and
    config) are not evaluated again; a partial results file is written after every phase.
    """
    seed_everything(cfg.seed)
    labels = labels_of(splits["train"])
    test = stratified_subset(splits["test"], cfg.eval.limit_per_class, cfg.seed)
    shots = pick_shots(splits["train"], cfg.eval.few_shot_per_class, cfg.seed)
    phases: dict[str, Measurement] = {}
    results: dict[str, EvalResult] = {}
    summaries: dict[str, dict[str, Any]] = {}
    reused = dict((reuse or {}).get("results", {}))

    with Meter("load", backend.device) as m:
        loaded = backend.load(log)
    phases["load"] = m.result
    device, tok = loaded.device, loaded.tokenizer
    log(
        f"model loaded in {m.result.seconds:.1f} s; eval set {len(test)} examples, {len(labels)} intents"
    )

    def run_eval(mode: Mode, model: Any) -> None:
        """Score one prompting mode and record its phase."""
        with Meter(mode, device) as meter:
            results[mode] = evaluate(
                model,
                tok,
                mode,
                test,
                labels,
                shots,
                batch_size=cfg.eval.batch_size,
                max_new_tokens=cfg.eval.max_new_tokens,
                device=device,
                progress=_progress(log, mode),
            )
        phases[mode] = meter.result
        summaries[mode] = {**results[mode].summary(), "energy_wh": meter.result.energy_wh}
        _save_partial(run_dir, cfg, summaries, phases)
        s = results[mode].scores
        log(
            f"{mode}: accuracy {s.accuracy:.4f}, macro-F1 {s.macro_f1:.4f}, invalid {s.invalid_rate:.4f}, {meter.result.seconds:.1f} s"
        )

    for mode in modes:
        if mode == "few-shot" and cfg.eval.few_shot_per_class == 0:
            continue
        if mode in reused:
            summaries[mode] = _reused(reused[mode], mode, tok, test, labels, shots)
            phases[mode] = Measurement(mode, seconds=float(reused[mode]["seconds"]))
            log(f"{mode}: reused recorded result (accuracy {summaries[mode]['accuracy']:.4f})")
        else:
            run_eval(mode, loaded.model)

    train_features, truncated = build_features(tok, splits["train"], cfg.train.max_seq_len)
    val_features, _ = build_features(tok, splits["val"], cfg.train.max_seq_len)
    peft_model = attach_lora(loaded.model, cfg.train, quantized=loaded.quantized)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else int(tok.eos_token_id or 0)
    val_before = mean_loss(
        peft_model, val_features, pad_id=pad_id, batch_size=cfg.eval.batch_size, device=device
    )
    with Meter("train", device) as meter:
        trained = train(
            peft_model,
            train_features,
            cfg.train,
            pad_id=pad_id,
            seed=cfg.seed,
            device=device,
            log=log,
            checkpoint=run_dir / "checkpoint.pt",
        )
    phases["train"] = meter.result
    trained.truncated = truncated
    for_inference(peft_model)
    val_after = mean_loss(
        peft_model, val_features, pad_id=pad_id, batch_size=cfg.eval.batch_size, device=device
    )
    log(f"validation loss {val_before:.4f} -> {val_after:.4f}")
    run_eval("fine-tuned", peft_model)

    adapter_dir = run_dir / "adapter"
    peft_model.save_pretrained(str(adapter_dir))
    adapter_mib = sum(p.stat().st_size for p in adapter_dir.rglob("*") if p.is_file()) / 2**20

    costs = [
        compare(
            phases["train"].seconds,
            phases["train"].energy_wh,
            summaries[mode],
            summaries["fine-tuned"],
        )
        for mode in ("zero-shot", "few-shot")
        if mode in summaries
    ]
    _write_predictions(run_dir, test, results)
    return {
        "run": cfg.name,
        "recorded": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "pins": {
            "model": {**vars(cfgmod.MODEL)},
            "dataset": {k: vars(v) for k, v in cfgmod.DATASET_FILES.items()},
            "dataset_licence": cfgmod.DATASET_LICENCE,
        },
        "versions": versions(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpus": os.cpu_count(),
            "gpu": gpu_info(),
        },
        "eval_set": {
            "n": len(test),
            "labels": len(labels),
            "limit_per_class": cfg.eval.limit_per_class,
            "few_shot_examples": len(shots),
        },
        "load": {**phases["load"].to_dict(), **loaded.info},
        "phases": {k: v.to_dict() for k, v in phases.items()},
        "results": summaries,
        "train": {
            "steps": trained.steps,
            "examples_seen": trained.examples_seen,
            "tokens_seen": trained.tokens_seen,
            "seconds": round(trained.seconds, 2),
            "loss_curve": trained.loss_curve,
            "val_loss_before": round(val_before, 4),
            "val_loss_after": round(val_after, 4),
            "truncated_examples": trained.truncated,
            "trainable_params": trained.trainable_params,
            "total_params": trained.total_params,
            "adapter_mib": round(adapter_mib, 2),
            "resumed_after_step": trained.resumed_at,
        },
        "cost": costs,
    }


def _reused(
    recorded: dict[str, Any],
    mode: Mode,
    tok: PreTrainedTokenizerBase,
    test: Sequence[Example],
    labels: Sequence[str],
    shots: Sequence[Example],
) -> dict[str, Any]:
    """A recorded prompting result, checked against this eval set; prompt tokens recomputed exactly."""
    n = int(recorded["n"])
    if n != len(test):
        raise ValueError(f"recorded {mode} result has n={n}, this eval set has {len(test)}")
    lengths = [
        len(
            tok(render(tok, messages_for(mode, e.text, labels, shots)), add_special_tokens=False)[
                "input_ids"
            ]
        )
        for e in test
    ]
    seconds = float(recorded["seconds"])
    return {
        "mode": mode,
        "n": n,
        "accuracy": float(recorded["accuracy"]),
        "macro_f1": float(recorded["macro_f1"]),
        "invalid_rate": float(recorded["invalid_rate"]),
        "seconds": round(seconds, 2),
        "ms_per_query": round(1000 * seconds / n, 2),
        "prompt_tokens_mean": round(sum(lengths) / n, 1),
        "prompt_tokens_total": sum(lengths),
        "generated_tokens_total": recorded.get("generated_tokens_total"),
        "top_confusions": recorded.get("top_confusions", []),
        "energy_wh": recorded.get("energy_wh"),
        "reused_from": str(recorded.get("source", "earlier run")),
    }


def _save_partial(
    run_dir: Path, cfg: RunConfig, summaries: dict[str, Any], phases: dict[str, Measurement]
) -> None:
    """Results so far, so a crash later in the run does not lose finished phases."""
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "run": cfg.name,
        "results": {
            m: {**s, "source": f"{cfg.name} partial results"} for m, s in summaries.items()
        },
        "phases": {k: v.to_dict() for k, v in phases.items()},
    }
    (run_dir / "partial.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _progress(log: Log, mode: str) -> Callable[[int, int], None]:
    """Log progress about every 10%."""
    marks: set[int] = set()

    def report(done: int, total: int) -> None:
        """Print when another tenth is done."""
        tenth = 10 * done // max(1, total)
        if tenth not in marks:
            marks.add(tenth)
            log(f"  {mode}: {done}/{total}")

    return report


def _write_predictions(
    run_dir: Path, test: Sequence[Example], results: dict[str, EvalResult]
) -> None:
    """Raw answers per example for error analysis (git-ignored, under .tmp)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as out:
        for i, e in enumerate(test):
            row = {"text": e.text, "gold": e.label, **{m: r.outputs[i] for m, r in results.items()}}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def purge(paths: Sequence[Path], root: Path) -> list[dict[str, Any]]:
    """Delete model, adapter or data files and folders inside .tmp; list every file removed."""
    base = tmp_dir(root).resolve()
    removed: list[dict[str, Any]] = []
    for path in paths:
        target = ensure_inside(path, base)
        if target == base:
            raise ValueError("refusing to delete the whole .tmp folder")
        if not target.exists():
            continue
        if target.is_file():
            size = target.stat().st_size
            removed.append({"path": target.relative_to(root.resolve()).as_posix(), "bytes": size})
            target.unlink()
            continue
        removed.extend(
            {"path": f.relative_to(root.resolve()).as_posix(), "bytes": f.lstat().st_size}
            for f in sorted(target.rglob("*"))
            if f.is_file() or f.is_symlink()
        )
        shutil.rmtree(target)
    return removed
