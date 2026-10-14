"""End to end on CPU: prompted tiny model vs its LoRA adapter, results recorded, model files purged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from lora_finetune_pipeline import cli
from lora_finetune_pipeline.config import EvalConfig, RunConfig, TrainConfig
from lora_finetune_pipeline.data import curate
from lora_finetune_pipeline.evaluate import generate
from lora_finetune_pipeline.gpu import Meter
from lora_finetune_pipeline.pipeline import Loaded, purge, run
from lora_finetune_pipeline.report import table

from .conftest import synthetic, tiny_model, tiny_tokenizer

CPU = torch.device("cpu")


class TinyBackend:
    """A random two-layer Qwen2 with the word-level tokenizer; 'downloads' a fake weights file."""

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self.device = CPU

    def load(self, log: object) -> Loaded:
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.cache / "model.safetensors").write_bytes(b"0" * 1024)
        tok = tiny_tokenizer()
        return Loaded(tiny_model(len(tok)), tok, CPU, False, {"footprint_mib": 0.1})

    def model_paths(self) -> list[Path]:
        return [self.cache]


CFG = RunConfig(
    name="tiny",
    seed=7,
    val_per_class=3,
    train=TrainConfig(
        lora_r=8,
        lora_alpha=16,
        learning_rate=3e-3,
        epochs=10,
        batch_size=4,
        max_seq_len=64,
        gradient_checkpointing=False,
        lora_dropout=0.0,
    ),
    eval=EvalConfig(batch_size=4, max_new_tokens=3, few_shot_per_class=1),
)


def test_generate_returns_answers_in_input_order(tokenizer: PreTrainedTokenizerFast) -> None:
    model = tiny_model(len(tokenizer))
    prompts = [
        "<|im_start|>user my card<|im_end|> <|im_start|>assistant",
        "<|im_start|>user pin<|im_end|> <|im_start|>assistant",
    ]
    one_by_one = [
        generate(model, tokenizer, [p], batch_size=1, max_new_tokens=3, device=CPU)[0][0]
        for p in prompts
    ]
    batched, lengths, _ = generate(
        model, tokenizer, prompts, batch_size=2, max_new_tokens=3, device=CPU
    )
    assert batched == one_by_one
    assert lengths[0] > lengths[1]


def test_end_to_end_adapter_beats_the_prompted_random_model(repo: Path) -> None:
    splits = curate(synthetic(40, 1), synthetic(6, 2), CFG.val_per_class, CFG.seed).as_dict()
    backend = TinyBackend(repo / ".tmp" / "hf")
    results = run(CFG, splits, backend, repo / ".tmp" / "runs" / "tiny", log=lambda _: None)
    r = results["results"]
    assert set(r) == {"zero-shot", "few-shot", "fine-tuned"}
    assert r["fine-tuned"]["accuracy"] >= 0.9
    assert r["fine-tuned"]["accuracy"] > max(r["zero-shot"]["accuracy"], r["few-shot"]["accuracy"])
    assert (
        r["few-shot"]["prompt_tokens_mean"]
        > r["zero-shot"]["prompt_tokens_mean"]
        > r["fine-tuned"]["prompt_tokens_mean"]
    )
    assert results["train"]["val_loss_after"] < results["train"]["val_loss_before"]
    assert results["eval_set"]["n"] == len(splits["test"])
    assert (repo / ".tmp" / "runs" / "tiny" / "adapter" / "adapter_config.json").exists()
    assert "QLoRA:" in table(json.loads(json.dumps(results)))
    removed = purge([*backend.model_paths(), repo / ".tmp" / "runs" / "tiny" / "adapter"], repo)
    assert any(p["path"].endswith("model.safetensors") for p in removed)
    assert any(p["path"].endswith("adapter_config.json") for p in removed)
    assert not (repo / ".tmp" / "hf").exists()


def test_reused_prompting_results_are_not_rerun_and_partials_are_saved(repo: Path) -> None:
    splits = curate(synthetic(40, 1), synthetic(6, 2), CFG.val_per_class, CFG.seed).as_dict()
    n = len(splits["test"])
    recorded = {
        "n": n,
        "accuracy": 0.25,
        "macro_f1": 0.2,
        "invalid_rate": 0.4,
        "seconds": 3.0,
        "source": "log",
    }
    reuse = {"results": {"zero-shot": recorded, "few-shot": {**recorded, "accuracy": 0.3}}}
    run_dir = repo / ".tmp" / "runs" / "tiny"
    results = run(
        CFG, splits, TinyBackend(repo / ".tmp" / "hf"), run_dir, log=lambda _: None, reuse=reuse
    )
    zs, fs = results["results"]["zero-shot"], results["results"]["few-shot"]
    assert (zs["accuracy"], fs["accuracy"]) == (0.25, 0.3)
    assert zs["reused_from"] == "log"
    assert zs["ms_per_query"] == round(3000 / n, 2)
    assert (
        fs["prompt_tokens_mean"]
        > zs["prompt_tokens_mean"]
        > results["results"]["fine-tuned"]["prompt_tokens_mean"]
    )
    assert results["phases"]["zero-shot"]["peak_allocated_mib"] is None
    partial = json.loads((run_dir / "partial.json").read_text(encoding="utf-8"))
    assert set(partial["results"]) == {"zero-shot", "few-shot", "fine-tuned"}
    assert "reused_from" not in partial["results"]["fine-tuned"]
    with pytest.raises(ValueError, match="n=1"):
        run(
            CFG,
            splits,
            TinyBackend(repo / ".tmp" / "hf"),
            run_dir,
            log=lambda _: None,
            reuse={"results": {"zero-shot": {**recorded, "n": 1}}},
        )


def test_purge_refuses_paths_outside_tmp(repo: Path) -> None:
    (repo / "src").mkdir()
    with pytest.raises(ValueError, match="outside"):
        purge([repo / "src"], repo)
    with pytest.raises(ValueError, match="whole"):
        purge([repo / ".tmp"], repo)
    assert purge([repo / ".tmp" / "missing"], repo) == []
    (repo / ".tmp" / "checkpoint.pt").write_bytes(b"12")
    assert purge([repo / ".tmp" / "checkpoint.pt"], repo) == [
        {"path": ".tmp/checkpoint.pt", "bytes": 2}
    ]


def test_meter_on_cpu_records_time_only() -> None:
    with Meter("x", CPU) as m:
        sum(range(1000))
    assert m.result.seconds >= 0
    assert m.result.peak_allocated_mib is None
    assert m.result.energy_wh is None


def test_cli_pins_show_and_errors(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["pins"]) == 0
    assert "Qwen/Qwen2.5-0.5B-Instruct" in capsys.readouterr().out
    bad = repo / "bad.json"
    bad.write_text('{"name": "x", "train": {"lora_r": 0}}', encoding="utf-8")
    assert cli.main(["prepare-data", "--config", str(bad)]) == cli.EXIT_BAD_INPUT
    assert "lora_r" in capsys.readouterr().err
    (repo / ".tmp" / "hf" / "hub").mkdir(parents=True, exist_ok=True)
    (repo / ".tmp" / "hf" / "hub" / "w.bin").write_bytes(b"1")
    assert cli.main(["purge"]) == 0
    assert "1 files deleted" in capsys.readouterr().out


def test_cli_run_stops_at_preflight_without_enough_memory(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from lora_finetune_pipeline import gpu

    def short(*_: object, **__: object) -> dict[str, object]:
        raise gpu.PreflightError("only 100 MiB RAM free, need 1500")

    monkeypatch.setattr(gpu, "preflight", short)
    cfg = repo / "c.json"
    cfg.write_text('{"name": "x"}', encoding="utf-8")
    monkeypatch.setattr(cli, "_prepare", lambda *_: {})
    assert (
        cli.main(["run", "--config", str(cfg), "--out", str(repo / "r.json")]) == cli.EXIT_PREFLIGHT
    )
    assert "preflight failed" in capsys.readouterr().err
