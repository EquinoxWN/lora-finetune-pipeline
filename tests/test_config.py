"""Pins and run configs are validated before anything runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lora_finetune_pipeline.config import (
    DATASET_FILES,
    MODEL,
    ModelPin,
    RemoteFile,
    load_run_config,
    parse_run_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_shipped_pins_are_full_hashes_over_https() -> None:
    assert len(MODEL.revision) == 40
    assert MODEL.licence == "Apache-2.0"
    assert all(
        f.url.startswith("https://") and "57ec275d8078af65b7731c2a98be812d844a6d6b" in f.url
        for f in DATASET_FILES.values()
    )


@pytest.mark.parametrize("name", ["m1.json", "lite.json"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_run_config(CONFIGS / name)
    assert cfg.name == name.removesuffix(".json")
    assert cfg.train.lora_r == 16


def test_lite_config_is_smaller_than_m1() -> None:
    m1, lite = load_run_config(CONFIGS / "m1.json"), load_run_config(CONFIGS / "lite.json")
    assert lite.eval.limit_per_class > 0 == m1.eval.limit_per_class
    assert lite.train.max_steps > 0


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"name": "x", "surprise": 1}, "unknown keys"),
        ({"name": "x", "train": {"lr": 1}}, "unknown keys in train"),
        ({"name": "../evil"}, "bad run name"),
        ({"name": "x", "train": {"lora_r": 0}}, "lora_r"),
        ({"name": "x", "train": {"learning_rate": 5}}, "learning_rate"),
        ({"name": "x", "train": {"target_modules": []}}, "target_modules"),
        ({"name": "x", "eval": {"max_new_tokens": 1000}}, "max_new_tokens"),
        ({"name": "x", "vram_fraction": 2.0}, "vram_fraction"),
        ([1, 2], "JSON object"),
    ],
)
def test_bad_configs_are_refused(raw: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_run_config(raw)


def test_pins_refuse_http_short_revisions_and_bad_hashes() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        RemoteFile("http://example.com/a.csv", "0" * 64, 10)
    with pytest.raises(ValueError, match="sha256"):
        RemoteFile("https://example.com/a.csv", "abc", 10)
    with pytest.raises(ValueError, match="40-character"):
        ModelPin("Qwen/x", "main", "model.safetensors", "0" * 64, "Apache-2.0")
    with pytest.raises(ValueError, match="repo id"):
        ModelPin("../../x", "0" * 40, "model.safetensors", "0" * 64, "Apache-2.0")


def test_oversized_config_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "big.json"
    path.write_text(json.dumps({"name": "x", "pad": "y" * 70_000}), encoding="utf-8")
    with pytest.raises(ValueError, match="larger than"):
        load_run_config(path)
