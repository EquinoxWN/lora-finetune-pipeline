"""Prompt construction, the strict answer parser, metrics and the cost arithmetic."""

from __future__ import annotations

import pytest

from lora_finetune_pipeline.cost import break_even_queries, compare, per_query
from lora_finetune_pipeline.data import Example
from lora_finetune_pipeline.metrics import score, top_confusions
from lora_finetune_pipeline.prompts import (
    TASK,
    few_shot,
    fine_tuned,
    parse_label,
    pick_shots,
    zero_shot,
)

from .conftest import synthetic

LABELS = ["card_arrival", "lost_pin", "top_up"]


def test_zero_and_few_shot_list_every_intent_and_fine_tuned_does_not() -> None:
    for messages in (zero_shot("hi", LABELS), few_shot("hi", LABELS, [])):
        assert all(label in messages[0]["content"] for label in LABELS)
    short = fine_tuned("hi")
    assert short == [{"role": "system", "content": TASK}, {"role": "user", "content": "hi"}]


def test_few_shot_adds_examples_as_prior_turns_then_the_query() -> None:
    shots = [Example("where is my card", "card_arrival"), Example("forgot pin", "lost_pin")]
    messages = few_shot("query text", LABELS, shots)
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[2]["content"] == "card_arrival"
    assert messages[-1]["content"] == "query text"


def test_pick_shots_takes_k_per_class_with_a_seed() -> None:
    train = synthetic(10, 0)
    shots = pick_shots(train, 2, seed=5)
    assert sorted(e.label for e in shots) == sorted(LABELS * 2)
    assert shots == pick_shots(train, 2, seed=5)
    assert pick_shots(train, 0, seed=5) == []


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("card_arrival", "card_arrival"),
        ("  Card Arrival.\nbecause...", "card_arrival"),
        ("Intent: lost-pin", "lost_pin"),
        ("`top_up`", "top_up"),
        ("card_arrival_late", None),
        ("I think it is card_arrival", None),
        ("", None),
    ],
)
def test_parse_label_accepts_only_one_allowed_intent(output: str, expected: str | None) -> None:
    assert parse_label(output, LABELS) == expected


def test_scores_count_invalid_answers_as_wrong() -> None:
    gold = ["a", "a", "b", "b"]
    s = score(["a", None, "b", "a"], gold)
    assert s.accuracy == 0.5
    assert s.invalid_rate == 0.25
    # F1(a) = 2*1/(2+1+1) = 0.5, F1(b) = 2*1/(2+0+1) = 2/3
    assert s.macro_f1 == pytest.approx((0.5 + 2 / 3) / 2)
    assert top_confusions(["a", None, "b", "a"], gold) == [("a", "<invalid>", 1), ("b", "a", 1)]
    with pytest.raises(ValueError, match="length"):
        score(["a"], gold)


def test_break_even_uses_only_measured_savings() -> None:
    assert break_even_queries(1000.0, 3.0, 1.0) == 500
    assert break_even_queries(1000.0, 1.0, 1.0) is None
    assert per_query(2.0, 0.5, 4) == {"gpu_ms_per_query": 500.0, "mwh_per_query": 125.0}
    prompting = {
        "mode": "few-shot",
        "seconds": 30.0,
        "energy_wh": 0.3,
        "n": 100,
        "prompt_tokens_mean": 900.0,
    }
    tuned = {
        "mode": "fine-tuned",
        "seconds": 10.0,
        "energy_wh": 0.1,
        "n": 100,
        "prompt_tokens_mean": 40.0,
    }
    c = compare(60.0, 0.6, prompting, tuned)
    assert c["break_even_queries_gpu_time"] == 300  # 60 s / (300 ms - 100 ms)
    assert c["break_even_queries_energy"] == 300
