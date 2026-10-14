"""Verified download, parsing, de-duplication and a split with no leakage."""

from __future__ import annotations

import hashlib
import io
import random
from pathlib import Path
from typing import IO

import pytest

from lora_finetune_pipeline.config import RemoteFile
from lora_finetune_pipeline.data import (
    DownloadError,
    Example,
    bag_key,
    canonical_label,
    csv_text,
    curate,
    dedupe,
    download,
    fingerprint,
    leakage,
    load_jsonl,
    normalise,
    parse_csv,
    prepare,
    save_jsonl,
    stratified_subset,
)

from .conftest import synthetic

PAYLOAD = b"text,category\nwhere is my card,card_arrival\n"


class FakeResponse(io.BytesIO):
    """In-memory HTTP response with a final URL."""

    def __init__(self, data: bytes, url: str) -> None:
        super().__init__(data)
        self.url = url


def opener_for(data: bytes, final_url: str | None = None) -> object:
    """An opener returning fixed bytes."""

    def open_(url: str) -> IO[bytes]:
        return FakeResponse(data, final_url or url)

    return open_


def pin(data: bytes) -> RemoteFile:
    """A pin matching the given bytes."""
    return RemoteFile("https://example.org/train.csv", hashlib.sha256(data).hexdigest(), len(data))


def test_download_keeps_a_matching_file(tmp_path: Path) -> None:
    dest = download(pin(PAYLOAD), tmp_path / "train.csv", opener_for(PAYLOAD))  # type: ignore[arg-type]
    assert dest.read_bytes() == PAYLOAD


@pytest.mark.parametrize(
    ("served", "final_url", "message"),
    [
        (PAYLOAD.replace(b"card", b"cart"), None, "does not match"),
        (PAYLOAD + b"more,rows\n", None, "larger than"),
        (PAYLOAD, "http://example.org/train.csv", "non-HTTPS"),
    ],
)
def test_download_refuses_tampered_oversized_or_downgraded(
    tmp_path: Path, served: bytes, final_url: str | None, message: str
) -> None:
    dest = tmp_path / "train.csv"
    with pytest.raises(DownloadError, match=message):
        download(pin(PAYLOAD), dest, opener_for(served, final_url))  # type: ignore[arg-type]
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []  # no partial file left


def test_parse_csv_normalises_whitespace_and_refuses_bad_rows() -> None:
    rows = parse_csv(csv_text([("  where   is my\ncard ", "card_arrival")]))
    assert rows == [Example("where is my card", "card_arrival")]
    with pytest.raises(ValueError, match="bad label"):
        parse_csv(csv_text([("hello", "Card Arrival!")]))
    assert parse_csv(csv_text([("hello", "Card_arrival")]))[0].label == "Card_arrival"
    assert canonical_label("Refund_not_showing_up") == "refund_not_showing_up"
    assert canonical_label("reverted_card_payment?") == "reverted_card_payment"
    with pytest.raises(ValueError, match="empty text"):
        parse_csv(csv_text([("   ", "card_arrival")]))
    with pytest.raises(ValueError, match="header"):
        parse_csv(io.StringIO("a,b\n1,2\n"))
    with pytest.raises(ValueError, match="more than"):
        parse_csv(csv_text([("a", "x"), ("b", "x")]), max_rows=1)


def test_normalise_and_bag_key_ignore_case_punctuation_accents_and_order() -> None:
    assert normalise("Where's   my CARD?!") == normalise("where s my card")
    assert normalise("Café") == "cafe"
    assert bag_key("my card is where") == bag_key("Where is my card?")


def test_dedupe_keeps_first_copy_and_drops_conflicting_labels() -> None:
    rows = [
        Example("Where is my card?", "card_arrival"),
        Example("where is my card", "card_arrival"),
        Example("Top up failed", "top_up_failed"),
        Example("top up failed!", "top_up_reverted"),
    ]
    kept, duplicates, conflicting = dedupe(rows)
    assert kept == [rows[0]]
    assert (duplicates, conflicting) == (1, 2)


def test_curate_removes_train_items_that_leak_into_test() -> None:
    train = [*synthetic(30, seed=1), Example("help please where card my", "card_arrival")]
    test = [Example("my card where please help", "card_arrival"), *synthetic(5, seed=2)]
    splits = curate(train, test, val_per_class=3, seed=7)
    assert all(v == 0 for v in splits.report.leakage.values())
    assert splits.report.train_overlapping_test_removed >= 1
    texts = {normalise(e.text) for e in splits.train + splits.val}
    assert not texts & {normalise(e.text) for e in splits.test}
    assert all(
        sum(e.label == lbl for e in splits.val) == 3
        for lbl in {"card_arrival", "lost_pin", "top_up"}
    )


def test_curate_on_randomised_data_never_leaks() -> None:
    rng = random.Random(0)
    words = [f"w{i}" for i in range(12)]
    for trial in range(20):
        rows = [
            Example(" ".join(rng.sample(words, 3)), rng.choice(["a_x", "b_y"])) for _ in range(150)
        ]
        labels_ok = {e.label for e in rows[:120]} >= {e.label for e in rows[120:]}
        if not labels_ok:  # tiny random sets can leave a label without data
            continue
        splits = curate(rows[:120], rows[120:], val_per_class=2, seed=trial)
        assert leakage(splits.as_dict()) == dict.fromkeys(splits.report.leakage, 0)
        assert len({bag_key(e.text) for e in splits.train}) == len(splits.train)


def test_curate_canonicalises_labels_so_one_intent_is_not_split_in_two() -> None:
    train = [
        Example(e.text, e.label.capitalize() if e.label == "lost_pin" else e.label)
        for e in synthetic(20, 1)
    ]
    splits = curate(train, synthetic(4, 2), val_per_class=2, seed=0)
    assert splits.report.labels_canonicalised == {"train": 20, "test": 0}
    assert {e.label for e in splits.train} == {"card_arrival", "lost_pin", "top_up"}


def test_curate_refuses_test_labels_missing_from_train() -> None:
    with pytest.raises(ValueError, match="no training data"):
        curate(
            synthetic(10, 1), [Example("brand new thing", "never_seen")], val_per_class=2, seed=0
        )


def test_split_is_seeded_and_fingerprinted() -> None:
    a = curate(synthetic(30, 1), synthetic(5, 2), 3, seed=11)
    b = curate(synthetic(30, 1), synthetic(5, 2), 3, seed=11)
    c = curate(synthetic(30, 1), synthetic(5, 2), 3, seed=12)
    assert a.report.fingerprints == b.report.fingerprints
    assert a.report.fingerprints["val"] != c.report.fingerprints["val"]
    assert fingerprint(list(reversed(a.train))) == a.report.fingerprints["train"]


def test_stratified_subset_and_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = synthetic(10, 3)
    sub = stratified_subset(rows, 2, seed=0)
    assert len(sub) == 6
    assert stratified_subset(rows, 0, seed=0) == rows
    save_jsonl(sub, tmp_path / "s.jsonl")
    assert load_jsonl(tmp_path / "s.jsonl") == sub


def test_prepare_writes_splits_and_report(tmp_path: Path) -> None:
    train_csv = csv_text([(e.text, e.label) for e in synthetic(20, 1)]).getvalue().encode()
    test_csv = csv_text([(e.text, e.label) for e in synthetic(4, 9)]).getvalue().encode()
    files = {
        "train": pin(train_csv),
        "test": RemoteFile(
            "https://example.org/test.csv", hashlib.sha256(test_csv).hexdigest(), len(test_csv)
        ),
    }

    def open_(url: str) -> IO[bytes]:
        return FakeResponse(train_csv if url.endswith("train.csv") else test_csv, url)

    splits = prepare(files, tmp_path, val_per_class=2, seed=0, opener=open_)
    for name in ("train", "val", "test", "report"):
        assert (tmp_path / "splits" / f"{name}.{'json' if name == 'report' else 'jsonl'}").exists()
    assert splits.report.final["val"] == 6
