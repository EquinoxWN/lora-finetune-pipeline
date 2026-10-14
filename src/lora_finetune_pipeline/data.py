"""Verified download, curation, de-duplication and a leakage-checked split of BANKING77."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import unicodedata
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

from .config import RemoteFile

LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}[?!.]?$")
CANONICAL = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_TEXT_CHARS = 1000
MAX_ROWS = 100_000
CHUNK = 1 << 16

Opener = Callable[[str], IO[bytes]]


@dataclass(frozen=True)
class Example:
    """One labelled customer message."""

    text: str
    label: str


class DownloadError(RuntimeError):
    """A download was refused or did not match its pin."""


def _https_open(url: str) -> IO[bytes]:
    """Open an HTTPS URL with a timeout."""
    if not url.startswith("https://"):
        raise DownloadError(f"refusing non-HTTPS URL {url}")
    response: IO[bytes] = urllib.request.urlopen(url, timeout=60)  # noqa: S310  (scheme checked above)
    return response


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(remote: RemoteFile, dest: Path, opener: Opener = _https_open) -> Path:
    """Fetch a pinned file; keep it only if size and SHA-256 match, leaving no partial file."""
    if dest.exists() and sha256_file(dest) == remote.sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest, size = hashlib.sha256(), 0
    try:
        with opener(remote.url) as response, part.open("wb") as out:
            final_url = getattr(response, "url", remote.url) or remote.url
            if not str(final_url).startswith("https://"):
                raise DownloadError(f"redirected to a non-HTTPS URL: {final_url}")
            for chunk in iter(lambda: response.read(CHUNK), b""):
                size += len(chunk)
                if size > remote.size:
                    raise DownloadError(
                        f"{remote.url} is larger than the pinned {remote.size} bytes"
                    )
                digest.update(chunk)
                out.write(chunk)
        if size != remote.size or digest.hexdigest() != remote.sha256:
            raise DownloadError(f"{remote.url} does not match its pinned size and SHA-256")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    return dest


def parse_csv(stream: Iterable[str], max_rows: int = MAX_ROWS) -> list[Example]:
    """Read text,category rows, refusing malformed labels and oversized inputs."""
    reader = csv.DictReader(stream)
    if reader.fieldnames is None or {"text", "category"} - set(reader.fieldnames):
        raise ValueError("expected a CSV header with text and category columns")
    rows: list[Example] = []
    for line, row in enumerate(reader, start=2):
        if len(rows) >= max_rows:
            raise ValueError(f"more than {max_rows} rows")
        text = " ".join((row.get("text") or "").split())
        label = (row.get("category") or "").strip()
        if not text:
            raise ValueError(f"line {line}: empty text")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"line {line}: text longer than {MAX_TEXT_CHARS} characters")
        if not LABEL.match(label):
            raise ValueError(f"line {line}: bad label {label!r}")
        rows.append(Example(text, label))
    return rows


def read_csv(path: Path) -> list[Example]:
    """Parse a downloaded CSV file."""
    with path.open(encoding="utf-8", newline="") as handle:
        return parse_csv(handle)


def normalise(text: str) -> str:
    """Case-, accent-, punctuation- and whitespace-insensitive form used for duplicate checks."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w]+", " ", folded).split())


def bag_key(text: str) -> str:
    """Order-insensitive key: the sorted set of normalised words (catches reordered near-duplicates)."""
    return " ".join(sorted(set(normalise(text).split())))


@dataclass
class CurationReport:
    """What curation removed and the leakage counts after the split."""

    raw: dict[str, int] = field(default_factory=dict)
    duplicates_removed: dict[str, int] = field(default_factory=dict)
    conflicting_removed: dict[str, int] = field(default_factory=dict)
    labels_canonicalised: dict[str, int] = field(default_factory=dict)
    train_overlapping_test_removed: int = 0
    train_same_words_removed: int = 0
    final: dict[str, int] = field(default_factory=dict)
    labels: int = 0
    leakage: dict[str, int] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready form."""
        return asdict(self)


@dataclass(frozen=True)
class Splits:
    """Train, validation and held-out test sets with the curation report."""

    train: list[Example]
    val: list[Example]
    test: list[Example]
    report: CurationReport

    def as_dict(self) -> dict[str, list[Example]]:
        """The three splits by name."""
        return {"train": self.train, "val": self.val, "test": self.test}


def canonical_label(label: str) -> str:
    """Lower-case and drop trailing punctuation ('Refund_not_showing_up', 'reverted_card_payment?')."""
    out = label.lower().rstrip("?!.")
    if not CANONICAL.match(out):
        raise ValueError(f"bad label {label!r}")
    return out


def fold_labels(rows: Sequence[Example]) -> list[Example]:
    """Rows with canonical labels."""
    return [Example(e.text, canonical_label(e.label)) for e in rows]


def dedupe(rows: Sequence[Example]) -> tuple[list[Example], int, int]:
    """Keep the first copy of each normalised text; drop every copy of texts with conflicting labels."""
    labels_by_key: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels_by_key[normalise(row.text)].add(row.label)
    kept: list[Example] = []
    seen: set[str] = set()
    duplicates = conflicting = 0
    for row in rows:
        key = normalise(row.text)
        if len(labels_by_key[key]) > 1:
            conflicting += 1
        elif key in seen:
            duplicates += 1
        else:
            seen.add(key)
            kept.append(row)
    return kept, duplicates, conflicting


def leakage(splits: dict[str, list[Example]]) -> dict[str, int]:
    """Count shared normalised texts and shared word bags between every pair of splits."""
    names = sorted(splits)
    exact = {n: {normalise(e.text) for e in splits[n]} for n in names}
    bags = {n: {bag_key(e.text) for e in splits[n]} for n in names}
    out: dict[str, int] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out[f"{a}/{b} exact"] = len(exact[a] & exact[b])
            out[f"{a}/{b} same-words"] = len(bags[a] & bags[b])
    return out


def fingerprint(rows: Sequence[Example]) -> str:
    """Order-independent SHA-256 of a split, to check a reproduction used the same data."""
    lines = sorted(f"{e.label}\t{normalise(e.text)}" for e in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def curate(
    train_raw: Sequence[Example], test_raw: Sequence[Example], val_per_class: int, seed: int
) -> Splits:
    """De-duplicate, remove train items that leak into test, carve a stratified validation set."""
    report = CurationReport(raw={"train": len(train_raw), "test": len(test_raw)})
    train_folded, test_folded = fold_labels(train_raw), fold_labels(test_raw)
    report.labels_canonicalised = {
        "train": sum(a.label != b.label for a, b in zip(train_raw, train_folded, strict=True)),
        "test": sum(a.label != b.label for a, b in zip(test_raw, test_folded, strict=True)),
    }
    train, dup_tr, con_tr = dedupe(train_folded)
    test, dup_te, con_te = dedupe(test_folded)
    report.duplicates_removed = {"train": dup_tr, "test": dup_te}
    report.conflicting_removed = {"train": con_tr, "test": con_te}

    test_exact = {normalise(e.text) for e in test}
    test_bags = {bag_key(e.text) for e in test}
    clean = [
        e for e in train if normalise(e.text) not in test_exact and bag_key(e.text) not in test_bags
    ]
    report.train_overlapping_test_removed = len(train) - len(clean)
    # Same-word bags inside train would also leak into validation: keep one per bag.
    before_bags = len(clean)
    clean = _first_per_bag(clean)
    report.train_same_words_removed = before_bags - len(clean)

    by_label: dict[str, list[Example]] = defaultdict(list)
    for e in clean:
        by_label[e.label].append(e)
    missing = {e.label for e in test} - set(by_label)
    if missing:
        raise ValueError(f"test labels with no training data: {sorted(missing)}")
    rng = random.Random(seed)
    train_out: list[Example] = []
    val_out: list[Example] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        if len(group) <= val_per_class:
            raise ValueError(f"label {label} has too few examples for validation")
        val_out.extend(group[:val_per_class])
        train_out.extend(group[val_per_class:])
    rng.shuffle(train_out)

    splits = Splits(train_out, val_out, list(test), report)
    report.final = {k: len(v) for k, v in splits.as_dict().items()}
    report.labels = len(by_label)
    report.leakage = leakage(splits.as_dict())
    report.fingerprints = {k: fingerprint(v) for k, v in splits.as_dict().items()}
    if any(report.leakage.values()):
        raise ValueError(f"leakage between splits: {report.leakage}")
    return splits


def _first_per_bag(rows: Sequence[Example]) -> list[Example]:
    """Keep the first example of each word bag."""
    seen: set[str] = set()
    out: list[Example] = []
    for e in rows:
        key = bag_key(e.text)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def stratified_subset(rows: Sequence[Example], per_class: int, seed: int) -> list[Example]:
    """At most per_class examples of each label, chosen with a seed (0 keeps everything)."""
    if per_class <= 0:
        return list(rows)
    by_label: dict[str, list[Example]] = defaultdict(list)
    for e in rows:
        by_label[e.label].append(e)
    rng = random.Random(seed)
    out: list[Example] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        out.extend(group[:per_class])
    return out


def save_jsonl(rows: Sequence[Example], path: Path) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for e in rows:
            out.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[Example]:
    """Read rows written by save_jsonl."""
    rows: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            rows.append(Example(str(raw["text"]), str(raw["label"])))
    return rows


def prepare(
    files: dict[str, RemoteFile],
    data_dir: Path,
    val_per_class: int,
    seed: int,
    opener: Opener = _https_open,
) -> Splits:
    """Download (verified), curate and split; write the splits and the curation report."""
    raw = {
        name: read_csv(download(remote, data_dir / "raw" / f"{name}.csv", opener))
        for name, remote in files.items()
    }
    splits = curate(raw["train"], raw["test"], val_per_class, seed)
    for name, rows in splits.as_dict().items():
        save_jsonl(rows, data_dir / "splits" / f"{name}.jsonl")
    (data_dir / "splits" / "report.json").write_text(
        json.dumps(splits.report.to_dict(), indent=2), encoding="utf-8"
    )
    return splits


def labels_of(rows: Iterable[Example]) -> list[str]:
    """Sorted distinct labels."""
    return sorted({e.label for e in rows})


def csv_text(rows: Sequence[tuple[str, str]]) -> io.StringIO:
    """Build an in-memory text,category CSV (for tests and synthetic data)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["text", "category"])
    writer.writerows(rows)
    buf.seek(0)
    return buf
