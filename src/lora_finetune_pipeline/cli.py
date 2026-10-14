"""Command line: prepare data, run the comparison, show results, delete model and data files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import env

EXIT_BAD_INPUT = 2
EXIT_PREFLIGHT = 3


def _parser() -> argparse.ArgumentParser:
    """Argument definitions."""
    p = argparse.ArgumentParser(
        prog="lfp", description="Fine-tune or prompt? A measured QLoRA comparison."
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("pins", help="print the pinned model and dataset")
    prep = sub.add_parser(
        "prepare-data", help="download (verified), curate, de-duplicate and split"
    )
    prep.add_argument("--config", type=Path, default=Path("configs/m1.json"))
    run = sub.add_parser(
        "run", help="prompted base model vs QLoRA adapter on the held-out split (GPU)"
    )
    run.add_argument("--config", type=Path, default=Path("configs/m1.json"))
    run.add_argument(
        "--out", type=Path, required=True, help="results JSON to write (inside the repo)"
    )
    run.add_argument(
        "--keep-models", action="store_true", help="do not delete model files after the run"
    )
    run.add_argument(
        "--reuse",
        type=Path,
        help="results JSON whose zero-shot/few-shot entries were recorded on the same split; not re-run",
    )
    show = sub.add_parser("show", help="print the table for a recorded results file")
    show.add_argument("results", type=Path)
    purge = sub.add_parser(
        "purge", help="delete downloaded models, adapters and (with --data) the dataset"
    )
    purge.add_argument("--data", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = _parser().parse_args(argv)
    root = env.repo_root()
    env.configure(root)  # before torch, transformers or huggingface_hub are imported
    try:
        return int(_dispatch(args, root))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT


def _dispatch(args: argparse.Namespace, root: Path) -> int:
    """Run one subcommand."""
    from .config import DATASET_FILES, DATASET_LICENCE, MODEL, load_run_config

    if args.command == "pins":
        pins: dict[str, Any] = {
            "model": vars(MODEL),
            "dataset": {k: vars(v) for k, v in DATASET_FILES.items()},
            "dataset_licence": DATASET_LICENCE,
        }
        print(json.dumps(pins, indent=2))
        return 0
    if args.command == "show":
        from .report import table

        print(table(json.loads(args.results.read_text(encoding="utf-8"))))
        return 0
    if args.command == "purge":
        return _purge(root, data=args.data)
    cfg = load_run_config(args.config)
    splits = _prepare(root, cfg.val_per_class, cfg.seed)
    if args.command == "prepare-data":
        return 0
    reuse = _load_reuse(root, args.reuse) if args.reuse else None
    return _run(root, cfg, splits, args.out, keep_models=args.keep_models, reuse=reuse)


def _prepare(root: Path, val_per_class: int, seed: int) -> dict[str, Any]:
    """Download, curate and split; print the curation report."""
    from .config import DATASET_FILES
    from .data import prepare

    splits = prepare(DATASET_FILES, env.tmp_dir(root) / "data", val_per_class, seed)
    r = splits.report
    print(
        f"raw: {r.raw}; duplicates removed: {r.duplicates_removed}; conflicting labels removed: {r.conflicting_removed}"
    )
    print(f"labels canonicalised: {r.labels_canonicalised}")
    print(
        f"train items overlapping test removed: {r.train_overlapping_test_removed}; "
        f"same-word copies inside train removed: {r.train_same_words_removed}"
    )
    print(f"final: {r.final} over {r.labels} intents; leakage: {r.leakage}")
    return splits.as_dict()


def _load_reuse(root: Path, path: Path) -> dict[str, Any]:
    """Read recorded prompting results (inside the repo, at most 1 MiB)."""
    path = env.ensure_inside(path if path.is_absolute() else root / path, root)
    if path.stat().st_size > 1 << 20:
        raise ValueError(f"{path} is larger than 1 MiB")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("results"), dict):
        raise ValueError(f"{path} has no results object")
    return data


def _run(
    root: Path, cfg: Any, splits: dict[str, Any], out: Path, *, keep_models: bool, reuse: Any = None
) -> int:
    """Preflight, run the comparison, write results, delete model files."""
    from .config import MODEL
    from .gpu import PreflightError, preflight
    from .pipeline import QloraBackend, purge, run
    from .report import table

    out = env.ensure_inside(out if out.is_absolute() else root / out, root)
    try:
        info = preflight(cfg.min_free_ram_mib, min_free_vram_mib=2048)
    except PreflightError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return EXIT_PREFLIGHT
    print(f"preflight: {json.dumps(info)}")
    run_dir = env.tmp_dir(root) / "runs" / cfg.name
    backend = QloraBackend(MODEL, root, cfg.vram_fraction)
    results = run(cfg, splits, backend, run_dir, log=lambda s: print(s, flush=True), reuse=reuse)
    results["preflight"] = info
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not keep_models:
        model_files = [*backend.model_paths(), run_dir / "adapter", run_dir / "checkpoint.pt"]
        results["deleted"] = purge(model_files, root)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"deleted {len(results['deleted'])} model/adapter files after recording the numbers")
    print(table(results))
    return 0


def _purge(root: Path, *, data: bool) -> int:
    """Delete model caches, run outputs and optionally the dataset."""
    from .pipeline import purge

    tmp = env.tmp_dir(root)
    targets = [tmp / "hf", tmp / "runs"] + ([tmp / "data"] if data else [])
    removed = purge(targets, root)
    for item in removed:
        print(f"deleted {item['path']} ({item['bytes']:,} bytes)")
    print(f"{len(removed)} files deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
