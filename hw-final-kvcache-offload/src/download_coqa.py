#!/usr/bin/env python3
"""Download CoQA from Hugging Face and export local JSONL copies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "stanfordnlp/coqa"
DEFAULT_OUTPUT_DIR = Path("data/raw/coqa")
DEFAULT_SPLITS = ["train", "validation"]


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_split_jsonl(dataset: Any, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(dict(row), ensure_ascii=False, default=json_default) + "\n")
            count += 1
    return count


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_splits(value: str) -> list[str]:
    splits = [part.strip() for part in value.split(",") if part.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("At least one split is required.")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--splits", type=parse_splits, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"), help="datasets cache dir; defaults to HF_HOME when set.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing local JSONL exports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency exists in project env.
        raise RuntimeError("The datasets package is required to download CoQA.") from exc

    metadata: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "cache_dir": args.cache_dir,
        "splits": {},
    }
    for split in args.splits:
        output = args.output_dir / f"{split}.jsonl"
        if output.exists() and not args.force:
            rows = sum(1 for line in output.open(encoding="utf-8") if line.strip())
            print(f"Keeping existing {output} ({rows} rows). Use --force to overwrite.")
        else:
            dataset = load_dataset(args.dataset_name, split=split, cache_dir=args.cache_dir)
            rows = write_split_jsonl(dataset, output)
            print(f"Wrote {rows} rows to {output}")
        metadata["splits"][split] = {
            "path": str(output),
            "rows": rows,
        }
    write_metadata(args.output_dir / "dataset_info.json", metadata)
    print(f"Wrote metadata to {args.output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
