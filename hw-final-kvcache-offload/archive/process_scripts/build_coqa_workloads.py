#!/usr/bin/env python3
"""Build vLLM prefix-cache workloads from the CoQA dataset.

CoQA rows contain one story and a sequence of conversational questions. This
builder treats each story as a prefix group and emits one request per selected
question so vLLM can reuse the story prefix across requests.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from build_baseline_workloads import TokenCounter, write_jsonl


DEFAULT_DATASET = "stanfordnlp/coqa"
DEFAULT_SPLIT = "validation"
DEFAULT_OUTPUT = Path("data/workloads/coqa/coqa_validation_story_reuse.jsonl")
DEFAULT_LOCAL_DIR = Path("data/raw/coqa")
DEFAULT_MAX_PROMPT_TOKENS = 1800


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def load_coqa_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_jsonl is not None:
        return read_jsonl(args.input_jsonl)

    local_split = args.local_dir / f"{args.split}.jsonl"
    if args.prefer_local and local_split.exists():
        return read_jsonl(local_split)

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency exists in project env.
        raise RuntimeError("Install the datasets package or pass --input-jsonl.") from exc

    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    return [dict(row) for row in dataset]


def answer_text(row: dict[str, Any], turn_index: int) -> str:
    answers = row.get("answers") or {}
    if isinstance(answers, dict):
        input_text = answers.get("input_text") or []
        if turn_index < len(input_text):
            return str(input_text[turn_index])
    return ""


def story_id(row: dict[str, Any], story_index: int) -> str:
    for key in ["id", "story_id", "filename", "name"]:
        value = row.get(key)
        if value not in [None, ""]:
            return str(value)
    return f"story-{story_index:05d}"


def build_prefix(story: str) -> str:
    return (
        "System: Answer the question using only the story.\n"
        "Story:\n"
        f"{story.strip()}\n"
    )


def build_suffix(question: str) -> str:
    return f"\nQuestion: {question.strip()}\nAnswer in one short sentence:"


def iter_selected_stories(
    rows: list[dict[str, Any]],
    counter: TokenCounter,
    args: argparse.Namespace,
) -> Iterable[tuple[int, dict[str, Any], str, int]]:
    selected = 0
    for story_index, row in enumerate(rows):
        story = str(row.get("story") or "").strip()
        questions = row.get("questions") or []
        if not story or not questions:
            continue
        prefix = build_prefix(story)
        prefix_tokens = counter.count(prefix)
        if prefix_tokens < args.min_prefix_tokens:
            continue
        if args.max_prefix_tokens is not None and prefix_tokens > args.max_prefix_tokens:
            continue
        selected += 1
        yield story_index, row, prefix, prefix_tokens
        if selected >= args.limit_stories:
            return


def order_requests(rows_by_story: list[list[dict[str, Any]]], access_pattern: str) -> list[dict[str, Any]]:
    if access_pattern == "grouped":
        return [request for story_rows in rows_by_story for request in story_rows]
    if access_pattern == "round_robin":
        ordered: list[dict[str, Any]] = []
        max_turns = max((len(story_rows) for story_rows in rows_by_story), default=0)
        for turn_index in range(max_turns):
            for story_rows in rows_by_story:
                if turn_index < len(story_rows):
                    ordered.append(story_rows[turn_index])
        return ordered
    raise ValueError(f"Unknown access pattern: {access_pattern}")


def make_workload_rows(
    source_rows: list[dict[str, Any]],
    counter: TokenCounter,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows_by_story: list[list[dict[str, Any]]] = []
    for selected_index, (story_index, row, prefix, prefix_tokens) in enumerate(
        iter_selected_stories(source_rows, counter, args)
    ):
        questions = [str(question) for question in (row.get("questions") or [])]
        coqa_id = story_id(row, story_index)
        group_rows: list[dict[str, Any]] = []
        turns_used = 0
        for turn_index, question in enumerate(questions[: args.turns_per_story]):
            suffix = build_suffix(question)
            prompt = prefix + suffix
            suffix_tokens = counter.count(suffix)
            prompt_tokens = counter.count(prompt)
            if prompt_tokens > args.max_prompt_tokens:
                continue
            is_warm = turns_used > 0
            request_id = f"coqa-{args.split}-s{selected_index:04d}-t{turn_index:02d}"
            group_rows.append(
                {
                    "id": request_id,
                    "request_id": request_id,
                    "scenario_id": "coqa_story_reuse",
                    "scenario_label": f"CoQA story reuse, {args.turns_per_story} turns/story",
                    "experiment_type": "coqa_story_reuse",
                    "target_prefix_tokens": "",
                    "actual_prefix_tokens": prefix_tokens,
                    "prefix_target_tokens": "",
                    "prefix_tokens": prefix_tokens,
                    "suffix_tokens": suffix_tokens,
                    "prompt_tokens": prompt_tokens,
                    "reuse_ratio_target": "",
                    "reuse_ratio_actual": "",
                    "num_prefix_groups": args.limit_stories,
                    "total_requests": "",
                    "requests_per_group": "",
                    "request_index_in_group": turns_used,
                    "occurrence_index": turns_used,
                    "is_warm_request": is_warm,
                    "expected_cache_state": "warm" if is_warm else "cold",
                    "expected_reused_prefix_tokens": prefix_tokens if is_warm else 0,
                    "expected_reuse_ratio_token": (prefix_tokens / prompt_tokens) if is_warm else 0.0,
                    "prefix_group": f"coqa-{args.split}-story-{selected_index:04d}",
                    "prefix_group_index": selected_index,
                    "workload_name": args.workload_name,
                    "phase": "warm" if is_warm else "seed",
                    "pressure_level": "",
                    "pressure_group_role": "",
                    "pressure_group_id": "",
                    "sequence_pass": turns_used,
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                    "coqa_story_id": coqa_id,
                    "coqa_source": row.get("source", ""),
                    "coqa_turn_id": turn_index,
                    "coqa_answer": answer_text(row, turn_index),
                    "prompt": prompt,
                }
            )
            turns_used += 1
        if len(group_rows) >= args.min_turns_per_story:
            rows_by_story.append(group_rows)

    ordered = order_requests(rows_by_story, args.access_pattern)
    group_counts = Counter(row["prefix_group"] for row in ordered)
    num_groups = len(group_counts)
    total_requests = len(ordered)
    actual_reuse_ratio = 1.0 - (num_groups / total_requests) if total_requests else 0.0
    for index, row in enumerate(ordered):
        row["request_id"] = row["id"]
        row["num_prefix_groups"] = num_groups
        row["total_requests"] = total_requests
        row["requests_per_group"] = group_counts[row["prefix_group"]]
        row["reuse_ratio_actual"] = actual_reuse_ratio
        row["request_order"] = index
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Use local CoQA-style rows instead of downloading from Hugging Face.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help="Directory containing local <split>.jsonl exports from src/download_coqa.py.",
    )
    parser.add_argument(
        "--prefer-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read data/raw/coqa/<split>.jsonl when it exists before using Hugging Face.",
    )
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"), help="datasets cache dir; defaults to HF_HOME when set.")
    parser.add_argument("--limit-stories", type=int, default=40)
    parser.add_argument("--turns-per-story", type=int, default=4)
    parser.add_argument("--min-turns-per-story", type=int, default=2)
    parser.add_argument("--min-prefix-tokens", type=int, default=128)
    parser.add_argument("--max-prefix-tokens", type=int, default=1536)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--workload-name", default="coqa_story_reuse")
    parser.add_argument("--access-pattern", choices=["grouped", "round_robin"], default="round_robin")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN05B_PATH"),
        help="Local tokenizer/model path. Defaults to QWEN05B_PATH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_stories <= 0:
        raise ValueError("--limit-stories must be positive")
    if args.turns_per_story <= 0:
        raise ValueError("--turns-per-story must be positive")
    if args.min_turns_per_story <= 0:
        raise ValueError("--min-turns-per-story must be positive")

    counter = TokenCounter(args.model_path)
    source_rows = load_coqa_rows(args)
    rows = make_workload_rows(source_rows, counter, args)
    if not rows:
        raise RuntimeError("No CoQA workload rows matched the selected token and turn filters.")
    write_jsonl(args.output, rows)

    prompt_tokens = [int(row["prompt_tokens"]) for row in rows]
    prefix_tokens = [int(row["prefix_tokens"]) for row in rows]
    states = Counter(row["expected_cache_state"] for row in rows)
    print(f"Wrote {len(rows)} CoQA requests to {args.output}")
    print(f"  stories: {len(set(row['prefix_group'] for row in rows))}")
    print(f"  access pattern: {args.access_pattern}")
    print(f"  expected states: " + ", ".join(f"{k}={v}" for k, v in sorted(states.items())))
    print(f"  prefix tokens: min={min(prefix_tokens)}, max={max(prefix_tokens)}")
    print(f"  prompt tokens: min={min(prompt_tokens)}, max={max(prompt_tokens)}")


if __name__ == "__main__":
    main()
