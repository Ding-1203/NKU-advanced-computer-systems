#!/usr/bin/env python3
"""Build quantitative synthetic workloads for vLLM prefix-cache baselines."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUESTION_BANK = [
    "What is the central problem described in the story?",
    "Which detail is most useful for making a scheduling decision?",
    "Summarize the user's likely next action in one sentence.",
    "What constraint should the assistant remember?",
    "Which event happened before the final decision?",
    "What is the safest concise answer to the user's question?",
    "Name one tradeoff implied by the story.",
    "What information is missing from the story?",
    "How should the system respond if resources are limited?",
    "Which part of the context is reused across requests?",
    "What operational metric should be monitored?",
    "State the most relevant conclusion using fewer than fifteen words.",
]

BASELINE_OUTPUT_DIR = Path("data/workloads/baseline_v1_1")
PREFIX_LENGTH_VALUES = [128, 256, 512, 1024, 1536]
REUSE_RATIO_VALUES = [0.0, 0.25, 0.5, 0.75, 0.9]


@dataclass(frozen=True)
class ScenarioSpec:
    experiment_type: str
    target_prefix_tokens: int
    reuse_ratio_target: float
    total_requests: int

    @property
    def scenario_id(self) -> str:
        reuse_pct = int(round(self.reuse_ratio_target * 100))
        return (
            f"{self.experiment_type}"
            f"_prefix{self.target_prefix_tokens}"
            f"_reuse{reuse_pct}"
        )

    @property
    def scenario_label(self) -> str:
        reuse_pct = int(round(self.reuse_ratio_target * 100))
        return f"{self.target_prefix_tokens} tok, {reuse_pct}% reuse"


class TokenCounter:
    def __init__(self, model_path: str | None) -> None:
        self.tokenizer = None
        if model_path:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=False,
                    local_files_only=True,
                )
            except Exception as exc:  # pragma: no cover - fallback is explicit.
                print(f"Warning: failed to load tokenizer from {model_path}: {exc}")
                print("Falling back to whitespace token estimates.")

    def count(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.split())
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def build_group_sequence(total_requests: int, reuse_ratio: float) -> list[int]:
    if not 0 <= reuse_ratio < 1:
        raise ValueError("reuse_ratio must be in [0, 1).")
    warm_requests = int(round(total_requests * reuse_ratio))
    num_groups = total_requests - warm_requests
    if num_groups <= 0:
        raise ValueError("reuse_ratio and total_requests leave no cold requests.")

    sequence = list(range(num_groups))
    sequence.extend(index % num_groups for index in range(warm_requests))
    if len(sequence) != total_requests:
        raise AssertionError("Invalid group sequence length.")
    return sequence


def build_prefix(
    counter: TokenCounter,
    spec: ScenarioSpec,
    group_id: int,
) -> tuple[str, int]:
    header = (
        "System: Use only this shared prefix context.\n"
        f"Experiment {spec.experiment_type}; target {spec.target_prefix_tokens}; "
        f"reuse {spec.reuse_ratio_target:.2f}; group {group_id}; "
        f"marker {spec.scenario_id}_{group_id}.\n"
        "Story:"
    )
    filler_unit = " cache"
    header_tokens = counter.count(header)
    if header_tokens >= spec.target_prefix_tokens:
        return header.rstrip(), header_tokens

    one_unit_tokens = max(1, counter.count(header + filler_unit) - header_tokens)
    filler_count = max(0, (spec.target_prefix_tokens - header_tokens) // one_unit_tokens)
    text = header + (filler_unit * filler_count)
    token_count = counter.count(text)

    while token_count < spec.target_prefix_tokens:
        text += filler_unit
        token_count = counter.count(text)

    while filler_count > 0:
        shorter = text[: -len(filler_unit)]
        shorter_count = counter.count(shorter)
        if shorter_count < spec.target_prefix_tokens:
            break
        text = shorter
        token_count = shorter_count
        filler_count -= 1

    return text.rstrip(), token_count


def build_rows_for_spec(
    counter: TokenCounter,
    spec: ScenarioSpec,
    request_offset: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    group_sequence = build_group_sequence(spec.total_requests, spec.reuse_ratio_target)
    group_totals = Counter(group_sequence)
    num_prefix_groups = len(group_totals)
    actual_reuse_ratio = 1.0 - (num_prefix_groups / spec.total_requests)
    prefixes = {
        group_id: build_prefix(counter=counter, spec=spec, group_id=group_id)
        for group_id in group_totals
    }

    rows: list[dict[str, Any]] = []
    seen_by_group: Counter[int] = Counter()
    for local_request_index, group_id in enumerate(group_sequence):
        prefix, actual_prefix_tokens = prefixes[group_id]
        request_index_in_group = seen_by_group[group_id]
        seen_by_group[group_id] += 1
        is_warm_request = request_index_in_group > 0
        question = QUESTION_BANK[local_request_index % len(QUESTION_BANK)]
        suffix = (
            f"\n\nQuestion {local_request_index:03d}: {question}\n"
            "Answer in one short sentence:"
        )
        prompt = prefix + suffix
        suffix_tokens = counter.count(suffix)
        prompt_tokens = counter.count(prompt)
        expected_reused_prefix_tokens = actual_prefix_tokens if is_warm_request else 0
        expected_reuse_ratio_token = (
            expected_reused_prefix_tokens / prompt_tokens if prompt_tokens else 0.0
        )
        request_id = f"req-{request_offset + local_request_index:05d}"
        rows.append(
            {
                "id": request_id,
                "request_id": request_id,
                "scenario_id": spec.scenario_id,
                "scenario_label": spec.scenario_label,
                "experiment_type": spec.experiment_type,
                "target_prefix_tokens": spec.target_prefix_tokens,
                "actual_prefix_tokens": actual_prefix_tokens,
                "prefix_target_tokens": spec.target_prefix_tokens,
                "prefix_tokens": actual_prefix_tokens,
                "suffix_tokens": suffix_tokens,
                "prompt_tokens": prompt_tokens,
                "reuse_ratio_target": spec.reuse_ratio_target,
                "reuse_ratio_actual": actual_reuse_ratio,
                "num_prefix_groups": num_prefix_groups,
                "total_requests": spec.total_requests,
                "requests_per_group": group_totals[group_id],
                "request_index_in_group": request_index_in_group,
                "occurrence_index": request_index_in_group,
                "is_warm_request": is_warm_request,
                "expected_cache_state": "warm" if is_warm_request else "cold",
                "expected_reused_prefix_tokens": expected_reused_prefix_tokens,
                "expected_reuse_ratio_token": expected_reuse_ratio_token,
                "prefix_group": f"{spec.scenario_id}-g{group_id}",
                "prefix_group_index": group_id,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "prompt": prompt,
            }
        )
    return rows


def workload_specs(scale: str) -> dict[str, list[ScenarioSpec]]:
    if scale == "smoke":
        return {
            "baseline_smoke": [
                ScenarioSpec("prefix_length_sweep", 128, 0.5, 4),
                ScenarioSpec("prefix_reuse_matrix", 512, 0.5, 4),
            ]
        }

    prefix_length_sweep = [
        ScenarioSpec("prefix_length_sweep", tokens, 0.9, 20)
        for tokens in PREFIX_LENGTH_VALUES
    ]
    reuse_ratio_sweep = [
        ScenarioSpec("reuse_ratio_sweep", 1024, ratio, 60)
        for ratio in REUSE_RATIO_VALUES
    ]
    prefix_reuse_matrix = [
        ScenarioSpec("prefix_reuse_matrix", tokens, ratio, 20)
        for tokens in PREFIX_LENGTH_VALUES
        for ratio in REUSE_RATIO_VALUES
    ]
    return {
        "prefix_length_sweep": prefix_length_sweep,
        "reuse_ratio_sweep": reuse_ratio_sweep,
        "prefix_reuse_matrix": prefix_reuse_matrix,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(name: str, rows: list[dict[str, Any]], output: Path) -> None:
    by_scenario = Counter(row["scenario_id"] for row in rows)
    by_cache_state = Counter(row["expected_cache_state"] for row in rows)
    prompt_tokens = [row["prompt_tokens"] for row in rows]
    print(f"Wrote {len(rows)} requests to {output}")
    print(f"  workload: {name}")
    print(f"  scenarios: {len(by_scenario)}")
    print(
        "  expected cache states: "
        + ", ".join(f"{key}={value}" for key, value in sorted(by_cache_state.items()))
    )
    print(f"  prompt tokens: min={min(prompt_tokens)}, max={max(prompt_tokens)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASELINE_OUTPUT_DIR,
        help="Directory for generated JSONL workloads.",
    )
    parser.add_argument(
        "--only",
        choices=[
            "all",
            "prefix_length_sweep",
            "reuse_ratio_sweep",
            "prefix_reuse_matrix",
            "baseline_smoke",
        ],
        default="all",
        help="Generate all workloads or one named workload.",
    )
    parser.add_argument(
        "--scale",
        choices=["full", "smoke"],
        default="full",
        help="Generate full quantitative workloads or a tiny smoke workload.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN05B_PATH"),
        help="Local tokenizer/model path. Defaults to QWEN05B_PATH.",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counter = TokenCounter(args.model_path)
    specs_by_name = workload_specs(args.scale)
    if args.only != "all":
        if args.only not in specs_by_name:
            raise ValueError(f"{args.only} is not available for --scale {args.scale}")
        specs_by_name = {args.only: specs_by_name[args.only]}

    request_offset = 0
    for name, specs in specs_by_name.items():
        rows: list[dict[str, Any]] = []
        for spec in specs:
            spec_rows = build_rows_for_spec(
                counter=counter,
                spec=spec,
                request_offset=request_offset,
                max_tokens=args.max_tokens,
            )
            rows.extend(spec_rows)
            request_offset += len(spec_rows)
        output = args.output_dir / f"{name}.jsonl"
        write_jsonl(output, rows)
        print_summary(name, rows, output)


if __name__ == "__main__":
    main()
