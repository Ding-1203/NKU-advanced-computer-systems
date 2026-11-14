#!/usr/bin/env python3
"""Build cache-pressure workloads for vLLM prefix-cache observation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from build_baseline_workloads import QUESTION_BANK, TokenCounter


DEFAULT_SMOKE_OUTPUT = Path("data/workloads/cache_observation/cache_pressure_smoke.jsonl")
DEFAULT_FORMAL_OUTPUT = Path("data/workloads/cache_observation/cache_pressure_sweep.jsonl")
SMOKE_LEVELS = [16, 32]
FORMAL_LEVELS = [8, 16, 24, 32, 48]


@dataclass(frozen=True)
class PressureSpec:
    prefix_groups: int
    prefix_tokens: int
    filler_multiplier: int

    @property
    def scenario_id(self) -> str:
        return f"cache_pressure_groups{self.prefix_groups}_prefix{self.prefix_tokens}"

    @property
    def scenario_label(self) -> str:
        return f"{self.prefix_groups} groups, {self.prefix_tokens} tok pressure"


def count_tokens(counter: TokenCounter, text: str) -> int:
    return counter.count(text)


def build_prefix(counter: TokenCounter, spec: PressureSpec, role: str, group_id: int) -> tuple[str, int]:
    header = (
        "System: Use only this shared cache-pressure prefix context.\n"
        f"Experiment cache_pressure; pressure groups {spec.prefix_groups}; "
        f"target {spec.prefix_tokens}; role {role}; group {group_id}; "
        f"marker {spec.scenario_id}_{role}_{group_id}.\n"
        "Story:"
    )
    filler_unit = " cache"
    header_tokens = count_tokens(counter, header)
    if header_tokens >= spec.prefix_tokens:
        return header.rstrip(), header_tokens

    one_unit_tokens = max(1, count_tokens(counter, header + filler_unit) - header_tokens)
    filler_count = max(0, (spec.prefix_tokens - header_tokens) // one_unit_tokens)
    text = header + (filler_unit * filler_count)
    token_count = count_tokens(counter, text)
    while token_count < spec.prefix_tokens:
        text += filler_unit
        token_count = count_tokens(counter, text)
    while filler_count > 0:
        shorter = text[: -len(filler_unit)]
        shorter_count = count_tokens(counter, shorter)
        if shorter_count < spec.prefix_tokens:
            break
        text = shorter
        token_count = shorter_count
        filler_count -= 1
    return text.rstrip(), token_count


def make_row(
    counter: TokenCounter,
    spec: PressureSpec,
    workload_name: str,
    request_index: int,
    phase: str,
    role: str,
    group_id: int,
    sequence_pass: int,
    max_tokens: int,
) -> dict[str, Any]:
    prefix, prefix_tokens = build_prefix(counter, spec, role, group_id)
    question = QUESTION_BANK[request_index % len(QUESTION_BANK)]
    suffix = (
        f"\n\nQuestion {request_index:03d}: {question}\n"
        "Answer in one short sentence:"
    )
    prompt = prefix + suffix
    suffix_tokens = counter.count(suffix)
    prompt_tokens = counter.count(prompt)
    request_id = f"pressure-l{spec.prefix_groups}-{phase}-g{group_id:03d}-p{sequence_pass}"
    expected_reused = prefix_tokens if phase == "revisit" else 0
    return {
        "id": request_id,
        "request_id": request_id,
        "scenario_id": spec.scenario_id,
        "scenario_label": spec.scenario_label,
        "experiment_type": "cache_pressure",
        "target_prefix_tokens": spec.prefix_tokens,
        "actual_prefix_tokens": prefix_tokens,
        "prefix_target_tokens": spec.prefix_tokens,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "prompt_tokens": prompt_tokens,
        "reuse_ratio_target": "",
        "reuse_ratio_actual": "",
        "num_prefix_groups": spec.prefix_groups,
        "total_requests": spec.prefix_groups * (spec.filler_multiplier + 2),
        "requests_per_group": 2 if role == "seed" else 1,
        "request_index_in_group": 1 if phase == "revisit" else 0,
        "occurrence_index": 1 if phase == "revisit" else 0,
        "is_warm_request": phase == "revisit",
        "expected_cache_state": "revisit_candidate" if phase == "revisit" else "cold",
        "expected_reused_prefix_tokens": expected_reused,
        "expected_reuse_ratio_token": expected_reused / prompt_tokens if prompt_tokens else 0.0,
        "prefix_group": f"{spec.scenario_id}-{role}-g{group_id}",
        "prefix_group_index": group_id,
        "workload_name": workload_name,
        "phase": phase,
        "pressure_level": spec.prefix_groups,
        "pressure_group_role": role,
        "pressure_group_id": group_id,
        "sequence_pass": sequence_pass,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "prompt": prompt,
    }


def build_rows(
    counter: TokenCounter,
    specs: list[PressureSpec],
    workload_name: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    request_index = 0
    for spec in specs:
        for group_id in range(spec.prefix_groups):
            rows.append(
                make_row(
                    counter=counter,
                    spec=spec,
                    workload_name=workload_name,
                    request_index=request_index,
                    phase="seed",
                    role="seed",
                    group_id=group_id,
                    sequence_pass=0,
                    max_tokens=max_tokens,
                )
            )
            request_index += 1

        filler_groups = spec.prefix_groups * spec.filler_multiplier
        for group_id in range(filler_groups):
            rows.append(
                make_row(
                    counter=counter,
                    spec=spec,
                    workload_name=workload_name,
                    request_index=request_index,
                    phase="filler",
                    role="filler",
                    group_id=group_id,
                    sequence_pass=0,
                    max_tokens=max_tokens,
                )
            )
            request_index += 1

        for group_id in range(spec.prefix_groups):
            rows.append(
                make_row(
                    counter=counter,
                    spec=spec,
                    workload_name=workload_name,
                    request_index=request_index,
                    phase="revisit",
                    role="seed",
                    group_id=group_id,
                    sequence_pass=1,
                    max_tokens=max_tokens,
                )
            )
            request_index += 1
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_levels(value: str) -> list[int]:
    levels = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not levels:
        raise argparse.ArgumentTypeError("At least one pressure level is required.")
    if any(level <= 0 for level in levels):
        raise argparse.ArgumentTypeError("Pressure levels must be positive integers.")
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["smoke", "formal", "custom"],
        default="smoke",
        help="Use smoke levels, formal pressure-sweep levels, or explicit --levels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults depend on --preset.",
    )
    parser.add_argument(
        "--levels",
        type=parse_levels,
        default=None,
        help="Comma-separated prefix group counts. Overrides --preset levels.",
    )
    parser.add_argument(
        "--workload-name",
        default=None,
        help="Name written into each row. Defaults to output file stem.",
    )
    parser.add_argument("--prefix-tokens", type=int, default=1536)
    parser.add_argument("--filler-multiplier", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN05B_PATH"),
        help="Local tokenizer/model path. Defaults to QWEN05B_PATH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.levels is not None:
        levels = args.levels
    elif args.preset == "formal":
        levels = FORMAL_LEVELS
    else:
        levels = SMOKE_LEVELS

    if args.output is None:
        output = DEFAULT_FORMAL_OUTPUT if args.preset == "formal" else DEFAULT_SMOKE_OUTPUT
    else:
        output = args.output
    workload_name = args.workload_name or output.stem

    counter = TokenCounter(args.model_path)
    specs = [
        PressureSpec(
            prefix_groups=level,
            prefix_tokens=args.prefix_tokens,
            filler_multiplier=args.filler_multiplier,
        )
        for level in levels
    ]
    rows = build_rows(counter, specs, workload_name, args.max_tokens)
    write_jsonl(output, rows)

    prompt_tokens = [row["prompt_tokens"] for row in rows]
    print(f"Wrote {len(rows)} requests to {output}")
    print(f"  workload: {workload_name}")
    print(f"  levels: {', '.join(str(level) for level in levels)}")
    print(f"  prefix tokens target: {args.prefix_tokens}")
    print(f"  prompt tokens: min={min(prompt_tokens)}, max={max(prompt_tokens)}")


if __name__ == "__main__":
    main()
