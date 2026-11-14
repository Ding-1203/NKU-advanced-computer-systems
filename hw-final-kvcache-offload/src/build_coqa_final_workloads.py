#!/usr/bin/env python3
"""Build final real-data CoQA workloads for prefix-cache experiments.

All prompt text comes from CoQA. Longer prefixes are formed by packing multiple
CoQA stories or story excerpts into one context, simulating multi-document QA/RAG
while keeping prefix length, popularity, and capacity pressure controllable.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment_utils import TokenCounter, answer_text, read_jsonl, story_id, write_jsonl


DEFAULT_LOCAL_DIR = Path("data/raw/coqa")
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_DIR = Path("data/workloads/coqa_final")
PREFIX_BUCKETS = [512, 768, 1024, 1280, 1536]
PRESSURE_LEVELS = [8, 16, 24, 32, 48]
POPULARITY_TIERS = [
    ("very_hot", 4, 12),
    ("hot", 8, 8),
    ("warm", 16, 4),
    ("cold", 32, 2),
    ("tail", 32, 1),
]


@dataclass(frozen=True)
class ContextPack:
    prefix: str
    actual_prefix_tokens: int
    story_ids: list[str]
    story_count: int


@dataclass(frozen=True)
class GroupSpec:
    experiment_type: str
    scenario_id: str
    scenario_label: str
    prefix_group: str
    prefix_group_index: int
    prefix_bucket: int
    traffic_class: str
    planned_reuse_count: int
    primary_row_index: int
    primary_row: dict[str, Any]
    context: ContextPack


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_rows(local_dir: Path, split: str) -> list[dict[str, Any]]:
    path = local_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing local CoQA export: {path}. Run src/download_coqa.py first.")
    rows = read_jsonl(path)
    usable = []
    for row in rows:
        story = str(row.get("story") or "").strip()
        questions = row.get("questions") or []
        if story and len(questions) >= 2:
            usable.append(row)
    if not usable:
        raise RuntimeError("No usable CoQA rows found.")
    return usable


def normalize_excerpt(text: str) -> str:
    return " ".join(text.strip().split())


def story_text(row: dict[str, Any]) -> str:
    return normalize_excerpt(str(row.get("story") or ""))


def question_text(row: dict[str, Any], turn_index: int) -> str:
    questions = row.get("questions") or []
    if turn_index < len(questions):
        return str(questions[turn_index]).strip()
    return str(questions[-1]).strip()


def render_prefix(primary_id: str, parts: list[tuple[str, str, str]]) -> str:
    docs = []
    for index, (doc_id, role, text) in enumerate(parts, 1):
        docs.append(f"[CoQA story {index:02d} | {role} | id={doc_id}]\n{text}")
    return (
        "System: Answer the question using only the CoQA context pack.\n"
        "The first story is the target story; other CoQA stories are additional retrieved context.\n"
        f"Target story id: {primary_id}\n\n"
        + "\n\n".join(docs)
        + "\n"
    )


def trim_last_part_to_target(
    counter: TokenCounter,
    primary_id: str,
    parts: list[tuple[str, str, str]],
    target_tokens: int,
) -> tuple[str, int, list[tuple[str, str, str]]]:
    last_id, last_role, last_text = parts[-1]
    words = last_text.split()
    if not words:
        prefix = render_prefix(primary_id, parts)
        return prefix, counter.count(prefix), parts

    best_parts = parts[:]
    best_prefix = render_prefix(primary_id, best_parts)
    best_count = counter.count(best_prefix)
    lo, hi = 1, len(words)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_parts = parts[:-1] + [(last_id, last_role, " ".join(words[:mid]))]
        candidate_prefix = render_prefix(primary_id, candidate_parts)
        candidate_count = counter.count(candidate_prefix)
        if candidate_count >= target_tokens:
            best_parts = candidate_parts
            best_prefix = candidate_prefix
            best_count = candidate_count
            hi = mid - 1
        else:
            lo = mid + 1
    return best_prefix, best_count, best_parts


def build_context_pack(
    rows: list[dict[str, Any]],
    counter: TokenCounter,
    primary_index: int,
    distractor_start: int,
    target_tokens: int,
) -> ContextPack:
    primary = rows[primary_index % len(rows)]
    primary_id = story_id(primary, primary_index)
    parts: list[tuple[str, str, str]] = [(primary_id, "target", story_text(primary))]
    prefix = render_prefix(primary_id, parts)
    actual_tokens = counter.count(prefix)

    distractor_cursor = distractor_start
    while actual_tokens < target_tokens:
        row_index = distractor_cursor % len(rows)
        distractor_cursor += 1
        if row_index == primary_index % len(rows):
            continue
        row = rows[row_index]
        doc_id = story_id(row, row_index)
        parts.append((doc_id, "context", story_text(row)))
        prefix = render_prefix(primary_id, parts)
        actual_tokens = counter.count(prefix)

    prefix, actual_tokens, parts = trim_last_part_to_target(counter, primary_id, parts, target_tokens)
    story_ids = [doc_id for doc_id, _, _ in parts]
    return ContextPack(prefix, actual_tokens, story_ids, len(story_ids))


def build_suffix(row: dict[str, Any], turn_index: int) -> str:
    return f"\nQuestion: {question_text(row, turn_index)}\nAnswer in one short sentence:"


def make_group(
    rows: list[dict[str, Any]],
    counter: TokenCounter,
    experiment_type: str,
    scenario_id: str,
    scenario_label: str,
    group_index: int,
    primary_index: int,
    distractor_start: int,
    prefix_bucket: int,
    traffic_class: str,
    planned_reuse_count: int,
) -> GroupSpec:
    context = build_context_pack(rows, counter, primary_index, distractor_start, prefix_bucket)
    return GroupSpec(
        experiment_type=experiment_type,
        scenario_id=scenario_id,
        scenario_label=scenario_label,
        prefix_group=f"{scenario_id}-g{group_index:04d}",
        prefix_group_index=group_index,
        prefix_bucket=prefix_bucket,
        traffic_class=traffic_class,
        planned_reuse_count=planned_reuse_count,
        primary_row_index=primary_index,
        primary_row=rows[primary_index % len(rows)],
        context=context,
    )


def make_request_row(
    counter: TokenCounter,
    workload_name: str,
    group: GroupSpec,
    turn_index: int,
    request_number: int,
    phase: str,
    pressure_level: str | int = "",
    pressure_group_role: str = "",
    pressure_group_id: str = "",
    max_tokens: int = 4,
) -> dict[str, Any]:
    suffix = build_suffix(group.primary_row, turn_index)
    prompt = group.context.prefix + suffix
    suffix_tokens = counter.count(suffix)
    prompt_tokens = counter.count(prompt)
    is_warm = turn_index > 0 or phase == "revisit"
    expected_reused = group.context.actual_prefix_tokens if is_warm else 0
    request_id = f"{workload_name}-{request_number:05d}"
    return {
        "id": request_id,
        "request_id": request_id,
        "scenario_id": group.scenario_id,
        "scenario_label": group.scenario_label,
        "experiment_type": group.experiment_type,
        "target_prefix_tokens": group.prefix_bucket,
        "actual_prefix_tokens": group.context.actual_prefix_tokens,
        "prefix_target_tokens": group.prefix_bucket,
        "prefix_tokens": group.context.actual_prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "prompt_tokens": prompt_tokens,
        "reuse_ratio_target": "",
        "reuse_ratio_actual": "",
        "num_prefix_groups": "",
        "total_requests": "",
        "requests_per_group": "",
        "request_index_in_group": turn_index,
        "occurrence_index": turn_index,
        "is_warm_request": is_warm,
        "expected_cache_state": "warm" if is_warm else "cold",
        "expected_reused_prefix_tokens": expected_reused,
        "expected_reuse_ratio_token": expected_reused / prompt_tokens if prompt_tokens else 0.0,
        "prefix_group": group.prefix_group,
        "prefix_group_index": group.prefix_group_index,
        "workload_name": workload_name,
        "phase": phase,
        "pressure_level": pressure_level,
        "pressure_group_role": pressure_group_role,
        "pressure_group_id": pressure_group_id,
        "sequence_pass": turn_index,
        "prefix_bucket": group.prefix_bucket,
        "traffic_class": group.traffic_class,
        "coqa_story_id": story_id(group.primary_row, group.primary_row_index),
        "coqa_turn_id": turn_index,
        "context_pack_size": group.context.story_count,
        "context_pack_story_ids": ";".join(group.context.story_ids),
        "planned_reuse_count": group.planned_reuse_count,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "coqa_answer": answer_text(group.primary_row, turn_index),
        "prompt": prompt,
    }


def finalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_counts = Counter(row["prefix_group"] for row in rows)
    num_groups = len(group_counts)
    total = len(rows)
    reuse_ratio = 1.0 - num_groups / total if total else 0.0
    for order, row in enumerate(rows):
        row["request_order"] = order
        row["num_prefix_groups"] = num_groups
        row["total_requests"] = total
        row["requests_per_group"] = group_counts[row["prefix_group"]]
        row["reuse_ratio_actual"] = reuse_ratio
    return rows


def build_prefix_length_rows(
    rows: list[dict[str, Any]],
    counter: TokenCounter,
    args: argparse.Namespace,
    start_primary: int,
    start_request: int,
) -> tuple[list[dict[str, Any]], int, int]:
    groups: list[GroupSpec] = []
    primary = start_primary
    for bucket in args.prefix_buckets:
        scenario_id = f"coqa_len_{bucket}"
        for _ in range(args.length_groups_per_bucket):
            group_index = len(groups)
            groups.append(
                make_group(
                    rows, counter, "coqa_prefix_length", scenario_id,
                    f"CoQA context prefix {bucket} tokens", group_index, primary,
                    primary + 1000, bucket, "", args.length_turns_per_group - 1,
                )
            )
            primary += 1

    output: list[dict[str, Any]] = []
    request_number = start_request
    for turn in range(args.length_turns_per_group):
        for group in groups:
            phase = "seed" if turn == 0 else "warm"
            output.append(make_request_row(counter, args.reuse_workload_name, group, turn, request_number, phase, max_tokens=args.max_tokens))
            request_number += 1
    return output, primary, request_number


def build_popularity_rows(
    rows: list[dict[str, Any]],
    counter: TokenCounter,
    args: argparse.Namespace,
    start_primary: int,
    start_request: int,
) -> tuple[list[dict[str, Any]], int, int]:
    groups: list[GroupSpec] = []
    primary = start_primary
    for tier_name, tier_groups, revisit_count in POPULARITY_TIERS:
        for _ in range(tier_groups):
            group_index = len(groups)
            groups.append(
                make_group(
                    rows, counter, "coqa_popularity_skew", f"coqa_popularity_{tier_name}",
                    f"CoQA popularity tier {tier_name}", group_index, primary,
                    primary + 1600, args.popularity_prefix_tokens, tier_name, revisit_count,
                )
            )
            primary += 1

    output: list[dict[str, Any]] = []
    request_number = start_request
    for group in groups:
        output.append(make_request_row(counter, args.reuse_workload_name, group, 0, request_number, "seed", max_tokens=args.max_tokens))
        request_number += 1

    max_revisit = max(revisit_count for _, _, revisit_count in POPULARITY_TIERS)
    for turn in range(1, max_revisit + 1):
        for group in groups:
            if turn <= group.planned_reuse_count:
                output.append(make_request_row(counter, args.reuse_workload_name, group, turn, request_number, "revisit", max_tokens=args.max_tokens))
                request_number += 1
    return output, primary, request_number


def build_reuse_workload(rows: list[dict[str, Any]], counter: TokenCounter, args: argparse.Namespace) -> list[dict[str, Any]]:
    primary = args.primary_offset
    request_number = 0
    length_rows, primary, request_number = build_prefix_length_rows(rows, counter, args, primary, request_number)
    popularity_rows, primary, request_number = build_popularity_rows(rows, counter, args, primary, request_number)
    return finalize_rows([*length_rows, *popularity_rows])


def build_pressure_workload_for_level(
    rows: list[dict[str, Any]],
    counter: TokenCounter,
    args: argparse.Namespace,
    level: int,
    primary_start: int,
) -> list[dict[str, Any]]:
    seed_groups: list[GroupSpec] = []
    filler_groups: list[GroupSpec] = []
    primary = primary_start
    scenario_id = f"coqa_pressure_{level}"
    for local_group in range(level):
        seed_groups.append(
            make_group(
                rows, counter, "coqa_capacity_pressure", scenario_id,
                f"CoQA pressure seed groups={level}", local_group, primary,
                primary + 2200, args.pressure_prefix_tokens, "seed", 1,
            )
        )
        primary += 1
    for local_group in range(level * args.pressure_filler_multiplier):
        group_index = level + local_group
        filler_groups.append(
            make_group(
                rows, counter, "coqa_capacity_pressure", scenario_id,
                f"CoQA pressure seed groups={level}", group_index, primary,
                primary + 2600, args.pressure_prefix_tokens, "filler", 0,
            )
        )
        primary += 1

    output: list[dict[str, Any]] = []
    request_number = 0
    workload_name = f"{args.pressure_workload_name}_g{level}"
    for group in seed_groups:
        output.append(make_request_row(counter, workload_name, group, 0, request_number, "seed", level, "seed", group.prefix_group, args.max_tokens))
        request_number += 1
    for group in filler_groups:
        output.append(make_request_row(counter, workload_name, group, 0, request_number, "filler", level, "filler", group.prefix_group, args.max_tokens))
        request_number += 1
    for group in seed_groups:
        output.append(make_request_row(counter, workload_name, group, 1, request_number, "revisit", level, "seed", group.prefix_group, args.max_tokens))
        request_number += 1
    return finalize_rows(output)


def validate_rows(rows: list[dict[str, Any]], max_prompt_tokens: int) -> None:
    too_long = [row for row in rows if int(row["prompt_tokens"]) > max_prompt_tokens]
    if too_long:
        longest = max(too_long, key=lambda row: int(row["prompt_tokens"]))
        raise RuntimeError(
            f"{len(too_long)} prompts exceed {max_prompt_tokens}; longest "
            f"{longest['request_id']} has {longest['prompt_tokens']} tokens"
        )


def print_summary(name: str, output: Path, rows: list[dict[str, Any]]) -> None:
    states = Counter(row["expected_cache_state"] for row in rows)
    experiments = Counter(row["experiment_type"] for row in rows)
    phases = Counter(row["phase"] for row in rows)
    prompt_tokens = [int(row["prompt_tokens"]) for row in rows]
    prefix_tokens = [int(row["actual_prefix_tokens"]) for row in rows]
    print(f"Wrote {len(rows)} requests to {output}")
    print(f"  workload: {name}")
    print("  experiments: " + ", ".join(f"{key}={value}" for key, value in sorted(experiments.items())))
    print("  phases: " + ", ".join(f"{key}={value}" for key, value in sorted(phases.items())))
    print("  expected states: " + ", ".join(f"{key}={value}" for key, value in sorted(states.items())))
    print(f"  prefix tokens: min={min(prefix_tokens)}, max={max(prefix_tokens)}")
    print(f"  prompt tokens: min={min(prompt_tokens)}, max={max(prompt_tokens)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reuse-workload-name", default="coqa_final_reuse")
    parser.add_argument("--pressure-workload-name", default="coqa_final_pressure")
    parser.add_argument("--prefix-buckets", type=parse_int_list, default=PREFIX_BUCKETS)
    parser.add_argument("--length-groups-per-bucket", type=int, default=40)
    parser.add_argument("--length-turns-per-group", type=int, default=4)
    parser.add_argument("--popularity-prefix-tokens", type=int, default=1536)
    parser.add_argument("--pressure-levels", type=parse_int_list, default=PRESSURE_LEVELS)
    parser.add_argument("--pressure-prefix-tokens", type=int, default=1536)
    parser.add_argument("--pressure-filler-multiplier", type=int, default=4)
    parser.add_argument("--primary-offset", type=int, default=0)
    parser.add_argument("--pressure-primary-offset", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1900)
    parser.add_argument("--model-path", default=os.environ.get("QWEN05B_PATH"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counter = TokenCounter(args.model_path)
    coqa_rows = load_rows(args.local_dir, args.split)

    reuse_rows = build_reuse_workload(coqa_rows, counter, args)
    validate_rows(reuse_rows, args.max_prompt_tokens)
    reuse_output = args.output_dir / f"{args.reuse_workload_name}.jsonl"
    write_jsonl(reuse_output, reuse_rows)
    print_summary(args.reuse_workload_name, reuse_output, reuse_rows)

    for offset, level in enumerate(args.pressure_levels):
        pressure_rows = build_pressure_workload_for_level(
            coqa_rows,
            counter,
            args,
            level,
            args.pressure_primary_offset + offset * 500,
        )
        validate_rows(pressure_rows, args.max_prompt_tokens)
        pressure_output = args.output_dir / f"{args.pressure_workload_name}_g{level}.jsonl"
        write_jsonl(pressure_output, pressure_rows)
        print_summary(f"{args.pressure_workload_name}_g{level}", pressure_output, pressure_rows)


if __name__ == "__main__":
    main()
