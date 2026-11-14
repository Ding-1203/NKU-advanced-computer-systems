#!/usr/bin/env python3
"""Run strict-online CARGO replay on CoQA trace summaries."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pandas as pd

from offload_policy import PolicyName, ReplayConfig, replay_pressure_level
from offload_replay_simulator import baselines, load_saved_curve

DEFAULT_REUSE_SUMMARY = Path("logs/cache_observation/coqa_final/coqa_final_reuse_summary.csv")
DEFAULT_PRESSURE_SUMMARY = Path("logs/cache_observation/coqa_final/coqa_final_pressure_summary.csv")
DEFAULT_OUTPUT_DIR = Path("logs/offload_online")
POLICIES: list[PolicyName] = [
    "gpu_only",
    "cpu_lru",
    "cpu_lfu",
    "size_aware",
    "cargo_utility",
    "cargo_coverage_guarded",
    "oracle_upper_bound",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-summary", type=Path, default=DEFAULT_REUSE_SUMMARY)
    parser.add_argument("--pressure-summary", type=Path, default=DEFAULT_PRESSURE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--capacity-groups", type=int, nargs="+", default=[16, 24, 32, 48])
    parser.add_argument("--tokens-per-group", type=int, default=1536)
    parser.add_argument("--cpu-capacity-ratio", type=float, default=1.0)
    parser.add_argument("--policies", nargs="*", default=POLICIES)
    parser.add_argument("--cpu-reload-fixed-ms", type=float, default=ReplayConfig.cpu_reload_fixed_ms)
    parser.add_argument("--cpu-bandwidth-gb-s", type=float, default=ReplayConfig.cpu_bandwidth_gb_s)
    return parser.parse_args()


def prepare_popularity_trace(reuse: pd.DataFrame) -> pd.DataFrame:
    trace = reuse[reuse["experiment_type"] == "coqa_popularity_skew"].copy()
    trace = trace.sort_values(["request_index", "request_id"]).reset_index(drop=True)
    trace["trace_name"] = "coqa_popularity_skew"
    return trace


def prepare_mixed_length_trace(reuse: pd.DataFrame) -> pd.DataFrame:
    trace = reuse[reuse["experiment_type"] == "coqa_prefix_length"].copy()
    trace["group_index_within_bucket"] = trace["prefix_group_index"].astype(int) % 40
    trace = trace.sort_values(["occurrence_index", "group_index_within_bucket", "prefix_bucket", "request_id"]).reset_index(drop=True)
    trace["request_index"] = range(len(trace))
    trace["trace_name"] = "coqa_mixed_length_interleaved"
    return trace.drop(columns=["group_index_within_bucket"])


def prepare_pressure_trace(pressure: pd.DataFrame) -> pd.DataFrame:
    trace = pressure[pressure["pressure_level"].isin([16, 24, 32, 48])].copy()
    trace = trace.sort_values(["pressure_level", "request_index", "request_id"]).reset_index(drop=True)
    trace["trace_name"] = "coqa_pressure_no_prior_signal"
    return trace


def summarize_online(requests: pd.DataFrame) -> pd.DataFrame:
    repeats = requests[requests["is_repeat_online"]].copy()
    if repeats.empty:
        return pd.DataFrame()
    keys = ["trace_name", "policy", "pressure_level"]
    repeat_summary = repeats.groupby(keys, as_index=False).agg(
        repeat_requests=("request_id", "count"),
        useful_hit_rate=("useful_hit", "mean"),
        median_estimated_ttft_ms=("estimated_ttft_ms", "median"),
        recovered_ttft_saved_ms=("recovered_ttft_saved_ms", "sum"),
        gpu_hit_count=("cache_result", lambda s: int((s == "gpu_hit").sum())),
        cpu_reload_count=("cache_result", lambda s: int((s == "cpu_reload").sum())),
        cold_miss_count=("cache_result", lambda s: int((s == "cold_miss").sum())),
        reload_bytes=("reload_bytes", "sum"),
    )
    all_summary = requests.groupby(keys, as_index=False).agg(
        requests=("request_id", "count"),
        offload_bytes=("offload_bytes", "sum"),
        damaging_eviction_count=("damaging_eviction", "sum"),
        evicted_utility=("evicted_utility", "sum"),
    )
    summary = repeat_summary.merge(all_summary, on=keys, how="left")
    total_hits = summary["gpu_hit_count"] + summary["cpu_reload_count"]
    summary["cpu_reload_share_of_hits"] = (summary["cpu_reload_count"] / total_hits.replace(0, pd.NA)).fillna(0.0)
    return summary.sort_values(["trace_name", "pressure_level", "policy"])


def run_trace(trace: pd.DataFrame, trace_name: str, capacity_groups: int, args: argparse.Namespace, base_config: ReplayConfig, saved_curve: dict[int, float], cold: float, warm: float, policies: list[PolicyName]) -> list[dict[str, object]]:
    gpu_tokens = capacity_groups * args.tokens_per_group
    cpu_tokens = int(gpu_tokens * args.cpu_capacity_ratio)
    config = replace(base_config, gpu_token_capacity=gpu_tokens, cpu_token_capacity=cpu_tokens)
    rows = trace.copy()
    rows["trace_name"] = trace_name
    rows["pressure_level"] = capacity_groups
    out: list[dict[str, object]] = []
    for policy in policies:
        for result in replay_pressure_level(rows, policy, config, saved_curve, cold, warm):
            out.append(asdict(result))
    return out


def run_pressure_sanity(trace: pd.DataFrame, args: argparse.Namespace, base_config: ReplayConfig, saved_curve: dict[int, float], cold: float, warm: float, policies: list[PolicyName]) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for pressure_level, group in trace.groupby("pressure_level"):
        capacity_groups = 90
        gpu_tokens = capacity_groups * args.tokens_per_group
        config = replace(base_config, gpu_token_capacity=gpu_tokens, cpu_token_capacity=int(gpu_tokens * args.cpu_capacity_ratio))
        group = group.copy()
        group["trace_name"] = "coqa_pressure_no_prior_signal"
        for policy in policies:
            for result in replay_pressure_level(group, policy, config, saved_curve, cold, warm):
                rows_out.append(asdict(result))
    return rows_out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reuse = pd.read_csv(args.reuse_summary)
    pressure = pd.read_csv(args.pressure_summary)
    saved_curve = load_saved_curve(reuse)
    cold, warm = baselines(reuse, pressure)
    base_config = ReplayConfig(
        cpu_reload_fixed_ms=args.cpu_reload_fixed_ms,
        cpu_bandwidth_gb_s=args.cpu_bandwidth_gb_s,
    )
    policies = [cast(PolicyName, p) for p in args.policies]

    rows: list[dict[str, object]] = []
    rows.extend(run_pressure_sanity(prepare_pressure_trace(pressure), args, base_config, saved_curve, cold, warm, policies))
    for capacity_groups in args.capacity_groups:
        rows.extend(run_trace(prepare_popularity_trace(reuse), "coqa_popularity_skew", capacity_groups, args, base_config, saved_curve, cold, warm, policies))
        rows.extend(run_trace(prepare_mixed_length_trace(reuse), "coqa_mixed_length_interleaved", capacity_groups, args, base_config, saved_curve, cold, warm, policies))

    requests = pd.DataFrame(rows)
    summary = summarize_online(requests)
    requests.to_csv(args.output_dir / "online_cargo_replay_requests.csv", index=False)
    summary.to_csv(args.output_dir / "online_cargo_replay_summary.csv", index=False)
    print(f"saved_curve={saved_curve}")
    print(f"cold_baseline_ms={cold:.3f} warm_baseline_ms={warm:.3f}")
    print(summary.round(4).to_string(index=False))
    print(f"wrote {len(requests)} request replay rows to {args.output_dir}")


if __name__ == "__main__":
    main()
