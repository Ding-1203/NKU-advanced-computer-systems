#!/usr/bin/env python3
"""Replay CoQA pressure traces through GPU/CPU offload policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pandas as pd

from offload_policy import PolicyName, ReplayConfig, replay_pressure_level

DEFAULT_REUSE_SUMMARY = Path("logs/cache_observation/coqa_final/coqa_final_reuse_summary.csv")
DEFAULT_PRESSURE_SUMMARY = Path("logs/cache_observation/coqa_final/coqa_final_pressure_summary.csv")
DEFAULT_OUTPUT_DIR = Path("logs/offload_replay")
DEFAULT_POLICIES: list[PolicyName] = [
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
    parser.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    parser.add_argument("--gpu-token-capacity", type=int, default=ReplayConfig.gpu_token_capacity)
    parser.add_argument("--cpu-token-capacity", type=int, default=ReplayConfig.cpu_token_capacity)
    parser.add_argument("--useful-coverage-threshold", type=float, default=ReplayConfig.useful_coverage_threshold)
    parser.add_argument("--cpu-reload-fixed-ms", type=float, default=ReplayConfig.cpu_reload_fixed_ms)
    parser.add_argument("--cpu-bandwidth-gb-s", type=float, default=ReplayConfig.cpu_bandwidth_gb_s)
    return parser.parse_args()


def load_saved_curve(reuse: pd.DataFrame) -> dict[int, float]:
    data = reuse[(reuse["experiment_type"] == "coqa_prefix_length") & (reuse["expected_cache_state"] == "warm")]
    grouped = data.groupby("prefix_bucket")["ttft_delta_ms"].median().dropna()
    return {int(bucket): max(0.0, float(saved)) for bucket, saved in grouped.items()}


def baselines(reuse: pd.DataFrame, pressure: pd.DataFrame) -> tuple[float, float]:
    warm = reuse[
        (reuse["experiment_type"] == "coqa_prefix_length")
        & (reuse["expected_cache_state"] == "warm")
        & (reuse["prefix_bucket"] == 1536)
    ]["prefix_on_ttft_ms"].median()
    cold = pressure[pressure["phase"].isin(["seed", "filler"])]["ttft_ms"].median()
    return float(cold), float(warm)


def summarize(requests: pd.DataFrame) -> pd.DataFrame:
    revisit = requests[requests["phase"] == "revisit"].copy()
    if revisit.empty:
        return pd.DataFrame()
    keys = ["policy", "pressure_level"]
    revisit_summary = revisit.groupby(keys, as_index=False).agg(
        requests=("request_id", "count"),
        useful_hit_rate=("useful_hit", "mean"),
        median_hit_coverage=("hit_coverage", "median"),
        estimated_ttft_ms=("estimated_ttft_ms", "median"),
        recovered_ttft_saved_ms=("recovered_ttft_saved_ms", "sum"),
        gpu_hit_count=("cache_result", lambda s: int((s == "gpu_hit").sum())),
        cpu_reload_count=("cache_result", lambda s: int((s == "cpu_reload").sum())),
        cold_miss_count=("cache_result", lambda s: int((s == "cold_miss").sum())),
        reload_bytes=("reload_bytes", "sum"),
    )
    all_phase = requests.copy()
    all_phase["eviction_event"] = (all_phase["evicted_utility"] > 0) | (all_phase["offload_bytes"] > 0)
    cost_summary = all_phase.groupby(keys, as_index=False).agg(
        offload_bytes=("offload_bytes", "sum"),
        evicted_utility=("evicted_utility", "sum"),
        damaging_eviction_count=("damaging_eviction", "sum"),
        eviction_event_count=("eviction_event", "sum"),
    )
    summary = revisit_summary.merge(cost_summary, on=keys, how="left")
    denominator = summary["eviction_event_count"].replace(0, pd.NA)
    summary["damaging_eviction_rate"] = (summary["damaging_eviction_count"] / denominator).fillna(0.0)
    return summary.sort_values(["pressure_level", "policy"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reuse = pd.read_csv(args.reuse_summary)
    pressure = pd.read_csv(args.pressure_summary)
    saved_curve = load_saved_curve(reuse)
    cold_baseline, warm_baseline = baselines(reuse, pressure)
    config = ReplayConfig(
        gpu_token_capacity=args.gpu_token_capacity,
        cpu_token_capacity=args.cpu_token_capacity,
        useful_coverage_threshold=args.useful_coverage_threshold,
        cpu_reload_fixed_ms=args.cpu_reload_fixed_ms,
        cpu_bandwidth_gb_s=args.cpu_bandwidth_gb_s,
    )

    rows = []
    for policy_name in args.policies:
        policy = cast(PolicyName, policy_name)
        for level, group in pressure.groupby("pressure_level"):
            for result in replay_pressure_level(group, policy, config, saved_curve, cold_baseline, warm_baseline):
                rows.append(asdict(result))
    requests = pd.DataFrame(rows)
    summary = summarize(requests)
    requests.to_csv(args.output_dir / "offload_replay_requests.csv", index=False)
    summary.to_csv(args.output_dir / "offload_replay_summary.csv", index=False)

    print("saved curve", saved_curve)
    print(f"cold_baseline_ms={cold_baseline:.3f} warm_baseline_ms={warm_baseline:.3f}")
    print(f"config={config}")
    print(summary.round(4).to_string(index=False))
    print(f"wrote {len(requests)} request replay rows to {args.output_dir}")


if __name__ == "__main__":
    main()
