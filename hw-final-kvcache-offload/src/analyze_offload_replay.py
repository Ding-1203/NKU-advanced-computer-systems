#!/usr/bin/env python3
"""Analyze trace-based offload replay results and generate report figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_REPLAY_DIR = Path("logs/offload_replay")
DEFAULT_FIGURE_DIR = Path("figures/offload_replay")
POLICY_LABELS = {
    "gpu_only": "GPU-only",
    "cpu_lru": "CPU+LRU",
    "cpu_lfu": "CPU+LFU",
    "size_aware": "Size-aware",
    "cargo_utility": "CARGO-H",
    "cargo_coverage_guarded": "CARGO-H+Guard",
    "oracle_upper_bound": "Oracle",
}
POLICY_ORDER = ["gpu_only", "cpu_lru", "cpu_lfu", "size_aware", "cargo_utility", "cargo_coverage_guarded", "oracle_upper_bound"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def savefig(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def ordered_policies(data: pd.DataFrame) -> list[str]:
    present = set(data["policy"])
    return [policy for policy in POLICY_ORDER if policy in present]


def plot_pressure_cliff(summary: pd.DataFrame, output_dir: Path) -> None:
    policies = [p for p in ["gpu_only", "cpu_lru", "cargo_utility", "cargo_coverage_guarded", "oracle_upper_bound"] if p in set(summary["policy"])]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.4, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.05, 0.95]})
    colors = {
        "gpu_only": "#64748b",
        "cpu_lru": "#f97316",
        "cargo_utility": "#2563eb",
        "cargo_coverage_guarded": "#0f766e",
        "oracle_upper_bound": "#7c3aed",
    }
    for policy in policies:
        data = summary[summary["policy"] == policy].sort_values("pressure_level")
        label = POLICY_LABELS.get(policy, policy)
        ax1.plot(data["pressure_level"], data["estimated_ttft_ms"], marker="o", linewidth=2.2, color=colors.get(policy), label=label)
        ax2.plot(data["pressure_level"], data["useful_hit_rate"] * 100.0, marker="o", linewidth=2.2, color=colors.get(policy), label=label)
    ax1.set_ylabel("Estimated revisit TTFT (ms)")
    ax1.set_title("Replay: CPU offload converts destructive eviction into recoverable reuse")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, ncols=3, loc="upper center")
    ax2.set_xlabel("Seed prefix groups before filler")
    ax2.set_ylabel("Useful-hit rate (%)")
    ax2.set_ylim(-4, 104)
    ax2.grid(axis="y", alpha=0.25)
    savefig(fig, output_dir, "01_replay_pressure_cliff.png")


def plot_policy_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary[summary["pressure_level"] >= 24].copy()
    policies = ordered_policies(data)
    grouped = data.groupby("policy").agg(
        recovered_ttft_saved_ms=("recovered_ttft_saved_ms", "sum"),
        useful_hit_rate=("useful_hit_rate", "mean"),
        damaging_eviction_rate=("damaging_eviction_rate", "mean"),
    ).reindex(policies)
    x = np.arange(len(grouped))
    fig, ax1 = plt.subplots(figsize=(9.2, 5.4))
    bars = ax1.bar(x, grouped["recovered_ttft_saved_ms"], color="#bfdbfe", edgecolor="#2563eb", linewidth=1.1)
    ax1.set_ylabel("Recovered TTFT saved at g24/g32/g48 (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([POLICY_LABELS.get(policy, policy) for policy in grouped.index], rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, grouped["damaging_eviction_rate"] * 100.0, marker="o", color="#dc2626", linewidth=2.2, label="Damaging eviction rate")
    ax2.set_ylabel("Damaging eviction rate (%)")
    ax2.set_ylim(0, max(5, grouped["damaging_eviction_rate"].max() * 120))
    for bar, rate in zip(bars, grouped["useful_hit_rate"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, f"useful {rate * 100:.0f}%", ha="center", va="bottom", fontsize=9)
    ax1.set_title("Historical CARGO is separated from the offline oracle upper bound")
    ax2.legend(frameon=False, loc="upper right")
    savefig(fig, output_dir, "02_policy_recovered_utility.png")


def plot_cost_breakdown(requests: pd.DataFrame, output_dir: Path) -> None:
    data = requests[(requests["phase"] == "revisit") & (requests["policy"].isin(["gpu_only", "cpu_lru", "cargo_coverage_guarded"]))].copy()
    if data.empty:
        return
    grouped = data.groupby(["policy", "cache_result"], as_index=False).agg(
        median_ttft_ms=("estimated_ttft_ms", "median"),
        requests=("request_id", "count"),
    )
    cache_order = ["gpu_hit", "cpu_reload", "cold_miss"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.5), sharey=True)
    colors = {"gpu_hit": "#22c55e", "cpu_reload": "#f97316", "cold_miss": "#94a3b8"}
    for ax, policy in zip(axes, ["gpu_only", "cpu_lru", "cargo_coverage_guarded"]):
        pdata = grouped[grouped["policy"] == policy].set_index("cache_result")
        heights = [float(pdata.loc[result, "median_ttft_ms"]) if result in pdata.index else 0.0 for result in cache_order]
        counts = [int(pdata.loc[result, "requests"]) if result in pdata.index else 0 for result in cache_order]
        x = np.arange(len(cache_order))
        ax.bar(x, heights, color=[colors[result] for result in cache_order], edgecolor="#0f172a", linewidth=0.6, alpha=0.78)
        ax.set_xticks(x)
        ax.set_xticklabels(["GPU hit", "CPU reload", "Cold miss"], rotation=25, ha="right")
        ax.set_title(POLICY_LABELS.get(policy, policy))
        ax.grid(axis="y", alpha=0.22)
        for xpos, height, count in zip(x, heights, counts):
            if count:
                ax.text(xpos, height + 1.5, f"n={count}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Median estimated TTFT (ms)")
    fig.suptitle("Replay cost breakdown: GPU hit < CPU reload < cold miss", y=1.02)
    savefig(fig, output_dir, "03_replay_cost_breakdown.png")


def main() -> None:
    args = parse_args()
    summary_path = args.replay_dir / "offload_replay_summary.csv"
    requests_path = args.replay_dir / "offload_replay_requests.csv"
    if not summary_path.exists() or not requests_path.exists():
        raise FileNotFoundError(f"Missing replay outputs under {args.replay_dir}; run src/offload_replay_simulator.py first.")
    summary = pd.read_csv(summary_path)
    requests = pd.read_csv(requests_path)
    plot_pressure_cliff(summary, args.figure_dir)
    plot_policy_comparison(summary, args.figure_dir)
    plot_cost_breakdown(requests, args.figure_dir)
    print(summary.round(4).to_string(index=False))
    print(f"wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
