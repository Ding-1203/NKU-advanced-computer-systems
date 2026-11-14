#!/usr/bin/env python3
"""Generate figures for strict-online CARGO replay validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_REPLAY_DIR = Path("logs/offload_online")
DEFAULT_FIGURE_DIR = Path("figures/offload_online")

POLICY_LABELS = {
    "gpu_only": "GPU-only",
    "cpu_lru": "CPU+LRU",
    "cpu_lfu": "CPU+LFU",
    "size_aware": "Size-aware",
    "cargo_utility": "CARGO-online",
    "cargo_coverage_guarded": "CARGO+guard",
    "oracle_upper_bound": "Oracle",
}
POLICY_ORDER = ["gpu_only", "cpu_lru", "cpu_lfu", "size_aware", "cargo_utility", "oracle_upper_bound"]
COLORS = {
    "gpu_only": "#64748b",
    "cpu_lru": "#f97316",
    "cpu_lfu": "#eab308",
    "size_aware": "#14b8a6",
    "cargo_utility": "#2563eb",
    "cargo_coverage_guarded": "#0f766e",
    "oracle_upper_bound": "#7c3aed",
    "best_conventional": "#475569",
}
TRACE_LABELS = {
    "coqa_pressure_no_prior_signal": "Pressure trace",
    "coqa_popularity_skew": "Popularity skew",
    "coqa_mixed_length_interleaved": "Mixed prefix lengths",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def savefig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / name, dpi=190, bbox_inches="tight")
    plt.close(fig)


def add_per_repeat(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()
    data["saved_per_repeat_ms"] = data["recovered_ttft_saved_ms"] / data["repeat_requests"].replace(0, pd.NA)
    return data


def plot_no_future_sanity(summary: pd.DataFrame, out: Path) -> None:
    data = add_per_repeat(summary[summary["trace_name"] == "coqa_pressure_no_prior_signal"])
    keep = ["gpu_only", "cpu_lru", "cargo_utility", "oracle_upper_bound"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for policy in keep:
        rows = data[data["policy"] == policy].sort_values("pressure_level")
        ax.plot(
            rows["pressure_level"],
            rows["saved_per_repeat_ms"],
            marker="o",
            linewidth=2.3,
            color=COLORS[policy],
            label=POLICY_LABELS[policy],
        )
    ax.set_title("No-future sanity: online CARGO stays below the oracle")
    ax.set_xlabel("Seed prefix groups in pressure trace")
    ax.set_ylabel("Recovered TTFT per repeat (ms)")
    ax.set_xticks(sorted(data["pressure_level"].unique()))
    ax.set_ylim(bottom=-1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    savefig(fig, out, "01_no_future_sanity.png")


def plot_policy_gain(summary: pd.DataFrame, out: Path) -> None:
    data = summary[
        summary["trace_name"].isin(["coqa_popularity_skew", "coqa_mixed_length_interleaved"])
        & summary["pressure_level"].isin([24, 32, 48])
        & summary["policy"].isin(POLICY_ORDER)
    ].copy()
    agg = data.groupby(["trace_name", "policy"], as_index=False).agg(saved_ms=("recovered_ttft_saved_ms", "sum"))
    traces = ["coqa_popularity_skew", "coqa_mixed_length_interleaved"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharex=True)
    for ax, trace in zip(axes, traces):
        rows = agg[agg["trace_name"] == trace].set_index("policy").reindex(POLICY_ORDER).dropna().reset_index()
        y = np.arange(len(rows))
        colors = [COLORS[p] for p in rows["policy"]]
        ax.barh(y, rows["saved_ms"] / 1000.0, color=colors, alpha=0.86, edgecolor="#0f172a", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([POLICY_LABELS[p] for p in rows["policy"]])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.24)
        ax.set_title(TRACE_LABELS[trace])
        conventional = rows[rows["policy"].isin(["cpu_lru", "cpu_lfu", "size_aware"])]
        cargo = float(rows.loc[rows["policy"] == "cargo_utility", "saved_ms"].iloc[0])
        best = float(conventional["saved_ms"].max())
        gain = (cargo / best - 1.0) * 100.0 if best else 0.0
        ax.text(0.98, 0.06, f"CARGO vs best conventional: {gain:+.1f}%", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="#0f172a")
    axes[0].set_xlabel("Recovered TTFT saved, g24+g32+g48 (s)")
    axes[1].set_xlabel("Recovered TTFT saved, g24+g32+g48 (s)")
    fig.suptitle("Online CARGO recovers more useful prefix-cache value than conventional policies", y=1.03)
    savefig(fig, out, "02_policy_gain.png")


def plot_capacity_sweep(summary: pd.DataFrame, out: Path) -> None:
    data = add_per_repeat(summary[summary["trace_name"].isin(["coqa_popularity_skew", "coqa_mixed_length_interleaved"])])
    traces = ["coqa_popularity_skew", "coqa_mixed_length_interleaved"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True)
    for ax, trace in zip(axes, traces):
        rows = data[data["trace_name"] == trace]
        conventional = rows[rows["policy"].isin(["cpu_lru", "cpu_lfu", "size_aware"])]
        best = conventional.sort_values("saved_per_repeat_ms").groupby("pressure_level", as_index=False).tail(1).sort_values("pressure_level")
        for policy, label in [("gpu_only", "GPU-only"), ("cargo_utility", "CARGO-online"), ("oracle_upper_bound", "Oracle")]:
            p = rows[rows["policy"] == policy].sort_values("pressure_level")
            ax.plot(p["pressure_level"], p["saved_per_repeat_ms"], marker="o", linewidth=2.2, color=COLORS[policy], label=label)
        ax.plot(best["pressure_level"], best["saved_per_repeat_ms"], marker="s", linewidth=2.0, color=COLORS["best_conventional"], label="Best conventional")
        ax.set_title(TRACE_LABELS[trace])
        ax.set_xlabel("GPU cache capacity (1536-token groups)")
        ax.set_xticks(sorted(rows["pressure_level"].unique()))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Recovered TTFT per repeat (ms)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Capacity sweep with online-only decision state", y=1.03)
    savefig(fig, out, "03_capacity_sweep.png")


def plot_decision_audit(requests: pd.DataFrame, out: Path) -> None:
    data = requests[
        (requests["policy"] == "cargo_utility")
        & requests["trace_name"].isin(["coqa_pressure_no_prior_signal", "coqa_popularity_skew", "coqa_mixed_length_interleaved"])
    ].copy()
    data["prior_bucket"] = pd.cut(
        data["prior_access_count"],
        bins=[-1, 0, 1, 2, 3, 1000],
        labels=["0", "1", "2", "3", "4+"],
    )
    grouped = data.groupby(["prior_bucket", "cache_result"], observed=True).size().unstack(fill_value=0)
    grouped = grouped.reindex(["0", "1", "2", "3", "4+"]).fillna(0)
    for col in ["cold_miss", "cpu_reload", "gpu_hit"]:
        if col not in grouped:
            grouped[col] = 0
    shares = grouped[["cold_miss", "cpu_reload", "gpu_hit"]].div(grouped.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0) * 100.0
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    bottom = np.zeros(len(shares))
    labels = {"cold_miss": "Cold miss", "cpu_reload": "CPU reload", "gpu_hit": "GPU hit"}
    colors = {"cold_miss": "#94a3b8", "cpu_reload": "#f97316", "gpu_hit": "#22c55e"}
    x = np.arange(len(shares))
    for col in ["cold_miss", "cpu_reload", "gpu_hit"]:
        vals = shares[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=colors[col], label=labels[col], edgecolor="#0f172a", linewidth=0.4)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(shares.index.astype(str))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Prior completed accesses for the same prefix")
    ax.set_ylabel("CARGO request outcomes (%)")
    ax.set_title("Decision audit: no cache hit appears before a prefix has been seen")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    savefig(fig, out, "04_decision_audit.png")


def main() -> None:
    args = parse_args()
    summary_path = args.replay_dir / "online_cargo_replay_summary.csv"
    requests_path = args.replay_dir / "online_cargo_replay_requests.csv"
    if not summary_path.exists() or not requests_path.exists():
        raise FileNotFoundError(f"Missing replay CSVs under {args.replay_dir}; run src/run_online_cargo_replay.py first.")
    summary = pd.read_csv(summary_path)
    requests = pd.read_csv(requests_path)
    plot_no_future_sanity(summary, args.figure_dir)
    plot_policy_gain(summary, args.figure_dir)
    plot_capacity_sweep(summary, args.figure_dir)
    plot_decision_audit(requests, args.figure_dir)
    print(summary.round(4).to_string(index=False))
    print(f"wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
