#!/usr/bin/env python3
"""Generate a compact formal-report figure set for prefix-cache observation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SUMMARY = Path("logs/cache_observation/formal/cache_event_summary.csv")
DEFAULT_OUTPUT_DIR = Path("figures/cache_observation/formal_report")
PREFIX_LENGTHS = [128, 256, 512, 1024, 1536]
REUSE_RATIOS = [0.0, 0.25, 0.5, 0.75, 0.9]
COVERAGE_ORDER = ["0%", "0-5%", "5-25%", "25-50%", "50-75%", "75-99%", "100%"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--useful-coverage-threshold",
        type=float,
        default=0.5,
        help="Coverage threshold used for useful-hit rate.",
    )
    return parser.parse_args()


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def load_summary(path: Path, useful_threshold: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    data = pd.read_csv(path)
    for column in [
        "ttft_ms",
        "hit_tokens",
        "actual_prefix_tokens",
        "prefix_tokens",
        "target_prefix_tokens",
        "reuse_ratio_target",
        "pressure_level",
        "block_evict_count_in_window",
    ]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["status", "experiment_type", "phase"]:
        if column not in data:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)
    if "is_warm_request" not in data:
        data["is_warm_request"] = False
    data["is_warm_request"] = to_bool(data["is_warm_request"])
    if "cache_hit" not in data:
        data["cache_hit"] = data["hit_tokens"].fillna(0) > 0
    else:
        data["cache_hit"] = to_bool(data["cache_hit"])

    denominator = data.get("actual_prefix_tokens", pd.Series(index=data.index, dtype=float))
    denominator = denominator.fillna(data.get("prefix_tokens", pd.Series(index=data.index, dtype=float)))
    denominator = denominator.fillna(data.get("target_prefix_tokens", pd.Series(index=data.index, dtype=float)))
    denominator = denominator.replace(0, np.nan)
    data["prefix_hit_coverage"] = (
        data["hit_tokens"].fillna(0) / denominator
    ).clip(lower=0, upper=1).fillna(0)
    data["useful_hit"] = data["prefix_hit_coverage"] >= useful_threshold
    return data[data["status"] == "ok"].copy()


def coverage_label(value: float) -> str:
    if value <= 0:
        return "0%"
    if value < 0.05:
        return "0-5%"
    if value < 0.25:
        return "5-25%"
    if value < 0.50:
        return "25-50%"
    if value < 0.75:
        return "50-75%"
    if value < 0.999:
        return "75-99%"
    return "100%"


def savefig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=180)
    plt.close(fig)


def save_ttft_by_coverage(data: pd.DataFrame, output_dir: Path) -> None:
    plot_data = data.dropna(subset=["ttft_ms"]).copy()
    plot_data["coverage_bucket"] = plot_data["prefix_hit_coverage"].map(coverage_label)
    bucket_values: list[np.ndarray] = []
    bucket_labels: list[str] = []
    for label in COVERAGE_ORDER:
        values = plot_data.loc[plot_data["coverage_bucket"] == label, "ttft_ms"].dropna().to_numpy()
        if len(values) == 0:
            continue
        bucket_values.append(values)
        bucket_labels.append(label)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    positions = np.arange(len(bucket_values))
    box = ax.boxplot(
        bucket_values,
        positions=positions,
        widths=0.58,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#0f172a", "linewidth": 2.0},
        boxprops={"facecolor": "#bfdbfe", "edgecolor": "#2563eb", "linewidth": 1.2},
        whiskerprops={"color": "#2563eb", "linewidth": 1.1},
        capprops={"color": "#2563eb", "linewidth": 1.1},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.68)

    rng = np.random.default_rng(13)
    for index, label in enumerate(bucket_labels):
        values = plot_data.loc[plot_data["coverage_bucket"] == label, "ttft_ms"].dropna().to_numpy()
        jitter = rng.normal(0, 0.055, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=17,
            color="#111827",
            alpha=0.34,
            edgecolors="none",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{label}\nn={len(values)}" for label, values in zip(bucket_labels, bucket_values)])
    ax.set_xlabel("Observed prefix-hit coverage bucket")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT improves only after substantial prefix coverage")
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, output_dir, "01_ttft_by_prefix_coverage")


def save_matrix_coverage_heatmap(data: pd.DataFrame, output_dir: Path) -> None:
    matrix = data[data["experiment_type"] == "prefix_reuse_matrix"].copy()
    values = pd.DataFrame(0.0, index=PREFIX_LENGTHS, columns=REUSE_RATIOS)
    for prefix in PREFIX_LENGTHS:
        for reuse in REUSE_RATIOS:
            if reuse == 0.0:
                values.loc[prefix, reuse] = 0.0
                continue
            cell = matrix[
                (matrix["is_warm_request"])
                & (matrix["target_prefix_tokens"] == prefix)
                & (np.isclose(matrix["reuse_ratio_target"], reuse))
            ]
            if not cell.empty:
                values.loc[prefix, reuse] = float(cell["prefix_hit_coverage"].median())

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    image = ax.imshow(values.to_numpy() * 100.0, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(REUSE_RATIOS)))
    ax.set_xticklabels([f"{int(ratio * 100)}%" for ratio in REUSE_RATIOS])
    ax.set_yticks(np.arange(len(PREFIX_LENGTHS)))
    ax.set_yticklabels([str(prefix) for prefix in PREFIX_LENGTHS])
    ax.set_xlabel("Reuse ratio target")
    ax.set_ylabel("Prefix length target (tokens)")
    ax.set_title("Observer validation: warm requests recover expected prefix")
    for row_index, prefix in enumerate(PREFIX_LENGTHS):
        for col_index, reuse in enumerate(REUSE_RATIOS):
            pct = values.loc[prefix, reuse] * 100.0
            color = "white" if pct >= 60 else "#111827"
            ax.text(col_index, row_index, f"{pct:.0f}%", ha="center", va="center", color=color, fontsize=10)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Median warm prefix-hit coverage")
    savefig(fig, output_dir, "02_matrix_prefix_coverage_heatmap")


def pressure_revisit(data: pd.DataFrame) -> pd.DataFrame:
    pressure = data[(data["experiment_type"] == "cache_pressure") & (data["phase"] == "revisit")].copy()
    if pressure.empty:
        raise RuntimeError("No cache_pressure revisit rows found in the summary.")
    return pressure


def save_pressure_summary(data: pd.DataFrame, output_dir: Path) -> None:
    revisit = pressure_revisit(data)
    grouped = (
        revisit.groupby("pressure_level", as_index=False)
        .agg(
            median_ttft_ms=("ttft_ms", "median"),
            median_coverage=("prefix_hit_coverage", "median"),
            eviction_count=("block_evict_count_in_window", "sum"),
            requests=("request_id", "count"),
        )
        .sort_values("pressure_level")
    )
    x = np.arange(len(grouped))
    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))
    bars = ax1.bar(x, grouped["median_ttft_ms"], width=0.58, color="#bfdbfe", edgecolor="#2563eb", linewidth=1.2)
    ax1.set_ylabel("Revisit median TTFT (ms)")
    ax1.set_xlabel("Prefix group count")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(level)) for level in grouped["pressure_level"]])
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, grouped["median_coverage"] * 100.0, color="#047857", marker="o", linewidth=2.5)
    ax2.set_ylim(-3, 105)
    ax2.set_ylabel("Revisit median prefix-hit coverage")
    ax1.set_title("Capacity pressure collapses useful prefix reuse")
    for bar, evictions in zip(bars, grouped["eviction_count"]):
        ax1.annotate(
            f"evict {int(evictions)}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#b91c1c",
        )
    savefig(fig, output_dir, "03_pressure_revisit_capacity_cliff")


def save_pressure_hit_quality(data: pd.DataFrame, output_dir: Path) -> None:
    revisit = pressure_revisit(data)
    grouped = (
        revisit.groupby("pressure_level", as_index=False)
        .agg(
            any_hit_rate=("cache_hit", "mean"),
            useful_hit_rate=("useful_hit", "mean"),
            median_coverage=("prefix_hit_coverage", "median"),
        )
        .sort_values("pressure_level")
    )
    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(x, grouped["any_hit_rate"] * 100.0, marker="o", linewidth=2.2, color="#b45309", label="Any-hit rate")
    ax.plot(
        x,
        grouped["useful_hit_rate"] * 100.0,
        marker="s",
        linewidth=2.2,
        color="#2563eb",
        label="Useful-hit rate (coverage >= 50%)",
    )
    ax.plot(x, grouped["median_coverage"] * 100.0, marker="^", linewidth=2.2, color="#047857", label="Median coverage")
    ax.set_ylim(-4, 106)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(level)) for level in grouped["pressure_level"]])
    ax.set_xlabel("Prefix group count")
    ax.set_ylabel("Rate / coverage (%)")
    ax.set_title("Binary hit rate masks useful-cache collapse")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left")
    savefig(fig, output_dir, "04_pressure_hit_quality")


def main() -> None:
    args = parse_args()
    data = load_summary(args.summary, args.useful_coverage_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_ttft_by_coverage(data, args.output_dir)
    save_matrix_coverage_heatmap(data, args.output_dir)
    save_pressure_summary(data, args.output_dir)
    save_pressure_hit_quality(data, args.output_dir)
    print(f"Wrote formal report figures to {args.output_dir}")


if __name__ == "__main__":
    main()
