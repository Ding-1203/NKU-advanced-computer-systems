#!/usr/bin/env python3
"""Plot quantitative vLLM prefix-cache baseline CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_LOG_DIR = Path("logs/baseline_v1_1")
BASELINE_FIGURE_DIR = Path("figures/baseline_v1_1")
OFF_COLOR = "#6b7280"
ON_COLOR = "#2563eb"
LINE_COLOR = "#b45309"


def existing_pair(off_path: Path, on_path: Path) -> bool:
    return off_path.exists() and on_path.exists()


def load_pair(off_path: Path, on_path: Path) -> pd.DataFrame:
    frames = []
    for config, path in [("prefix_off", off_path), ("prefix_on", on_path)]:
        frame = pd.read_csv(path)
        frame["config"] = config
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"] == "ok"].copy()
    numeric_columns = [
        "ttft_ms",
        "total_latency_ms",
        "target_prefix_tokens",
        "actual_prefix_tokens",
        "reuse_ratio_target",
        "reuse_ratio_actual",
        "num_prefix_groups",
        "request_index_in_group",
        "expected_reused_prefix_tokens",
        "expected_reuse_ratio_token",
    ]
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "is_warm_request" in df:
        df["is_warm_request"] = df["is_warm_request"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    elif "expected_cache_state" in df:
        df["is_warm_request"] = df["expected_cache_state"].eq("warm")
    else:
        df["is_warm_request"] = False
    return df.dropna(subset=["ttft_ms"])


def load_available_pairs(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    configured = {
        "prefix_length_sweep": (args.prefix_length_off, args.prefix_length_on),
        "reuse_ratio_sweep": (args.reuse_ratio_off, args.reuse_ratio_on),
        "prefix_reuse_matrix": (args.matrix_off, args.matrix_on),
    }
    loaded: dict[str, pd.DataFrame] = {}
    for experiment_type, (off_path, on_path) in configured.items():
        if existing_pair(off_path, on_path):
            df = load_pair(off_path, on_path)
            if "experiment_type" not in df:
                df["experiment_type"] = experiment_type
            loaded[experiment_type] = df
        elif not args.allow_missing:
            raise FileNotFoundError(f"Missing input pair: {off_path}, {on_path}")
    return loaded


def warm_or_cold_only(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, scenario_df in df.groupby("scenario_id", dropna=False):
        warm = scenario_df[scenario_df["is_warm_request"]]
        parts.append(warm if not warm.empty else scenario_df)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def summarize(
    df: pd.DataFrame,
    group_columns: list[str],
    warm_only: bool = True,
) -> pd.DataFrame:
    source = warm_or_cold_only(df) if warm_only else df
    summary = (
        source.groupby(group_columns + ["config"], as_index=False)
        .agg(
            median_ttft_ms=("ttft_ms", "median"),
            p25_ttft_ms=("ttft_ms", lambda values: values.quantile(0.25)),
            p75_ttft_ms=("ttft_ms", lambda values: values.quantile(0.75)),
            median_total_latency_ms=("total_latency_ms", "median"),
            requests=("request_id", "count"),
        )
        .sort_values(group_columns + ["config"])
    )
    return summary


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=180)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, bars: Iterable[plt.Rectangle]) -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_combo(
    summary: pd.DataFrame,
    x_column: str,
    x_labels: list[str],
    title: str,
    output_dir: Path,
    stem: str,
) -> None:
    off = summary[summary["config"] == "prefix_off"].set_index(x_column)
    on = summary[summary["config"] == "prefix_on"].set_index(x_column)
    x_values = list(dict.fromkeys(summary[x_column].tolist()))
    x = np.arange(len(x_values))
    width = 0.34

    off_median = off.reindex(x_values)["median_ttft_ms"].to_numpy()
    on_median = on.reindex(x_values)["median_ttft_ms"].to_numpy()
    off_p25 = off.reindex(x_values)["p25_ttft_ms"].to_numpy()
    off_p75 = off.reindex(x_values)["p75_ttft_ms"].to_numpy()
    on_p25 = on.reindex(x_values)["p25_ttft_ms"].to_numpy()
    on_p75 = on.reindex(x_values)["p75_ttft_ms"].to_numpy()
    speedup = off_median / on_median

    fig, ax1 = plt.subplots(figsize=(12, 6))
    off_errors = np.vstack([off_median - off_p25, off_p75 - off_median])
    on_errors = np.vstack([on_median - on_p25, on_p75 - on_median])
    bars_off = ax1.bar(
        x - width / 2,
        off_median,
        width,
        yerr=off_errors,
        capsize=4,
        label="Prefix Cache OFF",
        color=OFF_COLOR,
        alpha=0.86,
    )
    bars_on = ax1.bar(
        x + width / 2,
        on_median,
        width,
        yerr=on_errors,
        capsize=4,
        label="Prefix Cache ON",
        color=ON_COLOR,
        alpha=0.86,
    )
    ax1.set_ylabel("Median TTFT (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_title(title, fontsize=14, pad=12)
    add_bar_labels(ax1, bars_off)
    add_bar_labels(ax1, bars_on)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        speedup,
        color=LINE_COLOR,
        marker="o",
        linewidth=2,
        label="TTFT speedup (OFF / ON)",
    )
    for index, value in enumerate(speedup):
        ax2.annotate(
            f"{value:.2f}x",
            xy=(x[index], value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=LINE_COLOR,
            fontsize=10,
        )
    ax2.set_ylabel("TTFT speedup (OFF / ON)")
    ax2.axhline(1.0, color="#111827", linewidth=1, alpha=0.4)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    save_figure(fig, output_dir, stem)


def plot_prefix_length_sweep(df: pd.DataFrame, output_dir: Path) -> None:
    summary = summarize(
        df,
        ["target_prefix_tokens", "reuse_ratio_target", "scenario_id", "scenario_label"],
        warm_only=True,
    )
    summary = summary.sort_values(["target_prefix_tokens", "config"])
    summary.to_csv(output_dir / "prefix_length_sweep_summary.csv", index=False)
    x_values = sorted(summary["target_prefix_tokens"].dropna().unique())
    labels = [f"{int(value)} tok" for value in x_values]
    plot_combo(
        summary=summary,
        x_column="target_prefix_tokens",
        x_labels=labels,
        title="TTFT benefit increases with prefix length",
        output_dir=output_dir,
        stem="prefix_length_sweep_combo",
    )


def plot_reuse_ratio_sweep(df: pd.DataFrame, output_dir: Path) -> None:
    summary = summarize(
        df,
        ["reuse_ratio_target", "target_prefix_tokens", "scenario_id", "scenario_label"],
        warm_only=True,
    )
    summary = summary.sort_values(["reuse_ratio_target", "config"])
    summary.to_csv(output_dir / "reuse_ratio_sweep_summary.csv", index=False)
    x_values = sorted(summary["reuse_ratio_target"].dropna().unique())
    labels = [f"{int(round(value * 100))}%" for value in x_values]
    plot_combo(
        summary=summary,
        x_column="reuse_ratio_target",
        x_labels=labels,
        title="Prefix cache speedup grows with reuse ratio",
        output_dir=output_dir,
        stem="reuse_ratio_sweep_combo",
    )


def plot_prefix_reuse_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    summary = summarize(
        df,
        ["target_prefix_tokens", "reuse_ratio_target", "scenario_id", "scenario_label"],
        warm_only=True,
    )
    pivot = summary.pivot_table(
        index=["target_prefix_tokens", "reuse_ratio_target"],
        columns="config",
        values="median_ttft_ms",
        aggfunc="median",
    ).reset_index()
    pivot["speedup"] = pivot["prefix_off"] / pivot["prefix_on"]
    pivot.to_csv(output_dir / "prefix_reuse_matrix_speedup.csv", index=False)

    lengths = sorted(pivot["target_prefix_tokens"].dropna().unique())
    ratios = sorted(pivot["reuse_ratio_target"].dropna().unique())
    matrix = np.full((len(lengths), len(ratios)), np.nan)
    for row_index, length in enumerate(lengths):
        for col_index, ratio in enumerate(ratios):
            matched = pivot[
                (pivot["target_prefix_tokens"] == length)
                & (pivot["reuse_ratio_target"] == ratio)
            ]
            if not matched.empty:
                matrix[row_index, col_index] = matched.iloc[0]["speedup"]

    min_speedup = float(np.nanmin(matrix))
    max_speedup = float(np.nanmax(matrix))
    color_min = min(1.0, min_speedup)
    color_max = float(np.nanpercentile(matrix, 95))
    if color_max <= color_min:
        color_max = max_speedup
    if color_max <= color_min:
        color_max = color_min + 0.01
    colorbar_extend = "max" if max_speedup > color_max else "neither"

    fig, ax = plt.subplots(figsize=(10, 7.5))
    image = ax.imshow(
        matrix,
        cmap="Blues",
        aspect="auto",
        vmin=color_min,
        vmax=color_max,
    )
    ax.set_xticks(np.arange(len(ratios)))
    ax.set_yticks(np.arange(len(lengths)))
    ax.set_xticklabels([f"{int(round(ratio * 100))}%" for ratio in ratios])
    ax.set_yticklabels([f"{int(length)} tok" for length in lengths])
    ax.set_xlabel("Request-level reuse ratio")
    ax.set_ylabel("Target prefix length")
    ax.set_title("Prefix length and reuse ratio jointly determine TTFT speedup", pad=12)

    midpoint = color_min + ((color_max - color_min) * 0.58)
    for row_index in range(len(lengths)):
        for col_index in range(len(ratios)):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                label = "NA"
                text_color = "#111827"
            else:
                label = f"{value:.2f}x"
                text_color = "white" if value >= midpoint else "#111827"
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    colorbar = fig.colorbar(image, ax=ax, extend=colorbar_extend)
    colorbar.set_label("TTFT speedup (OFF / ON)")
    colorbar.set_ticks(np.linspace(color_min, color_max, 6))
    colorbar.ax.set_yticklabels([f"{value:.2f}x" for value in np.linspace(color_min, color_max, 6)])
    fig.tight_layout()
    save_figure(fig, output_dir, "prefix_reuse_speedup_heatmap")


def plot_cold_warm_breakdown(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    combined = pd.concat(frames.values(), ignore_index=True)
    combined = combined[combined["status"] == "ok"].copy()
    combined["cache_state"] = np.where(combined["is_warm_request"], "warm", "cold")
    summary = (
        combined.groupby(["cache_state", "config"], as_index=False)
        .agg(
            median_ttft_ms=("ttft_ms", "median"),
            p25_ttft_ms=("ttft_ms", lambda values: values.quantile(0.25)),
            p75_ttft_ms=("ttft_ms", lambda values: values.quantile(0.75)),
            requests=("request_id", "count"),
        )
        .sort_values(["cache_state", "config"])
    )
    summary.to_csv(output_dir / "cold_warm_breakdown_summary.csv", index=False)

    states = ["cold", "warm"]
    x = np.arange(len(states))
    width = 0.34
    off = summary[summary["config"] == "prefix_off"].set_index("cache_state")
    on = summary[summary["config"] == "prefix_on"].set_index("cache_state")
    off_median = off.reindex(states)["median_ttft_ms"].to_numpy()
    on_median = on.reindex(states)["median_ttft_ms"].to_numpy()
    off_p25 = off.reindex(states)["p25_ttft_ms"].to_numpy()
    off_p75 = off.reindex(states)["p75_ttft_ms"].to_numpy()
    on_p25 = on.reindex(states)["p25_ttft_ms"].to_numpy()
    on_p75 = on.reindex(states)["p75_ttft_ms"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars_off = ax.bar(
        x - width / 2,
        off_median,
        width,
        yerr=np.vstack([off_median - off_p25, off_p75 - off_median]),
        capsize=4,
        label="Prefix Cache OFF",
        color=OFF_COLOR,
        alpha=0.86,
    )
    bars_on = ax.bar(
        x + width / 2,
        on_median,
        width,
        yerr=np.vstack([on_median - on_p25, on_p75 - on_median]),
        capsize=4,
        label="Prefix Cache ON",
        color=ON_COLOR,
        alpha=0.86,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Cold requests", "Warm requests"])
    ax.set_ylabel("Median TTFT (ms)")
    ax.set_title("Prefix cache benefit comes from warm requests", pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    add_bar_labels(ax, bars_off)
    add_bar_labels(ax, bars_on)
    fig.tight_layout()
    save_figure(fig, output_dir, "cold_warm_breakdown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix-length-off",
        type=Path,
        default=BASELINE_LOG_DIR / "prefix_off_prefix_length_sweep.csv",
    )
    parser.add_argument(
        "--prefix-length-on",
        type=Path,
        default=BASELINE_LOG_DIR / "prefix_on_prefix_length_sweep.csv",
    )
    parser.add_argument(
        "--reuse-ratio-off",
        type=Path,
        default=BASELINE_LOG_DIR / "prefix_off_reuse_ratio_sweep.csv",
    )
    parser.add_argument(
        "--reuse-ratio-on",
        type=Path,
        default=BASELINE_LOG_DIR / "prefix_on_reuse_ratio_sweep.csv",
    )
    parser.add_argument(
        "--matrix-off",
        type=Path,
        default=BASELINE_LOG_DIR / "prefix_off_prefix_reuse_matrix.csv",
    )
    parser.add_argument(
        "--matrix-on",
        type=Path,
        default=BASELINE_LOG_DIR / "prefix_on_prefix_reuse_matrix.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=BASELINE_FIGURE_DIR)
    parser.add_argument(
        "--allow-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip figure types whose input CSV pair is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_available_pairs(args)
    if not frames:
        raise RuntimeError("No input CSV pairs found.")
    if "prefix_length_sweep" in frames:
        plot_prefix_length_sweep(frames["prefix_length_sweep"], args.output_dir)
    if "reuse_ratio_sweep" in frames:
        plot_reuse_ratio_sweep(frames["reuse_ratio_sweep"], args.output_dir)
    if "prefix_reuse_matrix" in frames:
        plot_prefix_reuse_heatmap(frames["prefix_reuse_matrix"], args.output_dir)
    plot_cold_warm_breakdown(frames, args.output_dir)
    print(f"Wrote quantitative baseline plots and summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
