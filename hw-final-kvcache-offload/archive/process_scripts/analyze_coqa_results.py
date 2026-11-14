#!/usr/bin/env python3
"""Analyze the real CoQA prefix-cache experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_NO_PREFIX_CSV = Path("logs/coqa/coqa_train_long_no_prefix.csv")
DEFAULT_PREFIX_CSV = Path("logs/coqa/coqa_train_long_prefix_observed.csv")
DEFAULT_EVENT_LOG = Path("logs/cache_observation/coqa/cache_events_train_long.jsonl")
DEFAULT_SUMMARY = Path("logs/cache_observation/coqa/coqa_train_long_summary.csv")
DEFAULT_FIGURE_DIR = Path("figures/cache_observation/coqa")

STATE_ORDER = ["cold", "warm"]
STATE_LABELS = {"cold": "Cold story turn", "warm": "Warm story turns"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-prefix-csv", type=Path, default=DEFAULT_NO_PREFIX_CSV)
    parser.add_argument("--prefix-csv", type=Path, default=DEFAULT_PREFIX_CSV)
    parser.add_argument("--event-log", type=Path, default=DEFAULT_EVENT_LOG)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--useful-coverage-threshold",
        type=float,
        default=0.5,
        help="Prefix-hit coverage threshold used for useful-hit rate.",
    )
    return parser.parse_args()


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def read_latency(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing latency CSV: {path}")
    data = pd.read_csv(path)
    data["run_label"] = label
    for column in [
        "request_index",
        "request_index_in_group",
        "occurrence_index",
        "actual_prefix_tokens",
        "prefix_tokens",
        "prompt_tokens",
        "expected_reused_prefix_tokens",
        "started_at_unix",
        "finished_at_unix",
        "ttft_ms",
        "total_latency_ms",
    ]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "is_warm_request" in data:
        data["is_warm_request"] = to_bool(data["is_warm_request"])
    else:
        data["is_warm_request"] = False
    for column in ["request_id", "expected_cache_state", "prefix_group", "status"]:
        if column not in data:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)
    return data


def read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing event log: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    events = pd.DataFrame(rows)
    if events.empty:
        return pd.DataFrame(columns=["request_id", "event_type"])
    for column in ["timestamp", "hit_tokens", "hit_blocks", "block_id", "ref_cnt"]:
        if column in events:
            events[column] = pd.to_numeric(events[column], errors="coerce")
    for column in ["request_id", "event_type"]:
        if column not in events:
            events[column] = ""
        events[column] = events[column].fillna("").astype(str)
    return events


def count_event(event_types: pd.Series, event_type: str) -> int:
    return int((event_types == event_type).sum())


def aggregate_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["request_id"])
    request_events = events[(events["request_id"] != "") & (events["request_id"] != "runner-warmup")]
    rows = []
    for request_id, group in request_events.groupby("request_id"):
        event_types = group["event_type"]
        hit_tokens = pd.to_numeric(group.get("hit_tokens"), errors="coerce")
        hit_blocks = pd.to_numeric(group.get("hit_blocks"), errors="coerce")
        rows.append(
            {
                "request_id": request_id,
                "prefix_lookup_count": count_event(event_types, "prefix_lookup"),
                "prefix_hit_count": count_event(event_types, "prefix_hit"),
                "prefix_miss_count": count_event(event_types, "prefix_miss"),
                "block_touch_count": count_event(event_types, "block_touch"),
                "block_store_count": count_event(event_types, "block_store"),
                "block_free_count": count_event(event_types, "block_free"),
                "block_evict_count": count_event(event_types, "block_evict"),
                "hit_tokens": float(hit_tokens.max()) if not hit_tokens.dropna().empty else 0.0,
                "hit_blocks": float(hit_blocks.max()) if not hit_blocks.dropna().empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_summary(no_prefix: pd.DataFrame, prefix: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    off = no_prefix[no_prefix["status"] == "ok"].copy()
    on = prefix[prefix["status"] == "ok"].copy()
    event_summary = aggregate_events(events)

    metadata_columns = [
        "request_id",
        "request_index",
        "request_index_in_group",
        "occurrence_index",
        "expected_cache_state",
        "is_warm_request",
        "prefix_group",
        "actual_prefix_tokens",
        "prefix_tokens",
        "prompt_tokens",
        "expected_reused_prefix_tokens",
    ]
    available_metadata = [column for column in metadata_columns if column in on]
    summary = on[available_metadata + ["ttft_ms", "total_latency_ms"]].rename(
        columns={
            "ttft_ms": "prefix_on_ttft_ms",
            "total_latency_ms": "prefix_on_total_latency_ms",
        }
    )
    summary = summary.merge(
        off[["request_id", "ttft_ms", "total_latency_ms"]].rename(
            columns={
                "ttft_ms": "prefix_off_ttft_ms",
                "total_latency_ms": "prefix_off_total_latency_ms",
            }
        ),
        on="request_id",
        how="inner",
    )
    summary = summary.merge(event_summary, on="request_id", how="left")
    for column in [
        "prefix_lookup_count",
        "prefix_hit_count",
        "prefix_miss_count",
        "block_touch_count",
        "block_store_count",
        "block_free_count",
        "block_evict_count",
        "hit_tokens",
        "hit_blocks",
    ]:
        if column not in summary:
            summary[column] = 0
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0)
    denominator = summary["actual_prefix_tokens"].replace(0, np.nan)
    summary["prefix_hit_coverage"] = (summary["hit_tokens"] / denominator).clip(0, 1).fillna(0)
    summary["event_joined"] = summary["prefix_lookup_count"] > 0
    summary["cache_hit"] = summary["prefix_hit_count"] > 0
    summary["useful_hit"] = summary["prefix_hit_coverage"] >= 0.5
    summary["ttft_delta_ms"] = summary["prefix_off_ttft_ms"] - summary["prefix_on_ttft_ms"]
    summary["ttft_speedup"] = summary["prefix_off_ttft_ms"] / summary["prefix_on_ttft_ms"]
    return summary


def savefig(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def boxplot_with_points(
    ax: plt.Axes,
    values: list[np.ndarray],
    positions: np.ndarray,
    color: str,
    rng: np.random.Generator,
    width: float = 0.32,
) -> None:
    box = ax.boxplot(
        values,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0f172a", "linewidth": 1.9},
        boxprops={"facecolor": color, "edgecolor": color, "linewidth": 1.2},
        whiskerprops={"color": color, "linewidth": 1.1},
        capprops={"color": color, "linewidth": 1.1},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.48)
    for position, series in zip(positions, values):
        if len(series) == 0:
            continue
        jitter = rng.normal(0, 0.035, size=len(series))
        ax.scatter(
            np.full(len(series), position) + jitter,
            series,
            s=17,
            color="#111827",
            alpha=0.32,
            edgecolors="none",
        )


def save_ttft_on_off(summary: pd.DataFrame, output_dir: Path) -> None:
    rng = np.random.default_rng(17)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    centers = np.arange(len(STATE_ORDER))
    off_values = [
        summary.loc[summary["expected_cache_state"] == state, "prefix_off_ttft_ms"].dropna().to_numpy()
        for state in STATE_ORDER
    ]
    on_values = [
        summary.loc[summary["expected_cache_state"] == state, "prefix_on_ttft_ms"].dropna().to_numpy()
        for state in STATE_ORDER
    ]
    boxplot_with_points(ax, off_values, centers - 0.18, "#94a3b8", rng)
    boxplot_with_points(ax, on_values, centers + 0.18, "#2563eb", rng)
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [f"{STATE_LABELS[state]}\nn={len(summary[summary['expected_cache_state'] == state])}" for state in STATE_ORDER]
    )
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("CoQA real workload: Prefix Cache ON vs OFF")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#94a3b8", linewidth=8, alpha=0.55, label="Prefix Cache OFF"),
            plt.Line2D([0], [0], color="#2563eb", linewidth=8, alpha=0.55, label="Prefix Cache ON"),
        ],
        frameon=False,
        loc="upper right",
    )
    savefig(fig, output_dir, "coqa_ttft_prefix_on_vs_off.png")


def save_coverage_by_turn(summary: pd.DataFrame, output_dir: Path) -> None:
    turns = sorted(int(turn) for turn in summary["occurrence_index"].dropna().unique())
    values = [
        (summary.loc[summary["occurrence_index"] == turn, "prefix_hit_coverage"].dropna() * 100.0).to_numpy()
        for turn in turns
    ]
    labels = [f"Turn {turn + 1}\nn={len(series)}" for turn, series in zip(turns, values)]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    positions = np.arange(len(turns))
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0f172a", "linewidth": 1.9},
        boxprops={"facecolor": "#bfdbfe", "edgecolor": "#2563eb", "linewidth": 1.2},
        whiskerprops={"color": "#2563eb", "linewidth": 1.1},
        capprops={"color": "#2563eb", "linewidth": 1.1},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.68)
    rng = np.random.default_rng(23)
    for position, series in zip(positions, values):
        jitter = rng.normal(0, 0.045, size=len(series))
        ax.scatter(
            np.full(len(series), position) + jitter,
            series,
            s=18,
            color="#111827",
            alpha=0.38,
            edgecolors="none",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(-5, 105)
    ax.set_ylabel("Observed prefix-hit coverage (%)")
    ax.set_title("CoQA story reuse appears after the first turn")
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, output_dir, "coqa_prefix_coverage_by_turn.png")


def save_speedup_vs_coverage(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    warm = summary["expected_cache_state"] == "warm"
    cold = ~warm
    ax.axhline(0, color="#64748b", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.scatter(
        summary.loc[cold, "prefix_hit_coverage"] * 100.0,
        summary.loc[cold, "ttft_delta_ms"],
        s=32,
        color="#94a3b8",
        alpha=0.72,
        edgecolors="none",
        label="Cold",
    )
    scatter = ax.scatter(
        summary.loc[warm, "prefix_hit_coverage"] * 100.0,
        summary.loc[warm, "ttft_delta_ms"],
        s=np.clip(summary.loc[warm, "actual_prefix_tokens"] / 12.0, 28, 82),
        c=summary.loc[warm, "actual_prefix_tokens"],
        cmap="Blues",
        alpha=0.76,
        edgecolors="#1e3a8a",
        linewidths=0.25,
        label="Warm",
    )
    ax.set_xlim(-4, 104)
    ax.set_xlabel("Observed prefix-hit coverage (%)")
    ax.set_ylabel("TTFT saved by Prefix Cache ON (ms)")
    ax.set_title("CoQA TTFT gains are modest despite near-complete prefix hits")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Actual prefix tokens")
    savefig(fig, output_dir, "coqa_speedup_by_hit_coverage.png")


def print_console_summary(summary: pd.DataFrame, events: pd.DataFrame) -> None:
    grouped = (
        summary.groupby("expected_cache_state")
        .agg(
            requests=("request_id", "count"),
            off_ttft_ms=("prefix_off_ttft_ms", "median"),
            on_ttft_ms=("prefix_on_ttft_ms", "median"),
            ttft_delta_ms=("ttft_delta_ms", "median"),
            ttft_speedup=("ttft_speedup", "median"),
            prefix_hit_coverage=("prefix_hit_coverage", "median"),
            useful_hit_rate=("useful_hit", "mean"),
        )
        .reindex(STATE_ORDER)
    )
    event_counts = events["event_type"].value_counts() if not events.empty else pd.Series(dtype=int)
    print(grouped.round(4).to_string())
    print()
    print("event counts")
    print(event_counts.to_string())
    print()
    print(f"join rate: {summary['event_joined'].mean():.3f} ({summary['event_joined'].sum()}/{len(summary)})")


def main() -> None:
    args = parse_args()
    no_prefix = read_latency(args.no_prefix_csv, "prefix_off")
    prefix = read_latency(args.prefix_csv, "prefix_on")
    events = read_events(args.event_log)
    summary = build_summary(no_prefix, prefix, events)
    summary["useful_hit"] = summary["prefix_hit_coverage"] >= args.useful_coverage_threshold

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    save_ttft_on_off(summary, args.figure_dir)
    save_coverage_by_turn(summary, args.figure_dir)
    save_speedup_vs_coverage(summary, args.figure_dir)
    print_console_summary(summary, events)
    print(f"wrote {args.summary_output}")
    print(f"wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
