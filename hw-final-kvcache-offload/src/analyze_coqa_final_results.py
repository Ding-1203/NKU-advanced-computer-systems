#!/usr/bin/env python3
"""Analyze final CoQA real-data prefix-cache experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_REUSE_OFF = Path("logs/coqa_final/coqa_final_reuse_no_prefix.csv")
DEFAULT_REUSE_ON = Path("logs/coqa_final/coqa_final_reuse_prefix_observed.csv")
DEFAULT_REUSE_EVENTS = Path("logs/cache_observation/coqa_final/cache_events_reuse.jsonl")
DEFAULT_SUMMARY_DIR = Path("logs/cache_observation/coqa_final")
DEFAULT_FIGURE_DIR = Path("figures/cache_observation/coqa_final")
DEFAULT_PRESSURE_LEVELS = [8, 16, 24, 32, 48]
STATE_ORDER = ["cold", "warm"]
STATE_LABELS = {"cold": "Cold", "warm": "Warm"}


def default_pressure_csvs() -> list[Path]:
    return [Path(f"logs/coqa_final/coqa_final_pressure_g{level}_observed.csv") for level in DEFAULT_PRESSURE_LEVELS]


def default_pressure_events() -> list[Path]:
    return [Path(f"logs/cache_observation/coqa_final/cache_events_pressure_g{level}.jsonl") for level in DEFAULT_PRESSURE_LEVELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-off-csv", type=Path, default=DEFAULT_REUSE_OFF)
    parser.add_argument("--reuse-on-csv", type=Path, default=DEFAULT_REUSE_ON)
    parser.add_argument("--reuse-events", type=Path, default=DEFAULT_REUSE_EVENTS)
    parser.add_argument("--pressure-csvs", type=Path, nargs="*", default=None)
    parser.add_argument("--pressure-events", type=Path, nargs="*", default=None)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--useful-coverage-threshold", type=float, default=0.5)
    return parser.parse_args()


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def read_latency(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing latency CSV: {path}")
    data = pd.read_csv(path)
    numeric_columns = [
        "request_index",
        "target_prefix_tokens",
        "actual_prefix_tokens",
        "prefix_target_tokens",
        "prefix_tokens",
        "suffix_tokens",
        "prompt_tokens",
        "num_prefix_groups",
        "total_requests",
        "requests_per_group",
        "request_index_in_group",
        "occurrence_index",
        "expected_reused_prefix_tokens",
        "expected_reuse_ratio_token",
        "prefix_group_index",
        "pressure_level",
        "sequence_pass",
        "prefix_bucket",
        "coqa_turn_id",
        "context_pack_size",
        "planned_reuse_count",
        "max_tokens",
        "started_at_unix",
        "finished_at_unix",
        "ttft_ms",
        "total_latency_ms",
        "output_chars",
    ]
    for column in numeric_columns:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in [
        "id",
        "request_id",
        "scenario_id",
        "scenario_label",
        "experiment_type",
        "expected_cache_state",
        "prefix_group",
        "workload_name",
        "phase",
        "pressure_group_role",
        "pressure_group_id",
        "traffic_class",
        "coqa_story_id",
        "status",
    ]:
        if column not in data:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)
    if "is_warm_request" not in data:
        data["is_warm_request"] = False
    data["is_warm_request"] = to_bool(data["is_warm_request"])
    data["latency_source"] = path.stem
    return data


def read_latency_many(paths: list[Path]) -> pd.DataFrame:
    frames = [read_latency(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def read_events(paths: Path | list[Path]) -> pd.DataFrame:
    if isinstance(paths, Path):
        paths = [paths]
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing event log: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                row["event_source"] = path.stem
                rows.append(row)
    events = pd.DataFrame(rows)
    if events.empty:
        return pd.DataFrame(columns=["request_id", "event_type"])
    for column in ["timestamp", "hit_tokens", "hit_blocks", "block_id", "ref_cnt"]:
        if column in events:
            events[column] = pd.to_numeric(events[column], errors="coerce")
    for column in ["request_id", "event_type", "event_source"]:
        if column not in events:
            events[column] = ""
        events[column] = events[column].fillna("").astype(str)
    return events


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
                "prefix_lookup_count": int((event_types == "prefix_lookup").sum()),
                "prefix_hit_count": int((event_types == "prefix_hit").sum()),
                "prefix_miss_count": int((event_types == "prefix_miss").sum()),
                "block_touch_count": int((event_types == "block_touch").sum()),
                "block_store_count": int((event_types == "block_store").sum()),
                "block_free_count": int((event_types == "block_free").sum()),
                "block_evict_count_by_request": int((event_types == "block_evict").sum()),
                "hit_tokens": float(hit_tokens.max()) if not hit_tokens.dropna().empty else 0.0,
                "hit_blocks": float(hit_blocks.max()) if not hit_blocks.dropna().empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def add_window_evictions(latency: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    if events.empty or "timestamp" not in events:
        return pd.Series([0] * len(latency), index=latency.index)
    evictions = events[events["event_type"] == "block_evict"].copy()
    if evictions.empty:
        return pd.Series([0] * len(latency), index=latency.index)
    evict_ts = pd.to_numeric(evictions["timestamp"], errors="coerce").dropna().to_numpy()
    counts = []
    for _, row in latency.iterrows():
        start = row.get("started_at_unix")
        finish = row.get("finished_at_unix")
        if pd.isna(start) or pd.isna(finish):
            counts.append(0)
            continue
        counts.append(int(((evict_ts >= start) & (evict_ts <= finish)).sum()))
    return pd.Series(counts, index=latency.index)


def add_event_columns(data: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    summary = data.merge(aggregate_events(events), on="request_id", how="left")
    for column in [
        "prefix_lookup_count",
        "prefix_hit_count",
        "prefix_miss_count",
        "block_touch_count",
        "block_store_count",
        "block_free_count",
        "block_evict_count_by_request",
        "hit_tokens",
        "hit_blocks",
    ]:
        if column not in summary:
            summary[column] = 0
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0)
    denominator = summary["actual_prefix_tokens"].replace(0, np.nan)
    summary["prefix_hit_coverage"] = (summary["hit_tokens"] / denominator).clip(0, 1).fillna(0)
    summary["cache_hit"] = summary["prefix_hit_count"] > 0
    summary["event_joined"] = summary["prefix_lookup_count"] > 0
    summary["block_evict_count_in_window"] = add_window_evictions(summary, events)
    return summary


def build_reuse_summary(off: pd.DataFrame, on: pd.DataFrame, events: pd.DataFrame, useful_threshold: float) -> pd.DataFrame:
    off_ok = off[off["status"] == "ok"].copy()
    on_ok = on[on["status"] == "ok"].copy()
    metadata_columns = [column for column in on_ok.columns if column not in {"ttft_ms", "total_latency_ms", "started_at_unix", "finished_at_unix", "output_chars", "error"}]
    summary = on_ok[metadata_columns + ["ttft_ms", "total_latency_ms", "started_at_unix", "finished_at_unix"]].rename(
        columns={
            "ttft_ms": "prefix_on_ttft_ms",
            "total_latency_ms": "prefix_on_total_latency_ms",
            "started_at_unix": "prefix_on_started_at_unix",
            "finished_at_unix": "prefix_on_finished_at_unix",
        }
    )
    summary = summary.merge(
        off_ok[["request_id", "ttft_ms", "total_latency_ms", "started_at_unix", "finished_at_unix"]].rename(
            columns={
                "ttft_ms": "prefix_off_ttft_ms",
                "total_latency_ms": "prefix_off_total_latency_ms",
                "started_at_unix": "prefix_off_started_at_unix",
                "finished_at_unix": "prefix_off_finished_at_unix",
            }
        ),
        on="request_id",
        how="inner",
    )
    summary = add_event_columns(summary, events)
    summary["ttft_delta_ms"] = summary["prefix_off_ttft_ms"] - summary["prefix_on_ttft_ms"]
    summary["ttft_speedup"] = summary["prefix_off_ttft_ms"] / summary["prefix_on_ttft_ms"]
    summary["useful_hit"] = summary["prefix_hit_coverage"] >= useful_threshold
    return summary


def build_pressure_summary(latency: pd.DataFrame, events: pd.DataFrame, useful_threshold: float) -> pd.DataFrame:
    ok = latency[latency["status"] == "ok"].copy()
    summary = add_event_columns(ok, events)
    summary["useful_hit"] = summary["prefix_hit_coverage"] >= useful_threshold
    return summary


def savefig(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def jitter_points(ax: plt.Axes, x: float, y: np.ndarray, rng: np.random.Generator, color: str, size: float = 14) -> None:
    if len(y) == 0:
        return
    jitter = rng.normal(0, 0.045, size=len(y))
    ax.scatter(np.full(len(y), x) + jitter, y, s=size, color=color, alpha=0.30, edgecolors="none")


def save_prefix_length_savings(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary[(summary["experiment_type"] == "coqa_prefix_length") & (summary["expected_cache_state"] == "warm")].copy()
    buckets = sorted(int(value) for value in data["prefix_bucket"].dropna().unique())
    values = [data.loc[data["prefix_bucket"] == bucket, "ttft_delta_ms"].dropna().to_numpy() for bucket in buckets]
    med_saved = [float(np.median(series)) if len(series) else np.nan for series in values]
    speedups = [float(data.loc[data["prefix_bucket"] == bucket, "ttft_speedup"].median()) for bucket in buckets]
    positions = np.arange(len(buckets))
    fig, ax1 = plt.subplots(figsize=(9.2, 5.4))
    box = ax1.boxplot(
        values,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0f172a", "linewidth": 2.0},
        boxprops={"facecolor": "#bfdbfe", "edgecolor": "#2563eb", "linewidth": 1.2},
        whiskerprops={"color": "#2563eb", "linewidth": 1.1},
        capprops={"color": "#2563eb", "linewidth": 1.1},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.68)
    rng = np.random.default_rng(41)
    for pos, series in zip(positions, values):
        jitter_points(ax1, pos, series, rng, "#111827")
    ax1.axhline(0, color="#64748b", linestyle="--", linewidth=1.0, alpha=0.75)
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f"{bucket}\nn={len(series)}" for bucket, series in zip(buckets, values)])
    ax1.set_xlabel("CoQA context prefix target (tokens)")
    ax1.set_ylabel("TTFT saved by Prefix Cache ON (ms)")
    ax1.grid(axis="y", alpha=0.25)

    y1_min = min(-6.0, np.nanmin([np.nanmin(series) for series in values if len(series)]) - 2.0)
    y1_max = max(44.0, np.nanmax([np.nanmax(series) for series in values if len(series)]) + 4.0)
    ax1.set_ylim(y1_min, y1_max)

    ax2 = ax1.twinx()
    ax2.plot(positions, speedups, color="#047857", marker="o", linewidth=2.4, label="Median speedup")
    ax2.set_ylabel("Median speedup")
    y2_max = max(2.35, max(speedups) * 1.14)
    zero_fraction = (0.0 - y1_min) / (y1_max - y1_min)
    # Align semantic no-benefit points: left 0 ms saved == right 1.0x speedup.
    y2_min = (1.0 - zero_fraction * y2_max) / max(1e-9, 1.0 - zero_fraction)
    ax2.set_ylim(y2_min, y2_max)
    ax2.set_yticks([tick for tick in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5] if y2_min <= tick <= y2_max])
    for pos, saved, speedup in zip(positions, med_saved, speedups):
        ax2.annotate(
            f"{saved:.1f} ms\n{speedup:.2f}x",
            xy=(pos, speedup),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#064e3b",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
    ax1.set_title("Longer CoQA prefixes yield larger TTFT savings")
    ax2.legend(frameon=False, loc="upper left")
    savefig(fig, output_dir, "01_coqa_prefix_length_ttft_savings.png")


def save_on_off_by_state(summary: pd.DataFrame, output_dir: Path) -> None:
    centers = np.arange(len(STATE_ORDER))
    off_values = [summary.loc[summary["expected_cache_state"] == state, "prefix_off_ttft_ms"].dropna().to_numpy() for state in STATE_ORDER]
    on_values = [summary.loc[summary["expected_cache_state"] == state, "prefix_on_ttft_ms"].dropna().to_numpy() for state in STATE_ORDER]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    rng = np.random.default_rng(43)
    for positions, values, color in [(centers - 0.18, off_values, "#94a3b8"), (centers + 0.18, on_values, "#2563eb")]:
        box = ax.boxplot(
            values,
            positions=positions,
            widths=0.32,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#0f172a", "linewidth": 1.9},
            boxprops={"facecolor": color, "edgecolor": color, "linewidth": 1.2},
            whiskerprops={"color": color, "linewidth": 1.1},
            capprops={"color": color, "linewidth": 1.1},
        )
        for patch in box["boxes"]:
            patch.set_alpha(0.5)
        for pos, series in zip(positions, values):
            jitter_points(ax, pos, series, rng, "#111827", size=10)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{STATE_LABELS[state]}\nn={len(summary[summary['expected_cache_state'] == state])}" for state in STATE_ORDER])
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Prefix Cache mainly benefits repeated CoQA context")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#94a3b8", linewidth=8, alpha=0.55, label="Prefix Cache OFF"),
            plt.Line2D([0], [0], color="#2563eb", linewidth=8, alpha=0.55, label="Prefix Cache ON"),
        ],
        frameon=False,
        loc="upper right",
    )
    savefig(fig, output_dir, "02_coqa_on_off_by_cache_state.png")


def save_popularity_pareto(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary[(summary["experiment_type"] == "coqa_popularity_skew") & (summary["expected_cache_state"] == "warm")].copy()
    grouped = (
        data.groupby(["prefix_group", "traffic_class"], as_index=False)
        .agg(cumulative_ttft_saved_ms=("ttft_delta_ms", "sum"), cumulative_hit_tokens=("hit_tokens", "sum"), revisits=("request_id", "count"))
        .sort_values("cumulative_ttft_saved_ms", ascending=False)
    )
    grouped["utility"] = grouped["cumulative_ttft_saved_ms"].clip(lower=0)
    if grouped["utility"].sum() <= 0:
        grouped["utility"] = grouped["cumulative_hit_tokens"].clip(lower=0)
    grouped = grouped.sort_values("utility", ascending=False).reset_index(drop=True)
    grouped["group_rank_pct"] = (np.arange(1, len(grouped) + 1) / len(grouped)) * 100.0
    grouped["cumulative_utility_pct"] = grouped["utility"].cumsum() / grouped["utility"].sum() * 100.0
    top20 = grouped[grouped["group_rank_pct"] <= 20.0]
    top20_share = float(top20["utility"].sum() / grouped["utility"].sum() * 100.0) if not top20.empty else 0.0

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(grouped["group_rank_pct"], grouped["cumulative_utility_pct"], color="#2563eb", linewidth=2.6)
    ax.fill_between(grouped["group_rank_pct"], grouped["cumulative_utility_pct"], color="#bfdbfe", alpha=0.42)
    ax.axvline(20, color="#0f766e", linestyle="--", linewidth=1.2)
    ax.axhline(top20_share, color="#0f766e", linestyle="--", linewidth=1.2)
    ax.scatter([20], [top20_share], color="#0f766e", s=46, zorder=3)
    ax.text(22, top20_share - 6, f"Top 20% prefixes\n{top20_share:.0f}% of saved TTFT", color="#064e3b", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 104)
    ax.set_xlabel("Prefix groups sorted by cumulative utility (%)")
    ax.set_ylabel("Cumulative TTFT saved (%)")
    ax.set_title("CoQA cache utility is highly skewed across prefix groups")
    ax.grid(alpha=0.24)
    savefig(fig, output_dir, "03_coqa_popularity_utility_pareto.png")


def save_pressure_cliff(pressure: pd.DataFrame, reuse: pd.DataFrame, output_dir: Path) -> None:
    revisit = pressure[pressure["phase"] == "revisit"].copy()
    levels = sorted(int(level) for level in revisit["pressure_level"].dropna().unique())
    positions = np.arange(len(levels))
    coverage_values = [(revisit.loc[revisit["pressure_level"] == level, "prefix_hit_coverage"] * 100.0).dropna().to_numpy() for level in levels]
    ttft_values = [revisit.loc[revisit["pressure_level"] == level, "ttft_ms"].dropna().to_numpy() for level in levels]
    useful_rates = [revisit.loc[revisit["pressure_level"] == level, "useful_hit"].mean() for level in levels]
    evictions = pressure.groupby("pressure_level")["block_evict_count_in_window"].sum().reindex(levels).fillna(0).astype(int)

    warm_baseline = reuse[(reuse["experiment_type"] == "coqa_prefix_length") & (reuse["expected_cache_state"] == "warm") & (reuse["prefix_bucket"] == 1536)]["prefix_on_ttft_ms"].median()
    cold_baseline = pressure[pressure["phase"].isin(["seed", "filler"])]["ttft_ms"].median()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 8.1), sharex=True, gridspec_kw={"height_ratios": [1.12, 1.05]})
    box1 = ax1.boxplot(
        coverage_values,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0f172a", "linewidth": 2.0},
        boxprops={"facecolor": "#bfdbfe", "edgecolor": "#2563eb", "linewidth": 1.2},
        whiskerprops={"color": "#2563eb", "linewidth": 1.1},
        capprops={"color": "#2563eb", "linewidth": 1.1},
    )
    for patch in box1["boxes"]:
        patch.set_alpha(0.68)
    ax1.plot(positions, [np.median(v) for v in coverage_values], color="#1d4ed8", marker="o", linewidth=2.0)
    for pos, rate, level in zip(positions, useful_rates, levels):
        ax1.text(
            pos,
            116,
            f"useful {rate * 100:.0f}%\nevict {int(evictions.loc[level])}",
            ha="center",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    ax1.set_ylim(-8, 124)
    ax1.set_ylabel("Revisit prefix-hit coverage (%)")
    ax1.set_title("Capacity pressure collapses useful CoQA prefix reuse")
    ax1.grid(axis="y", alpha=0.25)

    box2 = ax2.boxplot(
        ttft_values,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0f172a", "linewidth": 2.0},
        boxprops={"facecolor": "#bbf7d0", "edgecolor": "#047857", "linewidth": 1.2},
        whiskerprops={"color": "#047857", "linewidth": 1.1},
        capprops={"color": "#047857", "linewidth": 1.1},
    )
    for patch in box2["boxes"]:
        patch.set_alpha(0.70)
    ax2.axhline(warm_baseline, color="#2563eb", linestyle="--", linewidth=1.3, label=f"High-cap warm baseline ({warm_baseline:.1f} ms)")
    ax2.axhline(cold_baseline, color="#64748b", linestyle="--", linewidth=1.3, label=f"No useful hit baseline ({cold_baseline:.1f} ms)")
    ax2.set_xticks(positions)
    ax2.set_xticklabels([str(level) for level in levels])
    ax2.set_xlabel("Seed prefix groups before filler")
    ax2.set_ylabel("Revisit TTFT (ms)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False, loc="best")
    savefig(fig, output_dir, "04_coqa_pressure_cliff.png")


def print_summary(reuse: pd.DataFrame, pressure: pd.DataFrame, reuse_events: pd.DataFrame, pressure_events: pd.DataFrame) -> None:
    print("prefix length warm medians")
    length = reuse[(reuse["experiment_type"] == "coqa_prefix_length") & (reuse["expected_cache_state"] == "warm")]
    print(length.groupby("prefix_bucket").agg(requests=("request_id", "count"), off_ttft=("prefix_off_ttft_ms", "median"), on_ttft=("prefix_on_ttft_ms", "median"), saved_ms=("ttft_delta_ms", "median"), speedup=("ttft_speedup", "median"), coverage=("prefix_hit_coverage", "median")).round(4).to_string())
    print()
    print("popularity group utility")
    pop = reuse[(reuse["experiment_type"] == "coqa_popularity_skew") & (reuse["expected_cache_state"] == "warm")]
    grouped = pop.groupby(["traffic_class", "prefix_group"], as_index=False).agg(cum_saved=("ttft_delta_ms", "sum"), cum_hit_tokens=("hit_tokens", "sum"), revisits=("request_id", "count"))
    print(grouped.groupby("traffic_class").agg(groups=("prefix_group", "count"), median_revisits=("revisits", "median"), median_cum_saved_ms=("cum_saved", "median"), median_cum_hit_tokens=("cum_hit_tokens", "median")).round(4).to_string())
    total_utility = grouped["cum_saved"].clip(lower=0).sum()
    top_n = max(1, int(np.ceil(len(grouped) * 0.2)))
    top_share = grouped.assign(utility=grouped["cum_saved"].clip(lower=0)).sort_values("utility", ascending=False).head(top_n)["utility"].sum() / total_utility if total_utility else 0.0
    print(f"top20_share={top_share:.4f}")
    print()
    print("pressure revisit")
    revisit = pressure[pressure["phase"] == "revisit"]
    print(revisit.groupby("pressure_level").agg(requests=("request_id", "count"), ttft_ms=("ttft_ms", "median"), coverage=("prefix_hit_coverage", "median"), useful_hit_rate=("useful_hit", "mean")).round(4).to_string())
    print("pressure evictions")
    print(pressure.groupby("pressure_level")["block_evict_count_in_window"].sum().round(0).astype(int).to_string())
    print()
    print(f"reuse join rate: {reuse['event_joined'].mean():.3f} ({reuse['event_joined'].sum()}/{len(reuse)})")
    print(f"pressure join rate: {pressure['event_joined'].mean():.3f} ({pressure['event_joined'].sum()}/{len(pressure)})")
    print("reuse event counts")
    print(reuse_events["event_type"].value_counts().to_string())
    print("pressure event counts")
    print(pressure_events["event_type"].value_counts().to_string())


def main() -> None:
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    pressure_csvs = args.pressure_csvs if args.pressure_csvs else default_pressure_csvs()
    pressure_event_paths = args.pressure_events if args.pressure_events else default_pressure_events()

    reuse_off = read_latency(args.reuse_off_csv)
    reuse_on = read_latency(args.reuse_on_csv)
    reuse_events = read_events(args.reuse_events)
    pressure_latency = read_latency_many(pressure_csvs)
    pressure_events = read_events(pressure_event_paths)

    reuse_summary = build_reuse_summary(reuse_off, reuse_on, reuse_events, args.useful_coverage_threshold)
    pressure_summary = build_pressure_summary(pressure_latency, pressure_events, args.useful_coverage_threshold)

    reuse_summary.to_csv(args.summary_dir / "coqa_final_reuse_summary.csv", index=False)
    pressure_summary.to_csv(args.summary_dir / "coqa_final_pressure_summary.csv", index=False)

    save_prefix_length_savings(reuse_summary, args.figure_dir)
    save_on_off_by_state(reuse_summary, args.figure_dir)
    save_popularity_pareto(reuse_summary, args.figure_dir)
    save_pressure_cliff(pressure_summary, reuse_summary, args.figure_dir)

    print_summary(reuse_summary, pressure_summary, reuse_events, pressure_events)
    print(f"wrote summaries to {args.summary_dir}")
    print(f"wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
