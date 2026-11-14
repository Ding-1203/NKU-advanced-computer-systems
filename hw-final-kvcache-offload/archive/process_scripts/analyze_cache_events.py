#!/usr/bin/env python3
"""Analyze vLLM cache observer JSONL events and workload latency CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SMOKE_EVENT_LOGS = [Path("logs/cache_observation/smoke/cache_events.jsonl")]
SMOKE_LATENCY_CSVS = [
    Path("logs/cache_observation/smoke/observed_baseline_smoke.csv"),
    Path("logs/cache_observation/smoke/cache_pressure.csv"),
]
SMOKE_SUMMARY = Path("logs/cache_observation/smoke/cache_event_summary.csv")
SMOKE_FIGURE_DIR = Path("figures/cache_observation/smoke")

FORMAL_EVENT_LOGS = [
    Path("logs/cache_observation/formal/cache_events_matrix.jsonl"),
    Path("logs/cache_observation/formal/cache_events_pressure.jsonl"),
]
FORMAL_LATENCY_CSVS = [
    Path("logs/cache_observation/formal/observed_prefix_reuse_matrix.csv"),
    Path("logs/cache_observation/formal/cache_pressure_sweep.csv"),
]
FORMAL_SUMMARY = Path("logs/cache_observation/formal/cache_event_summary.csv")
FORMAL_FIGURE_DIR = Path("figures/cache_observation/formal")

HIT_BUCKET_LABELS = ["0", "32", "128", "256", "512", "1024", "1536+"]
HIT_BUCKET_EDGES = [-0.5, 16, 80, 192, 384, 768, 1280, np.inf]
PHASE_ORDER = ["seed", "filler", "revisit"]
PHASE_COLORS = {"seed": "#4b5563", "filler": "#9ca3af", "revisit": "#2563eb"}


def resolve_default_paths(args: argparse.Namespace) -> tuple[list[Path], list[Path], Path, Path]:
    if args.profile == "formal":
        default_events = FORMAL_EVENT_LOGS
        default_latency = FORMAL_LATENCY_CSVS
        default_summary = FORMAL_SUMMARY
        default_figures = FORMAL_FIGURE_DIR
    else:
        default_events = SMOKE_EVENT_LOGS
        default_latency = SMOKE_LATENCY_CSVS
        default_summary = SMOKE_SUMMARY
        default_figures = SMOKE_FIGURE_DIR
    return (
        args.events or default_events,
        args.latency_csv or default_latency,
        args.summary_output or default_summary,
        args.figure_dir or default_figures,
    )


def read_events(paths: list[Path]) -> pd.DataFrame:
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
    if not rows:
        return pd.DataFrame()
    events = pd.DataFrame(rows)
    for column in ["timestamp", "hit_tokens", "hit_blocks", "block_id", "ref_cnt"]:
        if column in events:
            events[column] = pd.to_numeric(events[column], errors="coerce")
    for column in ["request_id", "internal_request_id", "event_type", "event_source"]:
        if column not in events:
            events[column] = ""
        events[column] = events[column].fillna("").astype(str)
    return events


def read_latency(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["latency_source"] = path.stem
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No latency CSV inputs found.")
    latency = pd.concat(frames, ignore_index=True)
    for column in [
        "ttft_ms",
        "total_latency_ms",
        "started_at_unix",
        "finished_at_unix",
        "target_prefix_tokens",
        "actual_prefix_tokens",
        "prefix_tokens",
        "prompt_tokens",
        "reuse_ratio_target",
        "pressure_level",
        "request_index",
    ]:
        if column in latency:
            latency[column] = pd.to_numeric(latency[column], errors="coerce")
    for column in ["request_id", "phase", "experiment_type", "scenario_label", "scenario_id"]:
        if column not in latency:
            latency[column] = ""
        latency[column] = latency[column].fillna("").astype(str)
    if "is_warm_request" not in latency:
        latency["is_warm_request"] = False
    latency["is_warm_request"] = to_bool(latency["is_warm_request"])
    return latency


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def count_events(group: pd.Series, event_type: str) -> int:
    return int((group == event_type).sum())


def aggregate_request_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["request_id"])
    request_events = events[events["request_id"].astype(bool)].copy()
    request_events = request_events[request_events["request_id"] != "runner-warmup"]
    if request_events.empty:
        return pd.DataFrame(columns=["request_id"])

    rows = []
    for request_id, group in request_events.groupby("request_id"):
        event_types = group["event_type"]
        hit_tokens = pd.to_numeric(group.get("hit_tokens"), errors="coerce")
        hit_blocks = pd.to_numeric(group.get("hit_blocks"), errors="coerce")
        rows.append(
            {
                "request_id": request_id,
                "prefix_lookup_count": count_events(event_types, "prefix_lookup"),
                "prefix_hit_count": count_events(event_types, "prefix_hit"),
                "prefix_miss_count": count_events(event_types, "prefix_miss"),
                "block_touch_count": count_events(event_types, "block_touch"),
                "block_store_count": count_events(event_types, "block_store"),
                "block_free_count": count_events(event_types, "block_free"),
                "block_evict_count_by_request": count_events(event_types, "block_evict"),
                "hit_tokens": float(hit_tokens.max()) if not hit_tokens.dropna().empty else 0.0,
                "hit_blocks": float(hit_blocks.max()) if not hit_blocks.dropna().empty else 0.0,
                "first_event_ts": float(group["timestamp"].min()),
                "last_event_ts": float(group["timestamp"].max()),
            }
        )
    return pd.DataFrame(rows)


def add_time_window_evictions(latency: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
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


def build_summary(latency: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    request_summary = aggregate_request_events(events)
    summary = latency.merge(request_summary, on="request_id", how="left")
    numeric_fill = [
        "prefix_lookup_count",
        "prefix_hit_count",
        "prefix_miss_count",
        "block_touch_count",
        "block_store_count",
        "block_free_count",
        "block_evict_count_by_request",
        "hit_tokens",
        "hit_blocks",
    ]
    for column in numeric_fill:
        if column not in summary:
            summary[column] = 0
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0)
    summary["block_evict_count_in_window"] = add_time_window_evictions(summary, events)
    summary["cache_hit"] = summary["prefix_hit_count"] > 0
    summary["event_joined"] = summary["prefix_lookup_count"] > 0
    return summary


def ok_rows(summary: pd.DataFrame) -> pd.DataFrame:
    ok = summary[summary["status"] == "ok"].copy()
    ok["ttft_ms"] = pd.to_numeric(ok["ttft_ms"], errors="coerce")
    ok["hit_tokens"] = pd.to_numeric(ok["hit_tokens"], errors="coerce").fillna(0)
    return ok.dropna(subset=["ttft_ms"])


def save_ttft_vs_hit_tokens(summary: pd.DataFrame, output_dir: Path) -> None:
    ok = ok_rows(summary)
    if ok.empty:
        return
    ok["hit_bucket"] = pd.cut(
        ok["hit_tokens"],
        bins=HIT_BUCKET_EDGES,
        labels=HIT_BUCKET_LABELS,
        include_lowest=True,
    )
    bucket_data = []
    bucket_labels = []
    for label in HIT_BUCKET_LABELS:
        values = ok.loc[ok["hit_bucket"] == label, "ttft_ms"].dropna().to_numpy()
        if len(values) == 0:
            continue
        bucket_data.append(values)
        bucket_labels.append(label)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    positions = np.arange(len(bucket_data))
    box = ax.boxplot(
        bucket_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1d4ed8", "linewidth": 2},
        boxprops={"facecolor": "#bfdbfe", "edgecolor": "#2563eb", "linewidth": 1.2},
        whiskerprops={"color": "#2563eb"},
        capprops={"color": "#2563eb"},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.62)

    rng = np.random.default_rng(7)
    for index, label in enumerate(bucket_labels):
        values = ok.loc[ok["hit_bucket"] == label, "ttft_ms"].dropna().to_numpy()
        jitter = rng.normal(0, 0.055, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=24,
            color="#111827",
            alpha=0.42,
            edgecolors="none",
        )
    labels_with_counts = [
        f"{label}\nn={len(values)}" for label, values in zip(bucket_labels, bucket_data)
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_with_counts)
    ax.set_xlabel("Observed hit-token bucket")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Higher observed prefix-cache hits reduce TTFT")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ttft_vs_hit_tokens.png", dpi=180)
    fig.savefig(output_dir / "ttft_vs_hit_tokens.pdf")
    plt.close(fig)


def short_scenario_label(row: pd.Series) -> str:
    if row.get("experiment_type") == "cache_pressure":
        level = row.get("pressure_level")
        phase = str(row.get("phase") or "")
        if pd.notna(level):
            return f"{int(level)}g / {phase}"
        return phase
    tokens = row.get("target_prefix_tokens")
    reuse = row.get("reuse_ratio_target")
    if pd.notna(tokens) and pd.notna(reuse):
        return f"{int(tokens)} tok / {int(round(float(reuse) * 100))}%"
    return str(row.get("scenario_label") or row.get("scenario_id") or "unknown")


def save_cache_hit_by_scenario(summary: pd.DataFrame, output_dir: Path, kind: str) -> None:
    ok = ok_rows(summary)
    if kind == "baseline":
        data = ok[ok["experiment_type"].isin(["prefix_length_sweep", "reuse_ratio_sweep", "prefix_reuse_matrix"])]
        title = "Baseline observed prefix-cache hit behavior"
        filename = "baseline_cache_hit_by_scenario"
    else:
        data = ok[ok["experiment_type"] == "cache_pressure"]
        title = "Pressure observed prefix-cache hit behavior"
        filename = "pressure_cache_hit_by_scenario"
    if data.empty:
        return

    data = data.copy()
    data["plot_label"] = data.apply(short_scenario_label, axis=1)
    grouped = (
        data.groupby("plot_label", as_index=False)
        .agg(
            avg_hit_tokens=("hit_tokens", "mean"),
            hit_rate=("cache_hit", "mean"),
            requests=("request_id", "count"),
        )
        .sort_values(["avg_hit_tokens", "hit_rate"], ascending=False)
    )
    fig, ax1 = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(grouped))
    bars = ax1.bar(x, grouped["avg_hit_tokens"], color="#2563eb", alpha=0.82)
    ax1.set_ylabel("Average hit tokens")
    ax1.set_xticks(x)
    ax1.set_xticklabels(grouped["plot_label"], rotation=35, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, grouped["hit_rate"], color="#b45309", marker="o", linewidth=2)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Request hit rate")
    ax1.set_title(title)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.png", dpi=180)
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def save_hit_tokens_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    ok = ok_rows(summary)
    matrix = ok[ok["experiment_type"] == "prefix_reuse_matrix"].copy()
    if matrix.empty:
        return
    warm = matrix[matrix["is_warm_request"]]
    prefix_values = [128, 256, 512, 1024, 1536]
    reuse_values = [0.0, 0.25, 0.5, 0.75, 0.9]
    values = pd.DataFrame(0.0, index=prefix_values, columns=reuse_values)
    for prefix in prefix_values:
        for reuse in reuse_values:
            cell = warm[
                (warm["target_prefix_tokens"] == prefix)
                & (np.isclose(warm["reuse_ratio_target"], reuse))
            ]
            if not cell.empty:
                values.loc[prefix, reuse] = float(cell["hit_tokens"].median())

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    vmax = max(float(values.to_numpy().max()), 1.0)
    image = ax.imshow(values.to_numpy(), cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(reuse_values)))
    ax.set_xticklabels([f"{int(reuse * 100)}%" for reuse in reuse_values])
    ax.set_yticks(np.arange(len(prefix_values)))
    ax.set_yticklabels([str(prefix) for prefix in prefix_values])
    ax.set_xlabel("Reuse ratio target")
    ax.set_ylabel("Prefix length target (tokens)")
    ax.set_title("Median warm observed hit tokens in 5x5 matrix")
    for row_index, prefix in enumerate(prefix_values):
        for col_index, reuse in enumerate(reuse_values):
            value = values.loc[prefix, reuse]
            color = "white" if value > vmax * 0.58 else "#111827"
            ax.text(col_index, row_index, f"{value:.0f}", ha="center", va="center", color=color)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Median warm hit tokens")
    fig.tight_layout()
    fig.savefig(output_dir / "observed_hit_tokens_heatmap.png", dpi=180)
    fig.savefig(output_dir / "observed_hit_tokens_heatmap.pdf")
    plt.close(fig)


def save_pressure_sweep_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    ok = ok_rows(summary)
    revisit = ok[(ok["experiment_type"] == "cache_pressure") & (ok["phase"] == "revisit")].copy()
    if revisit.empty:
        return
    grouped = (
        revisit.groupby("pressure_level", as_index=False)
        .agg(
            median_ttft=("ttft_ms", "median"),
            median_hit_tokens=("hit_tokens", "median"),
            eviction_count=("block_evict_count_in_window", "sum"),
            requests=("request_id", "count"),
        )
        .sort_values("pressure_level")
    )
    fig, ax1 = plt.subplots(figsize=(8.8, 5.5))
    x = np.arange(len(grouped))
    bars = ax1.bar(x, grouped["median_ttft"], width=0.58, color="#93c5fd", edgecolor="#2563eb")
    ax1.set_ylabel("Revisit median TTFT (ms)")
    ax1.set_xlabel("Prefix group count")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(level)) for level in grouped["pressure_level"]])
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, grouped["median_hit_tokens"], color="#047857", marker="o", linewidth=2.4)
    ax2.set_ylabel("Revisit median hit tokens")
    ax1.set_title("Capacity pressure reduces revisit cache hits")
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
    fig.tight_layout()
    fig.savefig(output_dir / "cache_pressure_sweep_summary.png", dpi=180)
    fig.savefig(output_dir / "cache_pressure_sweep_summary.pdf")
    plt.close(fig)


def save_pressure_phase_breakdown(summary: pd.DataFrame, output_dir: Path) -> None:
    ok = ok_rows(summary)
    pressure = ok[ok["experiment_type"] == "cache_pressure"].copy()
    if pressure.empty:
        return
    levels = sorted(int(level) for level in pressure["pressure_level"].dropna().unique())
    if not levels:
        return
    grouped = (
        pressure.groupby(["pressure_level", "phase"], as_index=False)
        .agg(median_ttft=("ttft_ms", "median"), hit_rate=("cache_hit", "mean"))
    )
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True, height_ratios=[2.2, 1.2])
    x = np.arange(len(PHASE_ORDER))
    width = min(0.16, 0.78 / len(levels))
    offsets = (np.arange(len(levels)) - (len(levels) - 1) / 2) * width
    colors = plt.cm.viridis(np.linspace(0.18, 0.82, len(levels)))
    for level, offset, color in zip(levels, offsets, colors):
        rows = grouped[grouped["pressure_level"] == level]
        ttft = [rows.loc[rows["phase"] == phase, "median_ttft"].median() for phase in PHASE_ORDER]
        hit_rate = [rows.loc[rows["phase"] == phase, "hit_rate"].median() for phase in PHASE_ORDER]
        ax1.bar(x + offset, ttft, width=width, color=color, alpha=0.88, label=f"{level} groups")
        ax2.plot(x, hit_rate, color=color, marker="o", linewidth=2, label=f"{level} groups")
    ax1.set_ylabel("Median TTFT (ms)")
    ax1.set_title("Pressure phases expose revisit degradation")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(ncol=min(3, len(levels)), fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Hit rate")
    ax2.set_xticks(x)
    ax2.set_xticklabels(PHASE_ORDER)
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "cache_pressure_phase_breakdown.png", dpi=180)
    fig.savefig(output_dir / "cache_pressure_phase_breakdown.pdf")
    plt.close(fig)


def save_cache_pressure_timeline(summary: pd.DataFrame, output_dir: Path) -> None:
    pressure = ok_rows(summary)
    pressure = pressure[pressure["experiment_type"] == "cache_pressure"].copy()
    if pressure.empty:
        return
    pressure["phase_order"] = pressure["phase"].map({phase: i for i, phase in enumerate(PHASE_ORDER)}).fillna(99)
    pressure = pressure.sort_values(["pressure_level", "request_index"]).reset_index(drop=True)
    pressure["plot_index"] = np.arange(len(pressure))
    fig, ax = plt.subplots(figsize=(12, 5.8))

    for _, block in pressure.groupby(["pressure_level", "phase"], sort=False):
        start = float(block["plot_index"].min()) - 0.5
        end = float(block["plot_index"].max()) + 0.5
        phase = str(block["phase"].iloc[0])
        ax.axvspan(start, end, color=PHASE_COLORS.get(phase, "#e5e7eb"), alpha=0.06)

    for phase in PHASE_ORDER:
        phase_df = pressure[pressure["phase"] == phase]
        if phase_df.empty:
            continue
        ax.scatter(
            phase_df["plot_index"],
            phase_df["ttft_ms"],
            label=phase,
            color=PHASE_COLORS[phase],
            alpha=0.78,
            s=22 if phase == "filler" else 34,
        )

    for level, block in pressure.groupby("pressure_level"):
        x = float(block["plot_index"].min()) - 0.5
        ax.axvline(x, color="#d1d5db", linewidth=0.8)
        ax.text(x + 2, ax.get_ylim()[1], f"{int(level)} groups", va="top", fontsize=8, color="#4b5563")

    evict_rows = pressure[pressure["block_evict_count_in_window"] > 0]
    if not evict_rows.empty:
        ymin, ymax = ax.get_ylim()
        rug_y = ymin + (ymax - ymin) * 0.025
        ax.scatter(
            evict_rows["plot_index"],
            np.full(len(evict_rows), rug_y),
            marker="|",
            color="#dc2626",
            s=85,
            linewidths=1.1,
            label="eviction window",
        )
    ax.set_xlabel("Pressure workload request order")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Cache pressure timeline debug view")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncol=4)
    fig.tight_layout()
    fig.savefig(output_dir / "cache_pressure_timeline.png", dpi=180)
    fig.savefig(output_dir / "cache_pressure_timeline.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["smoke", "formal"],
        default="smoke",
        help="Default input/output path set to use when explicit paths are omitted.",
    )
    parser.add_argument(
        "--events",
        type=Path,
        action="append",
        default=None,
        help="Observer JSONL event log. Can be passed multiple times.",
    )
    parser.add_argument(
        "--latency-csv",
        type=Path,
        action="append",
        default=None,
        help="Latency CSV input. Can be passed multiple times.",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_paths, latency_paths, summary_output, figure_dir = resolve_default_paths(args)
    events = read_events(event_paths)
    latency = read_latency(latency_paths)
    summary = build_summary(latency, events)

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)

    save_ttft_vs_hit_tokens(summary, figure_dir)
    save_cache_hit_by_scenario(summary, figure_dir, "baseline")
    save_cache_hit_by_scenario(summary, figure_dir, "pressure")
    save_hit_tokens_heatmap(summary, figure_dir)
    save_pressure_sweep_summary(summary, figure_dir)
    save_pressure_phase_breakdown(summary, figure_dir)
    save_cache_pressure_timeline(summary, figure_dir)

    ok = summary[summary["status"] == "ok"]
    join_rate = float(ok["event_joined"].mean()) if not ok.empty else 0.0
    print(f"Wrote cache event summary to {summary_output}")
    print(f"Wrote figures to {figure_dir}")
    print(f"Joined request event rate: {join_rate:.2%} ({int(ok['event_joined'].sum())}/{len(ok)})")


if __name__ == "__main__":
    main()
