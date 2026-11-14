#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path.home() / "advanced-computer-systems" / "hw2-leveldb"

CSV_PATH = ROOT / "data" / "task3_raw" / "task3_cache_results.csv"
FIG_DIR = ROOT / "results" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

workload_order = ["point_get", "range_scan", "mixed_80r20w"]
mode_order = ["block_cache", "kv_cache"]

df["workload"] = pd.Categorical(
    df["workload"],
    categories=workload_order,
    ordered=True
)

df["mode"] = pd.Categorical(
    df["mode"],
    categories=mode_order,
    ordered=True
)

df = df.sort_values(["workload", "mode"])

def save_fig(filename):
    path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def annotate_bars(ax, fmt="{:.1f}", offset=3, suffix=""):
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height) + suffix,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

# ==================================================
# Figure 1: QPS comparison
# ==================================================
qps_pivot = df.pivot(
    index="workload",
    columns="mode",
    values="qps"
)

plt.figure(figsize=(8, 5))
ax = qps_pivot.plot(kind="bar", ax=plt.gca())

plt.title("Task 3 QPS Comparison")
plt.xlabel("Workload")
plt.ylabel("QPS")
plt.xticks(rotation=0)
plt.legend(title="Cache Mode")
plt.grid(axis="y", linestyle="--", alpha=0.5)

annotate_bars(ax, fmt="{:.0f}")
save_fig("fig_task3_qps_comparison.png")

# ==================================================
# Figure 2: KV Cache hit rate
# ==================================================
kv_df = df[df["mode"] == "kv_cache"].copy()
kv_df = kv_df.sort_values("workload")

plt.figure(figsize=(8, 5))
ax = plt.gca()
bars = ax.bar(
    kv_df["workload"].astype(str),
    kv_df["hit_rate"] * 100,
)

plt.title("Task 3 KV Cache Hit Rate")
plt.xlabel("Workload")
plt.ylabel("Hit Rate (%)")

y_max = (kv_df["hit_rate"] * 100).max()
plt.ylim(0, min(110, y_max * 1.08 + 5))
plt.grid(axis="y", linestyle="--", alpha=0.5)

annotate_bars(ax, fmt="{:.1f}", suffix="%")
save_fig("fig_task3_kv_hit_rate.png")

print("Task 3 figures generated.")
