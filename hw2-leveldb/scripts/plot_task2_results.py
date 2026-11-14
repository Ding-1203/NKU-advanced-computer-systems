#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path.home() / "advanced-computer-systems" / "hw2-leveldb"

SUMMARY_CSV = ROOT / "data" / "task2_raw" / "task2_latency_summary.csv"
TIMESERIES_CSV = ROOT / "data" / "task2_raw" / "task2_latency_timeseries.csv"
FIG_DIR = ROOT / "results" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(SUMMARY_CSV)
ts = pd.read_csv(TIMESERIES_CSV)

# Convert window_id to approximate operation progress.
# Each window corresponds to one row in timeseries.
ts["window_id"] = ts["window_id"].astype(int)

def save_fig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

# ==================================================
# Figure 1: P99 / P99.9 / Max latency over windows
# ==================================================
plt.figure(figsize=(10, 5))

plt.plot(ts["window_id"], ts["p99_us"], marker="o", markersize=3, label="P99")
plt.plot(ts["window_id"], ts["p999_us"], marker="o", markersize=3, label="P99.9")
plt.plot(ts["window_id"], ts["max_us"], marker="o", markersize=3, label="Max")

plt.title("Task 2 Write Latency Over Time")
plt.xlabel("Window ID (each window = 100,000 writes)")
plt.ylabel("Latency (us)")
plt.yscale("log")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()

save_fig("fig_task2_latency_timeseries.png")

# ==================================================
# Figure 2: Average / P50 / P90 / P99 / P99.9 / Max
# ==================================================
row = summary.iloc[0]

labels = ["Average", "P50", "P90", "P99", "P99.9", "Max"]
values = [
    row["avg_us"],
    row["p50_us"],
    row["p90_us"],
    row["p99_us"],
    row["p999_us"],
    row["max_us"],
]

plt.figure(figsize=(8, 5))
plt.bar(labels, values)

plt.title("Task 2 Write Latency Percentiles")
plt.xlabel("Metric")
plt.ylabel("Latency (us)")
plt.yscale("log")
plt.grid(axis="y", linestyle="--", alpha=0.5)

save_fig("fig_task2_latency_percentiles.png")

# ==================================================
# Figure 3: Average vs P50 latency over windows
# ==================================================
plt.figure(figsize=(10, 5))

plt.plot(ts["window_id"], ts["avg_us"], marker="o", markersize=3, label="Average")
plt.plot(ts["window_id"], ts["p50_us"], marker="o", markersize=3, label="P50")

plt.title("Task 2 Average and Median Write Latency Over Time")
plt.xlabel("Window ID (each window = 100,000 writes)")
plt.ylabel("Latency (us)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()

save_fig("fig_task2_avg_p50_timeseries.png")

# ==================================================
# Figure 4: Operations per window
# ==================================================
plt.figure(figsize=(10, 4))

plt.bar(ts["window_id"], ts["ops"])

plt.title("Task 2 Operations Per Window")
plt.xlabel("Window ID")
plt.ylabel("Operations")
plt.grid(axis="y", linestyle="--", alpha=0.5)

save_fig("fig_task2_ops_per_window.png")

print("All Task 2 figures generated.")
