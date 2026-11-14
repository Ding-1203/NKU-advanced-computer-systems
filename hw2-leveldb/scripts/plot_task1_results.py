#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path.home() / "advanced-computer-systems" / "hw2-leveldb"
CSV_PATH = ROOT / "results" / "tables" / "task1_summary.csv"
FIG_DIR = ROOT / "results" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# 只保留正式测试项
df = df[df["test_name"].isin([
    "fillseq",
    "fillrandom",
    "readrandom",
    "readseq",
    "readmissing"
])]

def save_fig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

def order_param(df_part, order):
    df_part = df_part.copy()
    df_part["param_value"] = pd.Categorical(
        df_part["param_value"],
        categories=order,
        ordered=True
    )
    return df_part.sort_values("param_value")

# ==================================================
# Figure 1: write_buffer_size write throughput
# ==================================================
wb = df[df["param_type"] == "write_buffer_size"]
wb = order_param(wb, ["4mb", "16mb", "64mb"])

wb_write = wb[wb["test_name"].isin(["fillseq", "fillrandom"])]

pivot = wb_write.pivot(
    index="param_value",
    columns="test_name",
    values="mb_per_sec"
)

plt.figure(figsize=(7, 4.5))
ax = plt.gca()
pivot.plot(kind="bar", ax=ax)
plt.title("Effect of write_buffer_size on Write Throughput")
plt.xlabel("write_buffer_size")
plt.ylabel("Throughput (MB/s)")
plt.xticks(rotation=0)
plt.legend(title="Benchmark")
plt.grid(axis="y", linestyle="--", alpha=0.5)
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f')
save_fig("fig_write_buffer_write_throughput.png")

# ==================================================
# Figure 2: write_buffer_size read latency
# ==================================================
wb_read = wb[wb["test_name"].isin(["readrandom", "readseq"])]

plt.figure(figsize=(7, 4.5))
for test_name in ["readrandom", "readseq"]:
    data = wb_read[wb_read["test_name"] == test_name]
    plt.plot(
        data["param_value"],
        data["micros_per_op"],
        marker="o",
        label=test_name
    )
    for i, (x, y) in enumerate(zip(data["param_value"], data["micros_per_op"])):
        plt.text(x, y, f'{y:.1f}', ha='center', va='bottom')
plt.title("Effect of write_buffer_size on Read Latency")
plt.xlabel("write_buffer_size")
plt.ylabel("Latency (us/op)")
plt.legend(title="Benchmark")
plt.grid(axis="y", linestyle="--", alpha=0.5)
save_fig("fig_write_buffer_read_latency.png")

# ==================================================
# Figure 3: block_size write throughput
# ==================================================
bs = df[df["param_type"] == "block_size"]
bs = order_param(bs, ["4kb", "16kb", "64kb"])

bs_write = bs[bs["test_name"].isin(["fillseq", "fillrandom"])]

pivot = bs_write.pivot(
    index="param_value",
    columns="test_name",
    values="mb_per_sec"
)

plt.figure(figsize=(7, 4.5))
ax = plt.gca()
pivot.plot(kind="bar", ax=ax)
plt.title("Effect of block_size on Write Throughput")
plt.xlabel("block_size")
plt.ylabel("Throughput (MB/s)")
plt.xticks(rotation=0)
plt.legend(title="Benchmark")
plt.grid(axis="y", linestyle="--", alpha=0.5)
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f')
save_fig("fig_block_size_write_throughput.png")


# ==================================================
# Figure 4: block_size read latency
# ==================================================
bs_read = bs[bs["test_name"].isin(["readrandom", "readseq"])]

plt.figure(figsize=(7, 4.5))
for test_name in ["readrandom", "readseq"]:
    data = bs_read[bs_read["test_name"] == test_name]
    plt.plot(
        data["param_value"],
        data["micros_per_op"],
        marker="o",
        label=test_name
    )
    for x, y in zip(data["param_value"], data["micros_per_op"]):
        plt.text(x, y, f'{y:.1f}', ha='center', va='bottom')

plt.title("Effect of block_size on Read Latency")
plt.xlabel("block_size")
plt.ylabel("Latency (us/op)")
plt.legend(title="Benchmark")
plt.grid(axis="y", linestyle="--", alpha=0.5)

max_val = bs_read["micros_per_op"].max()
plt.ylim(bottom=0, top=max_val * 1.1)

save_fig("fig_block_size_read_latency.png")


# ==================================================
# Figure 5: Bloom Filter read latency
# ==================================================
bf = df[df["param_type"] == "bloom_filter"]
bf = order_param(bf, ["off", "on"])

bf_read = bf[bf["test_name"].isin(["readrandom", "readmissing"])]

pivot = bf_read.pivot(
    index="param_value",
    columns="test_name",
    values="micros_per_op"
)

plt.figure(figsize=(7, 4.5))
ax = plt.gca()
pivot.plot(kind="bar", ax=ax)
plt.title("Effect of Bloom Filter on Read Latency")
plt.xlabel("Bloom Filter")
plt.ylabel("Latency (us/op)")
plt.xticks(rotation=0)
plt.legend(title="Benchmark")
plt.grid(axis="y", linestyle="--", alpha=0.5)
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f')
save_fig("fig_bloom_read_latency.png")

print("All Task 1 figures generated.")
