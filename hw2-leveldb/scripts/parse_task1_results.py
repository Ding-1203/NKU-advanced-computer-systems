#!/usr/bin/env python3
import re
import csv
from pathlib import Path

ROOT = Path.home() / "advanced-computer-systems" / "hw2-leveldb"
RAW_DIR = ROOT / "data" / "task1_raw"
OUT_DIR = ROOT / "results" / "tables"
OUT_CSV = OUT_DIR / "task1_summary.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = [
    "fillseq",
    "fillrandom",
    "readrandom",
    "readseq",
    "readmissing",
]

BENCH_RE = re.compile(
    r"^(fillseq|fillrandom|readrandom|readseq|readmissing)\s*:\s*([-\d.]+)\s+micros/op(?:;\s*([-\d.]+)\s+MB/s)?"
)

AVG_RE = re.compile(
    r"Count:\s*(\d+)\s+Average:\s*([-\d.]+)\s+StdDev:\s*([-\d.]+)"
)

MIN_MED_MAX_RE = re.compile(
    r"Min:\s*([-\d.]+)\s+Median:\s*([-\d.]+)\s+Max:\s*([-\d.]+)"
)

def parse_filename(path: Path):
    name = path.stem

    if name.endswith("_build_read_db"):
        exp_name = name[:-len("_build_read_db")]
        return exp_name, "build_read_db"

    tests = [
        "readmissing",
        "readrandom",
        "fillrandom",
        "fillseq",
        "readseq",
    ]

    for test in tests:
        suffix = "_" + test
        if name.endswith(suffix):
            exp_name = name[:-len(suffix)]
            return exp_name, test

    return name, "unknown"

def classify_experiment(exp_name: str):
    if exp_name.startswith("write_buffer_"):
        return "write_buffer_size", exp_name.replace("write_buffer_", "")
    if exp_name.startswith("block_"):
        return "block_size", exp_name.replace("block_", "")
    if exp_name.startswith("bloom_"):
        return "bloom_filter", exp_name.replace("bloom_", "")
    return "unknown", exp_name

def parse_file(path: Path):
    exp_name, test_name = parse_filename(path)
    param_type, param_value = classify_experiment(exp_name)

    rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        m = BENCH_RE.match(line)
        if not m:
            continue

        benchmark = m.group(1)
        micros_op = float(m.group(2))
        mb_s = float(m.group(3)) if m.group(3) is not None else ""

        row = {
            "file": path.name,
            "experiment": exp_name,
            "param_type": param_type,
            "param_value": param_value,
            "test_name": test_name,
            "benchmark": benchmark,
            "micros_per_op": micros_op,
            "mb_per_sec": mb_s,
            "count": "",
            "average_us": "",
            "stddev_us": "",
            "min_us": "",
            "median_us": "",
            "max_us": "",
        }

        for j in range(i + 1, min(i + 8, len(lines))):
            avg_match = AVG_RE.search(lines[j])
            if avg_match:
                row["count"] = int(avg_match.group(1))
                row["average_us"] = float(avg_match.group(2))
                row["stddev_us"] = float(avg_match.group(3))

            min_med_max_match = MIN_MED_MAX_RE.search(lines[j])
            if min_med_max_match:
                row["min_us"] = float(min_med_max_match.group(1))
                row["median_us"] = float(min_med_max_match.group(2))
                row["max_us"] = float(min_med_max_match.group(3))

        # 主 CSV 只保留正式测试项
        # build_read_db 只是为 readrandom/readseq/readmissing 准备数据库，不进入主分析
        if test_name == "build_read_db":
            continue

        # 一个文件可能包含多个 benchmark，这里只保留和文件名测试类型一致的结果
        if benchmark == test_name:
            rows.append(row)

    return rows

def main():
    all_rows = []

    txt_files = sorted(RAW_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No txt files found in {RAW_DIR}")
        return

    for path in txt_files:
        rows = parse_file(path)
        all_rows.extend(rows)

    fieldnames = [
        "file",
        "experiment",
        "param_type",
        "param_value",
        "test_name",
        "benchmark",
        "micros_per_op",
        "mb_per_sec",
        "count",
        "average_us",
        "stddev_us",
        "min_us",
        "median_us",
        "max_us",
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Parsed {len(all_rows)} benchmark rows.")
    print(f"CSV saved to: {OUT_CSV}")

    print()
    print("Tip:")
    print("  build_read_db files are skipped.")
    print("  Main benchmarks kept: fillseq, fillrandom, readrandom, readseq, readmissing")

if __name__ == "__main__":
    main()
