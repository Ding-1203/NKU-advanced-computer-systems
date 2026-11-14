#!/bin/bash

set -e

ROOT="$HOME/advanced-computer-systems/hw2-leveldb"

SRC="$ROOT/src/task2_latency_bench.cpp"
BIN="$ROOT/bin/task2_latency_bench"

LEVELDB_INCLUDE="$ROOT/third_party/leveldb/include"
LEVELDB_LIB="$ROOT/third_party/leveldb/build"

DB_PATH="$ROOT/db/task2/latency_db"
OUT_DIR="$ROOT/data/task2_raw"
SUMMARY_CSV="$OUT_DIR/task2_latency_summary.csv"
TIMESERIES_CSV="$OUT_DIR/task2_latency_timeseries.csv"
DROP_CACHE="$ROOT/scripts/drop_cache.sh"

mkdir -p "$ROOT/bin"
mkdir -p "$ROOT/db/task2"
mkdir -p "$OUT_DIR"

# Debug mode: about 3GB
DEBUG_NUM_OPS=3000000

# Formal mode: about 22GB
FORMAL_NUM_OPS=22000000

VALUE_SIZE=1024

MODE="${1:-debug}"

if [ "$MODE" = "formal" ]; then
    NUM_OPS=$FORMAL_NUM_OPS
elif [ "$MODE" = "debug" ]; then
    NUM_OPS=$DEBUG_NUM_OPS
else
    echo "Usage: $0 [debug|formal]"
    exit 1
fi

echo "=================================================="
echo "Task 2 Latency Benchmark"
echo "Mode       : $MODE"
echo "NUM_OPS    : $NUM_OPS"
echo "VALUE_SIZE : $VALUE_SIZE bytes"
echo "Data size  : approximately $((NUM_OPS * VALUE_SIZE / 1024 / 1024)) MB"
echo "DB Path    : $DB_PATH"
echo "=================================================="

echo
echo "Compiling task2_latency_bench.cpp..."

g++ -std=c++17 -O2 \
    "$SRC" \
    -I"$LEVELDB_INCLUDE" \
    -L"$LEVELDB_LIB" \
    -lleveldb \
    -lpthread \
    -o "$BIN"

echo "Compile finished: $BIN"

echo
echo "Cleaning old database..."
rm -rf "$DB_PATH"

echo
echo "Dropping Linux Page Cache..."
if [ -x "$DROP_CACHE" ]; then
    "$DROP_CACHE"
else
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches
fi

echo
echo "Running benchmark..."
"$BIN" \
    "$DB_PATH" \
    "$NUM_OPS" \
    "$VALUE_SIZE" \
    "$SUMMARY_CSV" \
    "$TIMESERIES_CSV" \
    2>&1 | tee "$OUT_DIR/task2_run.log"

echo
echo "Task 2 finished."
echo "Summary:"
echo "$SUMMARY_CSV"
echo
echo "Timeseries:"
echo "$TIMESERIES_CSV"
