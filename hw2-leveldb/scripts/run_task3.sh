#!/bin/bash

set -e

ROOT="$HOME/advanced-computer-systems/hw2-leveldb"

SRC_MAIN="$ROOT/src/task3_cache_bench.cpp"
SRC_CACHE="$ROOT/src/kv_cache.cpp"
SRC_ZIPF="$ROOT/src/zipf.cpp"

BIN="$ROOT/bin/task3_cache_bench"

LEVELDB_INCLUDE="$ROOT/third_party/leveldb/include"
LEVELDB_LIB="$ROOT/third_party/leveldb/build"

DB_PATH="$ROOT/db/task3/cache_db"
OUT_DIR="$ROOT/data/task3_raw"
OUT_CSV="$OUT_DIR/task3_cache_results.csv"
DROP_CACHE="$ROOT/scripts/drop_cache.sh"

mkdir -p "$ROOT/bin"
mkdir -p "$ROOT/db/task3"
mkdir -p "$OUT_DIR"

MODE="${1:-debug}"

if [ "$MODE" = "debug" ]; then
    NUM_KEYS=200000
    VALUE_SIZE=1024
    POINT_OPS=200000
    SCAN_OPS=20000
    SCAN_LENGTH=100
    MIXED_OPS=200000
elif [ "$MODE" = "formal" ]; then
    NUM_KEYS=2000000
    VALUE_SIZE=1024
    POINT_OPS=1000000
    SCAN_OPS=100000
    SCAN_LENGTH=100
    MIXED_OPS=1000000
else
    echo "Usage: $0 [debug|formal]"
    exit 1
fi

echo "=================================================="
echo "Task 3 Cache Benchmark"
echo "Mode        : $MODE"
echo "NUM_KEYS    : $NUM_KEYS"
echo "VALUE_SIZE  : $VALUE_SIZE"
echo "POINT_OPS   : $POINT_OPS"
echo "SCAN_OPS    : $SCAN_OPS"
echo "SCAN_LENGTH : $SCAN_LENGTH"
echo "MIXED_OPS   : $MIXED_OPS"
echo "DB_PATH     : $DB_PATH"
echo "OUT_CSV     : $OUT_CSV"
echo "=================================================="

echo
echo "Compiling..."

g++ -std=c++17 -O2 \
    "$SRC_MAIN" \
    "$SRC_CACHE" \
    "$SRC_ZIPF" \
    -I"$ROOT/src" \
    -I"$LEVELDB_INCLUDE" \
    -L"$LEVELDB_LIB" \
    -lleveldb \
    -lpthread \
    -o "$BIN"

echo "Compile finished: $BIN"

echo
echo "Cleaning previous CSV..."
rm -f "$OUT_CSV"

echo
echo "Cleaning old database..."
rm -rf "$DB_PATH"

drop_cache() {
    if [ -x "$DROP_CACHE" ]; then
        "$DROP_CACHE"
    else
        sync
        echo 3 | sudo tee /proc/sys/vm/drop_caches
    fi
}

echo
echo "Building database only..."
drop_cache

"$BIN" \
    "$DB_PATH" \
    "$OUT_CSV" \
    "build_only" \
    "$NUM_KEYS" \
    "$VALUE_SIZE" \
    "$POINT_OPS" \
    "$SCAN_OPS" \
    "$SCAN_LENGTH" \
    "$MIXED_OPS" \
    2>&1 | tee "$OUT_DIR/task3_build_db.log"

echo
echo "Sync and drop cache after build..."
sync
drop_cache

echo
echo "Running Block Cache mode..."

"$BIN" \
    "$DB_PATH" \
    "$OUT_CSV" \
    "block_cache" \
    "$NUM_KEYS" \
    "$VALUE_SIZE" \
    "$POINT_OPS" \
    "$SCAN_OPS" \
    "$SCAN_LENGTH" \
    "$MIXED_OPS" \
    2>&1 | tee "$OUT_DIR/task3_block_cache.log"

echo
echo "Sync and drop cache before KV Cache mode..."
sync
drop_cache

echo
echo "Running KV Cache mode..."

"$BIN" \
    "$DB_PATH" \
    "$OUT_CSV" \
    "kv_cache" \
    "$NUM_KEYS" \
    "$VALUE_SIZE" \
    "$POINT_OPS" \
    "$SCAN_OPS" \
    "$SCAN_LENGTH" \
    "$MIXED_OPS" \
    2>&1 | tee "$OUT_DIR/task3_kv_cache.log"

echo
echo "Task 3 finished."
echo "Results saved to:"
echo "$OUT_CSV"
