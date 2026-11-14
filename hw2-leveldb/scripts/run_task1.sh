#!/bin/bash

set -e

ROOT="$HOME/advanced-computer-systems/hw2-leveldb"
BENCH="$ROOT/third_party/leveldb/build/db_bench"

DB_ROOT="$ROOT/db/task1"
OUT_ROOT="$ROOT/data/task1_raw"
DROP_CACHE="$ROOT/scripts/drop_cache.sh"

NUM=2100000
READS=1000000
VALUE_SIZE=1024

mkdir -p "$DB_ROOT"
mkdir -p "$OUT_ROOT"

drop_cache() {
    if [ -x "$DROP_CACHE" ]; then
        "$DROP_CACHE"
    else
        echo "Warning: $DROP_CACHE not found or not executable."
        sync
        echo 3 | sudo tee /proc/sys/vm/drop_caches
    fi
}

run_write_test() {
    EXP_NAME="$1"
    TEST_NAME="$2"
    BENCHMARK="$3"
    EXTRA_ARGS="$4"

    DB_PATH="$DB_ROOT/${EXP_NAME}_${TEST_NAME}"
    OUT_FILE="$OUT_ROOT/${EXP_NAME}_${TEST_NAME}.txt"

    echo "=================================================="
    echo "Write Test : $EXP_NAME / $TEST_NAME"
    echo "Benchmark  : $BENCHMARK"
    echo "DB Path    : $DB_PATH"
    echo "Output     : $OUT_FILE"
    echo "Args       : $EXTRA_ARGS"
    echo "=================================================="

    rm -rf "$DB_PATH"
    drop_cache

    "$BENCH" \
      --benchmarks="$BENCHMARK" \
      --db="$DB_PATH" \
      --num="$NUM" \
      --value_size="$VALUE_SIZE" \
      --histogram=1 \
      $EXTRA_ARGS \
      2>&1 | tee "$OUT_FILE"

    echo
}

build_read_db() {
    EXP_NAME="$1"
    EXTRA_ARGS="$2"

    DB_PATH="$DB_ROOT/${EXP_NAME}_read_db"
    OUT_FILE="$OUT_ROOT/${EXP_NAME}_build_read_db.txt"

    echo "=================================================="
    echo "Build Read DB : $EXP_NAME"
    echo "DB Path       : $DB_PATH"
    echo "Output        : $OUT_FILE"
    echo "Args          : $EXTRA_ARGS"
    echo "=================================================="

    rm -rf "$DB_PATH"
    drop_cache

    "$BENCH" \
      --benchmarks=fillrandom \
      --db="$DB_PATH" \
      --num="$NUM" \
      --value_size="$VALUE_SIZE" \
      --histogram=1 \
      $EXTRA_ARGS \
      2>&1 | tee "$OUT_FILE"

    echo
}

run_read_test() {
    EXP_NAME="$1"
    TEST_NAME="$2"
    BENCHMARK="$3"
    EXTRA_ARGS="$4"

    DB_PATH="$DB_ROOT/${EXP_NAME}_read_db"
    OUT_FILE="$OUT_ROOT/${EXP_NAME}_${TEST_NAME}.txt"

    echo "=================================================="
    echo "Read Test  : $EXP_NAME / $TEST_NAME"
    echo "Benchmark  : $BENCHMARK"
    echo "DB Path    : $DB_PATH"
    echo "Output     : $OUT_FILE"
    echo "Args       : $EXTRA_ARGS"
    echo "=================================================="

    if [ ! -d "$DB_PATH" ]; then
        echo "Error: read DB does not exist: $DB_PATH"
        exit 1
    fi

    drop_cache

    "$BENCH" \
      --benchmarks="$BENCHMARK" \
      --db="$DB_PATH" \
      --num="$NUM" \
      --reads="$READS" \
      --value_size="$VALUE_SIZE" \
      --use_existing_db=1 \
      --histogram=1 \
      $EXTRA_ARGS \
      2>&1 | tee "$OUT_FILE"

    echo
}

run_group() {
    EXP_NAME="$1"
    EXTRA_ARGS="$2"
    ENABLE_READMISSING="$3"

    echo
    echo "##################################################"
    echo "Running group: $EXP_NAME"
    echo "Args         : $EXTRA_ARGS"
    echo "##################################################"
    echo

    # 1. 顺序写
    run_write_test "$EXP_NAME" "fillseq" "fillseq" "$EXTRA_ARGS"

    # 2. 随机写
    run_write_test "$EXP_NAME" "fillrandom" "fillrandom" "$EXTRA_ARGS"

    # 3. 构建一个新的 DB 用于后续的读测试
    build_read_db "$EXP_NAME" "$EXTRA_ARGS"

    # 4. 随机读取现有 DB
    run_read_test "$EXP_NAME" "readrandom" "readrandom" "$EXTRA_ARGS"

    # 5. 顺序读取现有 DB
    run_read_test "$EXP_NAME" "readseq" "readseq" "$EXTRA_ARGS"

    # 6. 缺失键读取，作为验证 Bloom Filter 效果的补充测试
    if [ "$ENABLE_READMISSING" = "1" ]; then
        run_read_test "$EXP_NAME" "readmissing" "readmissing" "$EXTRA_ARGS"
    fi
}

echo "Task 1 benchmark started."
echo "NUM=$NUM"
echo "READS=$READS"
echo "VALUE_SIZE=$VALUE_SIZE"
echo "Raw data target is about: $((NUM * VALUE_SIZE / 1024 / 1024)) MB"
echo

# ==================================================
# Experiment 1: write_buffer_size
# ==================================================
run_group "write_buffer_4mb"  "--write_buffer_size=4194304"  "0"
run_group "write_buffer_16mb" "--write_buffer_size=16777216" "0"
run_group "write_buffer_64mb" "--write_buffer_size=67108864" "0"

# ==================================================
# Experiment 2: block_size
# ==================================================
run_group "block_4kb"  "--block_size=4096"  "0"
run_group "block_16kb" "--block_size=16384" "0"
run_group "block_64kb" "--block_size=65536" "0"

# ==================================================
# Experiment 3: Bloom Filter
# For Bloom Filter, readmissing is important.
# ==================================================
run_group "bloom_off" "--bloom_bits=0"  "1"
run_group "bloom_on"  "--bloom_bits=10" "1"

echo
echo "Task 1 benchmark finished."
echo "Raw results are saved in:"
echo "$OUT_ROOT"
