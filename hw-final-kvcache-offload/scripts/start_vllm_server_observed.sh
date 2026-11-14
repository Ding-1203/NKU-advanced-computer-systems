#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ -f "env.sh" ]]; then
  # shellcheck disable=SC1091
  source "env.sh"
fi

if [[ -z "${QWEN05B_PATH:-}" || ! -d "${QWEN05B_PATH}" ]]; then
  echo "QWEN05B_PATH is empty or invalid. Run: source env.sh" >&2
  exit 1
fi

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_SERVED_MODEL="${VLLM_SERVED_MODEL:-qwen05b}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-16}"
VLLM_LOG_LEVEL="${VLLM_LOG_LEVEL:-info}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_TMPDIR="${VLLM_TMPDIR:-/tmp/vllm-${USER:-user}}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export VLLM_CACHE_OBSERVER=1
export VLLM_CACHE_OBSERVER_LOG="${VLLM_CACHE_OBSERVER_LOG:-${PROJECT_DIR}/logs/cache_observation/cache_events.jsonl}"

mkdir -p "${VLLM_TMPDIR}" "$(dirname "${VLLM_CACHE_OBSERVER_LOG}")"
if [[ "${VLLM_CACHE_OBSERVER_RESET_LOG:-0}" == "1" ]]; then
  : > "${VLLM_CACHE_OBSERVER_LOG}"
fi

export TMPDIR="${VLLM_TMPDIR}"
export TEMP="${VLLM_TMPDIR}"
export TMP="${VLLM_TMPDIR}"

echo "Starting observed vLLM with Prefix Cache ON"
echo "  model: ${QWEN05B_PATH}"
echo "  served model: ${VLLM_SERVED_MODEL}"
echo "  endpoint: http://${VLLM_HOST}:${VLLM_PORT}/v1"
echo "  observer log: ${VLLM_CACHE_OBSERVER_LOG}"
echo "  tmpdir: ${VLLM_TMPDIR}"

exec vllm serve "${QWEN05B_PATH}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --served-model-name "${VLLM_SERVED_MODEL}" \
  --dtype "${VLLM_DTYPE}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  --block-size "${VLLM_BLOCK_SIZE}" \
  --enforce-eager \
  --generation-config vllm \
  --disable-uvicorn-access-log \
  --uvicorn-log-level "${VLLM_LOG_LEVEL}" \
  --enable-prefix-caching \
  ${VLLM_EXTRA_ARGS}
