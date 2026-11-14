#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

WORKLOAD="${1:-data/workloads/coqa_final/coqa_final_pressure_g24.jsonl}"
STEM="$(basename "${WORKLOAD}" .jsonl)"
OUTPUT="${2:-logs/lmcache_real/${STEM}.csv}"
LOG_DIR="${LMCACHE_REAL_LOG_DIR:-logs/lmcache_real}"
SERVER_LOG="${LMCACHE_SERVER_LOG:-${LOG_DIR}/${STEM}_lmcache_server.log}"
VLLM_LOG="${LMCACHE_VLLM_LOG:-${LOG_DIR}/${STEM}_vllm.log}"
METRICS_OUT="${LMCACHE_METRICS_OUT:-${LOG_DIR}/${STEM}_metrics.prom}"

RUNNER_VENV="${RUNNER_VENV:-.venv}"
LMCACHE_PORT="${LMCACHE_PORT:-5555}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_SERVED_MODEL="${VLLM_SERVED_MODEL:-qwen05b}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-240}"
LMCACHE_BOOT_SLEEP_S="${LMCACHE_BOOT_SLEEP_S:-3}"
RUNNER_TIMEOUT="${RUNNER_TIMEOUT:-120}"
RUNNER_SLEEP_S="${RUNNER_SLEEP_S:-0.02}"

mkdir -p "$(dirname "${OUTPUT}")" "${LOG_DIR}"

lmcache_pid=""
vllm_pid=""

cleanup() {
  set +e
  if [[ -n "${vllm_pid}" ]] && kill -0 "${vllm_pid}" 2>/dev/null; then
    kill -INT "${vllm_pid}" 2>/dev/null
    wait "${vllm_pid}" 2>/dev/null
  fi
  if [[ -n "${lmcache_pid}" ]] && kill -0 "${lmcache_pid}" 2>/dev/null; then
    kill -INT "${lmcache_pid}" 2>/dev/null
    wait "${lmcache_pid}" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

scripts/start_lmcache_server.sh >"${SERVER_LOG}" 2>&1 &
lmcache_pid="$!"
sleep "${LMCACHE_BOOT_SLEEP_S}"

scripts/start_vllm_server_lmcache.sh >"${VLLM_LOG}" 2>&1 &
vllm_pid="$!"

deadline=$((SECONDS + READY_TIMEOUT_S))
until curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "${vllm_pid}" 2>/dev/null; then
    echo "vLLM exited before readiness. See ${VLLM_LOG}" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM readiness. See ${VLLM_LOG}" >&2
    exit 1
  fi
  sleep 2
done

# shellcheck disable=SC1091
source "${RUNNER_VENV}/bin/activate"
if [[ -f "env.sh" ]]; then
  # shellcheck disable=SC1091
  source "env.sh"
fi

python src/run_baseline_workload.py \
  --workload "${WORKLOAD}" \
  --output "${OUTPUT}" \
  --base-url "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
  --model "${VLLM_SERVED_MODEL}" \
  --timeout "${RUNNER_TIMEOUT}" \
  --wait \
  --sleep-s "${RUNNER_SLEEP_S}"

curl -fsS "http://127.0.0.1:8080/metrics" >"${METRICS_OUT}" 2>/dev/null || true

echo "LMCache pressure run complete"
echo "  workload: ${WORKLOAD}"
echo "  latency:  ${OUTPUT}"
echo "  metrics:  ${METRICS_OUT}"
echo "  logs:     ${SERVER_LOG}"
echo "            ${VLLM_LOG}"
