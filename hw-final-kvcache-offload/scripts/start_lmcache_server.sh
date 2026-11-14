#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

LMCACHE_VENV="${LMCACHE_VENV:-.venv-lmcache}"
if [[ -f "${LMCACHE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${LMCACHE_VENV}/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi
if [[ -f "env.sh" ]]; then
  # shellcheck disable=SC1091
  source "env.sh"
fi

LMCACHE_HOST="${LMCACHE_HOST:-localhost}"
LMCACHE_PORT="${LMCACHE_PORT:-5555}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-6}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-16}"
LMCACHE_EVICTION_POLICY="${LMCACHE_EVICTION_POLICY:-LRU}"
LMCACHE_TRANSFER_MODE="${LMCACHE_TRANSFER_MODE:-non_gpu}"

exec lmcache server \
  --host "${LMCACHE_HOST}" \
  --port "${LMCACHE_PORT}" \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
  --eviction-policy "${LMCACHE_EVICTION_POLICY}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --transfer-mode "${LMCACHE_TRANSFER_MODE}"
