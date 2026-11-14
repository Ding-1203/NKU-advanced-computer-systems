#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_LMCACHE_INSTALL:-0}" != "1" ]]; then
  echo "Refusing to install. Re-run with ALLOW_LMCACHE_INSTALL=1 after reviewing dry-run output." >&2
  exit 2
fi

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

VERSION="${LMCACHE_VERSION:-0.5.0}"
FIND_LINKS_TAG="${LMCACHE_FIND_LINKS_TAG:-v${VERSION}-cu129}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="logs/lmcache_install/install_${STAMP}"
mkdir -p "${OUT_DIR}"

scripts/freeze_stage3_env.sh "${OUT_DIR}/before_env" >/dev/null
scripts/check_lmcache_install_plan.sh | tee "${OUT_DIR}/guard_dry_run.txt"

uv pip install "lmcache==${VERSION}" \
  --constraints configs/lmcache_constraints_cu129.txt \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --find-links "https://github.com/LMCache/LMCache/releases/expanded_assets/${FIND_LINKS_TAG}" \
  --index-strategy unsafe-best-match 2>&1 | tee "${OUT_DIR}/install_output.txt"

scripts/freeze_stage3_env.sh "${OUT_DIR}/after_env" >/dev/null
python - <<'PY' "${OUT_DIR}/before_env/runtime_versions.txt" "${OUT_DIR}/after_env/runtime_versions.txt"
import sys
before = dict(line.strip().split('=', 1) if '==' not in line else line.strip().split('==', 1) for line in open(sys.argv[1]) if '=' in line)
after = dict(line.strip().split('=', 1) if '==' not in line else line.strip().split('==', 1) for line in open(sys.argv[2]) if '=' in line)
for key in ['torch', 'vllm', 'triton']:
    if before.get(key) != after.get(key):
        raise SystemExit(f'critical package changed: {key}: {before.get(key)} -> {after.get(key)}')
PY

python -c "import lmcache; print('lmcache import ok')"
lmcache --help >/dev/null
echo "LMCache guarded install completed. Report: ${OUT_DIR}"
