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

VERSION="${LMCACHE_VERSION:-0.5.0}"
FIND_LINKS_TAG="${LMCACHE_FIND_LINKS_TAG:-v${VERSION}-cu129}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="logs/lmcache_install/dry_run_${STAMP}"
mkdir -p "${OUT_DIR}"

scripts/freeze_stage3_env.sh "${OUT_DIR}/before_env" >/dev/null

CMD=(
  uv pip install --dry-run "lmcache==${VERSION}"
  --constraints configs/lmcache_constraints_cu129.txt
  --extra-index-url https://download.pytorch.org/whl/cu129
  --find-links "https://github.com/LMCache/LMCache/releases/expanded_assets/${FIND_LINKS_TAG}"
  --index-strategy unsafe-best-match
)

printf '%q ' "${CMD[@]}" > "${OUT_DIR}/dry_run_command.txt"
echo >> "${OUT_DIR}/dry_run_command.txt"
set +e
"${CMD[@]}" 2>&1 | tee "${OUT_DIR}/dry_run_output.txt"
STATUS=${PIPESTATUS[0]}
set -e
if [[ ${STATUS} -ne 0 ]]; then
  echo "LMCache dry-run failed; no install attempted. See ${OUT_DIR}/dry_run_output.txt" >&2
  exit ${STATUS}
fi

if rg -i "Would install .*\b(torch|vllm|triton)\b|Would uninstall|Uninstalling" "${OUT_DIR}/dry_run_output.txt" >/dev/null; then
  echo "Unsafe resolver result: dry-run wants to touch critical packages, uninstall packages, or downgrade the validated stack." >&2
  echo "No install attempted. See ${OUT_DIR}/dry_run_output.txt" >&2
  exit 20
fi

echo "LMCache dry-run passed safety gate. Report: ${OUT_DIR}"
