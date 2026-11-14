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

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${1:-logs/env_freeze/stage3_${STAMP}}"
mkdir -p "${OUT_DIR}"

python - <<'PY' > "${OUT_DIR}/runtime_versions.txt"
import importlib.metadata as md
import platform
import sys
print(f"python={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
for name in ["torch", "vllm", "triton", "transformers", "datasets", "flashinfer-python", "flashinfer-cubin", "lmcache"]:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=MISSING")
try:
    import torch
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"torch_probe_error={exc}")
PY

python -m pip freeze > "${OUT_DIR}/pip_freeze.txt"
uv pip freeze > "${OUT_DIR}/uv_freeze.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_DIR}/nvidia_smi.txt" || true
fi
cp configs/lmcache_constraints_cu129.txt "${OUT_DIR}/lmcache_constraints_cu129.txt"
echo "${OUT_DIR}"
