#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

LMCACHE_VENV="${LMCACHE_VENV:-.venv-lmcache}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VERSION="${LMCACHE_VERSION:-0.4.6}"
FIND_LINKS_TAG="${LMCACHE_FIND_LINKS_TAG:-v${VERSION}-cu129}"
VLLM_CU129_ARCHIVE="${VLLM_CU129_ARCHIVE:-}"

if [[ ! -d "${LMCACHE_VENV}" ]]; then
  uv venv "${LMCACHE_VENV}" --python "${PYTHON_VERSION}"
fi

# shellcheck disable=SC1091
source "${LMCACHE_VENV}/bin/activate"

uv pip install "lmcache==${VERSION}" "vllm==0.20.0" "torch==2.11.0+cu129" "triton==3.6.0" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --find-links "https://github.com/LMCache/LMCache/releases/expanded_assets/${FIND_LINKS_TAG}" \
  --index-strategy unsafe-best-match

python - <<'INNERPY'
from pathlib import Path
import os
import shutil
import site
import time

archive = os.environ.get("VLLM_CU129_ARCHIVE", "")
if not archive:
    root = Path.home() / ".cache" / "uv" / "archive-v0"
    candidates = sorted(root.glob("*/vllm-0.20.0+cu129.dist-info")) if root.exists() else []
    if candidates:
        archive_path = candidates[-1].parent
    else:
        raise SystemExit(
            "Could not find cached vllm-0.20.0+cu129 archive. "
            "Set VLLM_CU129_ARCHIVE to the archive directory containing vllm/ and vllm-0.20.0+cu129.dist-info/."
        )
else:
    archive_path = Path(archive)

site_packages = Path(site.getsitepackages()[0])
if not (archive_path / "vllm").exists() or not (archive_path / "vllm-0.20.0+cu129.dist-info").exists():
    raise SystemExit(f"Invalid VLLM cu129 archive: {archive_path}")

backup = site_packages / f"_vllm_public_backup_{time.strftime('%Y%m%d_%H%M%S')}"
backup.mkdir(exist_ok=False)
for name in ["vllm", "vllm-0.20.0.dist-info"]:
    path = site_packages / name
    if path.exists():
        shutil.move(str(path), str(backup / name))
for name in ["vllm", "vllm-0.20.0+cu129.dist-info"]:
    dst = site_packages / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(archive_path / name, dst, symlinks=True)
print(f"Patched isolated LMCache env with vLLM cu129 archive: {archive_path}")
INNERPY

python - <<'INNERPY'
from pathlib import Path
import site

site_packages = Path(site.getsitepackages()[0])
path = site_packages / "lmcache/v1/multiprocess/transfer_context.py"
text = path.read_text()
if "import os\n" not in "\n".join(text.splitlines()[:20]) + "\n":
    text = text.replace("from typing import Any, Callable, Protocol\n", "from typing import Any, Callable, Protocol\nimport os\n")
old = """    if device_type == \"cuda\":\n        return HandleTransferContext()\n    return DataTransferContext()\n"""
new = """    if os.environ.get(\"LMCACHE_FORCE_NON_GPU_TRANSFER\", \"0\") == \"1\":\n        logger.info(\"Forcing DataTransferContext via LMCACHE_FORCE_NON_GPU_TRANSFER=1\")\n        return DataTransferContext()\n    if device_type == \"cuda\":\n        return HandleTransferContext()\n    return DataTransferContext()\n"""
if new not in text:
    if old not in text:
        raise SystemExit("expected transfer context block not found")
    text = text.replace(old, new)
path.write_text(text)
print("Patched LMCache transfer context for optional non-GPU transfer mode")
INNERPY

vllm --version
python - <<'INNERPY'
import importlib
import importlib.metadata as md
for name in ["lmcache", "vllm", "torch", "triton", "prometheus-client"]:
    print(f"{name}=={md.version(name)}")
importlib.import_module("lmcache.integration.vllm.lmcache_mp_connector")
print("LMCache MP connector import ok")
INNERPY
