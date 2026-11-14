# Project root
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Hugging Face cache settings
export HF_HOME="$PROJECT_DIR/models/hf_home"
export HF_HUB_CACHE="$PROJECT_DIR/models/hf_cache"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=120

# Local model path, if already downloaded
export QWEN05B_PATH="$(find "$HF_HUB_CACHE/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1)"
