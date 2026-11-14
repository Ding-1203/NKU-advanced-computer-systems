#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${VLLM_SERVED_MODEL:-qwen05b}"
PROMPT_PREFIX="LMCache smoke shared prefix: CoQA-style story context with a repeated long prefix. The assistant must answer using only this shared context. The city museum opened a night archive for researchers, and the same curator repeated the schedule, access rule, and storage policy many times so the prefix cache has aligned chunks."

curl -sS "${BASE_URL}/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT_PREFIX} Question: who opened the archive?\",\"max_tokens\":8,\"temperature\":0}" >/dev/null

curl -sS "${BASE_URL}/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT_PREFIX} Question: what policy was repeated?\",\"max_tokens\":8,\"temperature\":0}" >/dev/null

echo "Sent two LMCache smoke requests. Check lmcache server/vLLM logs for Stored/Retrieved tokens."
