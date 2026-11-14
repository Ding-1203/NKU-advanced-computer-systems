#!/usr/bin/env python3
"""Small JSONL logger for vLLM prefix-cache observation events."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


DEFAULT_EVENT_LOG = Path("logs/cache_observation/cache_events.jsonl")
BASE_FIELDS = [
    "timestamp",
    "pid",
    "event_type",
    "internal_request_id",
    "request_id",
    "hit_tokens",
    "hit_blocks",
    "block_id",
    "block_hash",
    "ref_cnt",
]
VLLM_COMPLETION_SUFFIX = re.compile(r"^(?P<base>.+)-\d+-[0-9a-fA-F]{8}$")


def event_log_path() -> Path:
    return Path(os.environ.get("VLLM_CACHE_OBSERVER_LOG", str(DEFAULT_EVENT_LOG)))


def normalize_request_id(internal_request_id: str | None) -> str:
    if not internal_request_id:
        return ""
    request_id = str(internal_request_id)
    if request_id.startswith("cmpl-"):
        request_id = request_id[len("cmpl-") :]

    # vLLM OpenAI completion serving formats request ids as
    # cmpl-{client_request_id}-{prompt_index}-{short_random_suffix}.
    matched = VLLM_COMPLETION_SUFFIX.match(request_id)
    if matched:
        return matched.group("base")

    if "-" in request_id:
        base, maybe_index = request_id.rsplit("-", 1)
        if maybe_index.isdigit():
            return base
    return request_id


def serialize_block_hash(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def log_cache_event(event_type: str, **fields: Any) -> None:
    path = event_log_path()
    row: dict[str, Any] = {field: "" for field in BASE_FIELDS}
    row.update(
        {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "event_type": event_type,
        }
    )
    row.update(fields)
    if not row.get("request_id"):
        row["request_id"] = normalize_request_id(row.get("internal_request_id"))

    safe_row = {str(key): _json_safe(value) for key, value in row.items()}
    line = json.dumps(safe_row, ensure_ascii=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
