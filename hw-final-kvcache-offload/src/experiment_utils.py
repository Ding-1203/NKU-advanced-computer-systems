"""Shared helpers for the final vLLM prefix-cache experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TokenCounter:
    def __init__(self, model_path: str | None) -> None:
        self.tokenizer = None
        if model_path:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=False,
                    local_files_only=True,
                )
            except Exception as exc:  # pragma: no cover - fallback is explicit.
                print(f"Warning: failed to load tokenizer from {model_path}: {exc}")
                print("Falling back to whitespace token estimates.")

    def count(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.split())
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_text(row: dict[str, Any], turn_index: int) -> str:
    answers = row.get("answers") or {}
    if isinstance(answers, dict):
        input_text = answers.get("input_text") or []
        if turn_index < len(input_text):
            return str(input_text[turn_index])
    return ""


def story_id(row: dict[str, Any], story_index: int) -> str:
    for key in ["id", "story_id", "filename", "name"]:
        value = row.get(key)
        if value not in [None, ""]:
            return str(value)
    return f"story-{story_index:05d}"
