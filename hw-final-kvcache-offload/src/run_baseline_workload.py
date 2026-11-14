#!/usr/bin/env python3
"""Run a JSONL workload against a vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from openai import OpenAI


DEFAULT_WORKLOAD = Path("data/workloads/baseline_v1_1/prefix_length_sweep.jsonl")

WORKLOAD_FIELDS = [
    "id",
    "request_id",
    "scenario_id",
    "scenario_label",
    "experiment_type",
    "target_prefix_tokens",
    "actual_prefix_tokens",
    "prefix_target_tokens",
    "prefix_tokens",
    "suffix_tokens",
    "prompt_tokens",
    "reuse_ratio_target",
    "reuse_ratio_actual",
    "num_prefix_groups",
    "total_requests",
    "requests_per_group",
    "request_index_in_group",
    "occurrence_index",
    "is_warm_request",
    "expected_cache_state",
    "expected_reused_prefix_tokens",
    "expected_reuse_ratio_token",
    "prefix_group",
    "prefix_group_index",
    "workload_name",
    "phase",
    "pressure_level",
    "pressure_group_role",
    "pressure_group_id",
    "sequence_pass",
    "prefix_bucket",
    "traffic_class",
    "coqa_story_id",
    "coqa_turn_id",
    "context_pack_size",
    "context_pack_story_ids",
    "planned_reuse_count",
    "max_tokens",
    "temperature",
]


RESULT_FIELDS = [
    "request_index",
    "status",
    "started_at_unix",
    "finished_at_unix",
    "ttft_ms",
    "total_latency_ms",
    "output_chars",
    "error",
]


CSV_FIELDS = RESULT_FIELDS[:1] + WORKLOAD_FIELDS + RESULT_FIELDS[1:]


def read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def health_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid base URL: {base_url}")
    return f"{parsed.scheme}://{parsed.netloc}/health"


def wait_for_server(base_url: str, timeout_s: float) -> None:
    url = health_url(base_url)
    deadline = time.perf_counter() + timeout_s
    last_error = ""
    while time.perf_counter() < deadline:
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:120]}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2.0)
    raise TimeoutError(f"Server did not become healthy at {url}: {last_error}")


def run_one_request(
    client: OpenAI,
    model: str,
    row: dict[str, Any],
    timeout_s: float,
    send_request_id: bool,
) -> dict[str, Any]:
    started_at_unix = time.time()
    started = time.perf_counter()
    first_token_at: float | None = None
    output_parts: list[str] = []

    try:
        request_kwargs: dict[str, Any] = {
            "model": model,
            "prompt": row["prompt"],
            "max_tokens": int(row.get("max_tokens", 16)),
            "temperature": float(row.get("temperature", 0.0)),
            "stream": True,
            "timeout": timeout_s,
        }
        request_id = row.get("request_id") or row.get("id")
        if send_request_id and request_id:
            request_kwargs["extra_body"] = {"request_id": str(request_id)}
        stream = client.completions.create(**request_kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            text = chunk.choices[0].text or ""
            if text and first_token_at is None:
                first_token_at = time.perf_counter()
            output_parts.append(text)
        finished = time.perf_counter()
        if first_token_at is None:
            first_token_at = finished
        return {
            "status": "ok",
            "started_at_unix": started_at_unix,
            "finished_at_unix": time.time(),
            "ttft_ms": (first_token_at - started) * 1000.0,
            "total_latency_ms": (finished - started) * 1000.0,
            "output_chars": len("".join(output_parts)),
            "error": "",
        }
    except Exception as exc:
        finished = time.perf_counter()
        return {
            "status": "error",
            "started_at_unix": started_at_unix,
            "finished_at_unix": time.time(),
            "ttft_ms": "",
            "total_latency_ms": (finished - started) * 1000.0,
            "output_chars": 0,
            "error": repr(exc),
        }


def run_warmup(client: OpenAI, model: str, timeout_s: float) -> None:
    stream = client.completions.create(
        model=model,
        prompt=(
            "Warmup request for vLLM baseline measurement. "
            "This prompt is intentionally unrelated to the benchmark prefixes."
        ),
        max_tokens=4,
        temperature=0.0,
        stream=True,
        timeout=timeout_s,
        extra_body={"request_id": "runner-warmup"},
    )
    for _ in stream:
        pass


def output_row(index: int, row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "request_index": index,
        "status": result["status"],
        "started_at_unix": result["started_at_unix"],
        "finished_at_unix": result["finished_at_unix"],
        "ttft_ms": result["ttft_ms"],
        "total_latency_ms": result["total_latency_ms"],
        "output_chars": result["output_chars"],
        "error": result["error"],
    }
    for field in WORKLOAD_FIELDS:
        normalized[field] = row.get(field, "")
    if not normalized["request_id"]:
        normalized["request_id"] = row.get("id", f"req-{index:05d}")
    if not normalized["id"]:
        normalized["id"] = normalized["request_id"]
    return normalized


def write_results(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    if args.wait:
        wait_for_server(args.base_url, args.wait_timeout)

    for _ in range(args.warmup_requests):
        run_warmup(client, args.model, args.timeout)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows):
            result = run_one_request(
                client=client,
                model=args.model,
                row=row,
                timeout_s=args.timeout,
                send_request_id=args.send_request_id,
            )
            writer.writerow(output_row(index, row, result))
            f.flush()
            request_id = row.get("request_id") or row.get("id") or f"req-{index:05d}"
            scenario = row.get("scenario_label") or row.get("scenario_id", "")
            state = row.get("expected_cache_state", "")
            print(
                f"[{index + 1:03d}/{len(rows):03d}] "
                f"{request_id} {scenario} {state} "
                f"status={result['status']} ttft_ms={result['ttft_ms']}"
            )
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen05b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--wait", action="store_true", help="Wait for /health before running.")
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument(
        "--send-request-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send workload request_id to vLLM via extra_body.request_id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.workload, args.limit)
    if not rows:
        raise RuntimeError(f"No workload rows found in {args.workload}")
    write_results(rows, args)
    print(f"Wrote CSV results to {args.output}")


if __name__ == "__main__":
    main()
