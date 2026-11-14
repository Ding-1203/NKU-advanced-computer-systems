"""Optional startup hook for the vLLM cache observer."""

from __future__ import annotations

import os
import sys
import traceback


if os.environ.get("VLLM_CACHE_OBSERVER", "") == "1":
    try:
        from vllm_cache_observer import install

        install()
    except Exception as exc:  # pragma: no cover - startup diagnostics.
        print(f"[cache-observer] failed to install: {exc}", file=sys.stderr)
        traceback.print_exc()
        if os.environ.get("VLLM_CACHE_OBSERVER_STRICT", "") == "1":
            raise
