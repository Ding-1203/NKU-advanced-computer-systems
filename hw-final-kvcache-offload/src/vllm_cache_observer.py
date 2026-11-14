#!/usr/bin/env python3
"""Monkey patches for observing vLLM v1 prefix-cache behavior.

This module only records events. It does not change cache policy, scheduling,
allocation, or eviction decisions.
"""

from __future__ import annotations

import contextvars
import os
import time
from collections.abc import Iterable, Sequence
from typing import Any

from cache_event_logger import (
    log_cache_event,
    normalize_request_id,
    serialize_block_hash,
)


_CURRENT_INTERNAL_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "vllm_cache_observer_request_id",
    default="",
)
_ACCESS_COUNTS: dict[str, int] = {}
_INSTALLED = False


def _request_fields(request: Any | None = None) -> dict[str, str]:
    internal_request_id = ""
    if request is not None:
        internal_request_id = str(getattr(request, "request_id", "") or "")
    if not internal_request_id:
        internal_request_id = _CURRENT_INTERNAL_REQUEST_ID.get("")
    return {
        "internal_request_id": internal_request_id,
        "request_id": normalize_request_id(internal_request_id),
    }


def _block_fields(block: Any) -> dict[str, Any]:
    block_hash = serialize_block_hash(getattr(block, "block_hash", None))
    fields: dict[str, Any] = {
        "block_id": getattr(block, "block_id", ""),
        "block_hash": block_hash,
        "ref_cnt": getattr(block, "ref_cnt", ""),
        "is_null": bool(getattr(block, "is_null", False)),
    }
    if block_hash:
        _ACCESS_COUNTS[block_hash] = _ACCESS_COUNTS.get(block_hash, 0) + 1
        fields["access_count"] = _ACCESS_COUNTS[block_hash]
        fields["last_access_time"] = time.time()
    return fields


def _flatten_blocks(kv_cache_blocks: Any) -> list[Any]:
    groups = getattr(kv_cache_blocks, "blocks", ()) or ()
    blocks: list[Any] = []
    for group in groups:
        blocks.extend(block for block in group if not getattr(block, "is_null", False))
    return blocks


def _set_current_request(request: Any):
    return _CURRENT_INTERNAL_REQUEST_ID.set(str(getattr(request, "request_id", "") or ""))


def _reset_current_request(token: contextvars.Token[str]) -> None:
    _CURRENT_INTERNAL_REQUEST_ID.reset(token)


def _install_kv_cache_manager_patch() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if getattr(KVCacheManager, "_cache_observer_patched", False):
        return

    original_get_computed_blocks = KVCacheManager.get_computed_blocks
    original_allocate_slots = KVCacheManager.allocate_slots
    original_free = KVCacheManager.free

    def get_computed_blocks_wrapper(self, request):
        token = _set_current_request(request)
        try:
            kv_cache_blocks, hit_tokens = original_get_computed_blocks(self, request)
        finally:
            _reset_current_request(token)

        hit_blocks_list = _flatten_blocks(kv_cache_blocks)
        hit_blocks = len(hit_blocks_list)
        request_meta = _request_fields(request)
        base = {
            **request_meta,
            "hit_tokens": int(hit_tokens or 0),
            "hit_blocks": hit_blocks,
            "prompt_tokens": getattr(request, "num_prompt_tokens", ""),
            "request_tokens": getattr(request, "num_tokens", ""),
            "block_hashes": [
                serialize_block_hash(getattr(block, "block_hash", None))
                for block in hit_blocks_list
            ],
            "block_ids": [getattr(block, "block_id", "") for block in hit_blocks_list],
            "cache_enabled": bool(getattr(self, "enable_caching", False)),
            "skip_reading_prefix_cache": bool(
                getattr(request, "skip_reading_prefix_cache", False)
            ),
        }
        log_cache_event("prefix_lookup", **base)
        log_cache_event("prefix_hit" if hit_tokens else "prefix_miss", **base)
        return kv_cache_blocks, hit_tokens

    def allocate_slots_wrapper(self, request, *args, **kwargs):
        token = _set_current_request(request)
        try:
            return original_allocate_slots(self, request, *args, **kwargs)
        finally:
            _reset_current_request(token)

    def free_wrapper(self, request):
        token = _set_current_request(request)
        try:
            return original_free(self, request)
        finally:
            _reset_current_request(token)

    KVCacheManager.get_computed_blocks = get_computed_blocks_wrapper
    KVCacheManager.allocate_slots = allocate_slots_wrapper
    KVCacheManager.free = free_wrapper
    KVCacheManager._cache_observer_patched = True


def _install_block_pool_patch() -> None:
    from vllm.v1.core.block_pool import BlockPool

    if getattr(BlockPool, "_cache_observer_patched", False):
        return

    original_cache_full_blocks = BlockPool.cache_full_blocks
    original_touch = BlockPool.touch
    original_free_blocks = BlockPool.free_blocks
    original_maybe_evict = BlockPool._maybe_evict_cached_block

    def cache_full_blocks_wrapper(
        self,
        request,
        blocks,
        num_cached_blocks,
        num_full_blocks,
        block_size,
        kv_cache_group_id,
    ):
        result = original_cache_full_blocks(
            self,
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            kv_cache_group_id,
        )
        request_meta = _request_fields(request)
        for index, block in enumerate(blocks[num_cached_blocks:num_full_blocks]):
            if getattr(block, "is_null", False) or getattr(block, "block_hash", None) is None:
                continue
            block_index = num_cached_blocks + index
            log_cache_event(
                "block_store",
                **request_meta,
                **_block_fields(block),
                block_size=block_size,
                kv_cache_group_id=kv_cache_group_id,
                token_start=block_index * block_size,
                token_end=(block_index + 1) * block_size,
            )
        return result

    def touch_wrapper(self, blocks: Sequence[Any]):
        blocks_list = list(blocks)
        result = original_touch(self, blocks_list)
        request_meta = _request_fields()
        for block in blocks_list:
            if getattr(block, "is_null", False):
                continue
            log_cache_event("block_touch", **request_meta, **_block_fields(block))
        return result

    def free_blocks_wrapper(self, ordered_blocks: Iterable[Any]):
        blocks_list = list(ordered_blocks)
        result = original_free_blocks(self, blocks_list)
        request_meta = _request_fields()
        for block in blocks_list:
            if getattr(block, "is_null", False):
                continue
            log_cache_event(
                "block_free",
                **request_meta,
                **_block_fields(block),
                returned_to_free_queue=getattr(block, "ref_cnt", 1) == 0,
            )
        return result

    def maybe_evict_cached_block_wrapper(self, block):
        old_hash = serialize_block_hash(getattr(block, "block_hash", None))
        old_ref_cnt = getattr(block, "ref_cnt", "")
        result = original_maybe_evict(self, block)
        if result:
            log_cache_event(
                "block_evict",
                **_request_fields(),
                block_id=getattr(block, "block_id", ""),
                block_hash=old_hash,
                ref_cnt=old_ref_cnt,
            )
        return result

    BlockPool.cache_full_blocks = cache_full_blocks_wrapper
    BlockPool.touch = touch_wrapper
    BlockPool.free_blocks = free_blocks_wrapper
    BlockPool._maybe_evict_cached_block = maybe_evict_cached_block_wrapper
    BlockPool._cache_observer_patched = True


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("VLLM_CACHE_OBSERVER", "") != "1":
        return
    _install_kv_cache_manager_patch()
    _install_block_pool_patch()
    _INSTALLED = True
