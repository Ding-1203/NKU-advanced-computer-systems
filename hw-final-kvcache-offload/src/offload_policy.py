"""Policy primitives for trace-based KV offload replay.

The replay is prefix-group level by design: it evaluates placement, offload, and
reload decisions using the same metadata exposed by the CoQA final experiment.
It does not attempt to mutate vLLM's internal block tables or reload tensors.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

PolicyName = Literal[
    "gpu_only",
    "cpu_lru",
    "cpu_lfu",
    "size_aware",
    "cargo_utility",
    "cargo_coverage_guarded",
    "oracle_upper_bound",
]
Location = Literal["gpu", "cpu", "evicted", "none"]


@dataclass
class ReplayConfig:
    gpu_token_capacity: int = 138_240
    cpu_token_capacity: int = 1_105_920
    useful_coverage_threshold: float = 0.5
    cpu_reload_fixed_ms: float = 8.0
    cpu_bandwidth_gb_s: float = 2.0
    block_size: int = 16
    hidden_layers: int = 24
    kv_heads: int = 2
    head_dim: int = 64
    dtype_bytes: int = 2
    cargo_min_admit_tokens: int = 768
    cargo_cpu_score_threshold: float = 0.0
    cargo_reload_gain_threshold_ms: float = 0.0
    cargo_first_access_reuse_prior: float = 0.25

    @property
    def bytes_per_token(self) -> int:
        return self.hidden_layers * 2 * self.kv_heads * self.head_dim * self.dtype_bytes

    def estimate_bytes(self, tokens: int) -> int:
        aligned_tokens = ((max(tokens, 0) + self.block_size - 1) // self.block_size) * self.block_size
        return aligned_tokens * self.bytes_per_token

    def reload_penalty_ms(self, tokens: int) -> float:
        bytes_to_load = self.estimate_bytes(tokens)
        transfer_ms = bytes_to_load / max(self.cpu_bandwidth_gb_s, 1e-9) / 1_000_000_000 * 1000.0
        return self.cpu_reload_fixed_ms + transfer_ms


@dataclass
class PrefixState:
    prefix_group: str
    prefix_tokens: int
    planned_reuse_count: int = 0
    first_seen_index: int = 0
    last_access_index: int = 0
    access_count: int = 0
    reuse_count_so_far: int = 0
    cumulative_saved_ms: float = 0.0
    location: Location = "none"
    offload_count: int = 0
    reload_count: int = 0
    evict_count: int = 0


@dataclass
class RequestReplayResult:
    policy: str
    trace_name: str
    pressure_level: int
    capacity_tokens: int
    request_id: str
    request_index: int
    prefix_group: str
    phase: str
    prefix_tokens: int
    prior_access_count: int
    is_repeat_online: bool
    cache_result: str
    hit_coverage: float
    useful_hit: bool
    estimated_ttft_ms: float
    recovered_ttft_saved_ms: float
    offload_bytes: int
    reload_bytes: int
    evicted_utility: float
    damaging_eviction: bool
    gpu_tokens_used: int
    cpu_tokens_used: int


@dataclass
class TwoTierCache:
    policy: PolicyName
    config: ReplayConfig
    saved_ms_by_bucket: dict[int, float]
    cold_baseline_ms: float
    warm_baseline_ms: float
    gpu: OrderedDict[str, None] = field(default_factory=OrderedDict)
    cpu: OrderedDict[str, None] = field(default_factory=OrderedDict)
    states: dict[str, PrefixState] = field(default_factory=dict)
    gpu_tokens_used: int = 0
    cpu_tokens_used: int = 0

    def estimate_saved_ms(self, prefix_tokens: int) -> float:
        if not self.saved_ms_by_bucket:
            return max(0.0, self.cold_baseline_ms - self.warm_baseline_ms)
        points = sorted((int(k), max(0.0, float(v))) for k, v in self.saved_ms_by_bucket.items())
        if prefix_tokens <= points[0][0]:
            return points[0][1]
        if prefix_tokens >= points[-1][0]:
            return points[-1][1]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x0 <= prefix_tokens <= x1:
                ratio = (prefix_tokens - x0) / (x1 - x0)
                return y0 + ratio * (y1 - y0)
        return points[-1][1]

    def estimated_ttft_for(self, cache_result: str, prefix_tokens: int) -> float:
        saved = self.estimate_saved_ms(prefix_tokens)
        if cache_result == "gpu_hit":
            return max(self.warm_baseline_ms, self.cold_baseline_ms - saved)
        if cache_result == "cpu_reload":
            return max(self.warm_baseline_ms, self.cold_baseline_ms - saved + self.config.reload_penalty_ms(prefix_tokens))
        return self.cold_baseline_ms

    def historical_reuse_signal(self, state: PrefixState) -> float:
        # Strict online signal: a fixed first-access prior plus completed repeat
        # observations. The prior is a policy hyperparameter; it is not estimated
        # from future requests in the replayed trace.
        return self.config.cargo_first_access_reuse_prior + float(max(0, state.access_count - 1))

    def policy_value(self, state: PrefixState) -> float:
        saved = self.estimate_saved_ms(state.prefix_tokens)
        return max(0.0, saved) * self.historical_reuse_signal(state)

    def oracle_value(self, state: PrefixState) -> float:
        saved = self.estimate_saved_ms(state.prefix_tokens)
        completed_reuses = max(0, state.access_count - 1)
        remaining_reuse = max(0, state.planned_reuse_count - completed_reuses)
        return max(0.0, saved) * remaining_reuse

    def policy_score(self, state: PrefixState) -> float:
        return self.policy_value(state) / max(1, self.config.estimate_bytes(state.prefix_tokens))

    def oracle_score(self, state: PrefixState) -> float:
        return self.oracle_value(state) / max(1, self.config.estimate_bytes(state.prefix_tokens))

    def cpu_score(self, state: PrefixState) -> float:
        saved = self.estimate_saved_ms(state.prefix_tokens)
        net = max(0.0, saved - self.config.reload_penalty_ms(state.prefix_tokens))
        if self.policy == "oracle_upper_bound":
            completed_reuses = max(0, state.access_count - 1)
            signal = max(0, state.planned_reuse_count - completed_reuses)
        else:
            signal = self.historical_reuse_signal(state)
        return net * signal / max(1, self.config.estimate_bytes(state.prefix_tokens))

    def should_admit_gpu(self, state: PrefixState) -> bool:
        if self.policy in {"cargo_utility", "cargo_coverage_guarded", "oracle_upper_bound"}:
            if state.prefix_tokens < self.config.cargo_min_admit_tokens:
                return False
            if self.policy == "oracle_upper_bound":
                return self.oracle_value(state) > 0
            return True
        return True

    def should_offload_cpu(self, state: PrefixState) -> bool:
        if self.policy == "gpu_only":
            return False
        if self.policy in {"cargo_utility", "cargo_coverage_guarded", "oracle_upper_bound"}:
            if state.prefix_tokens < self.config.cargo_min_admit_tokens:
                return False
            return self.cpu_score(state) > self.config.cargo_cpu_score_threshold
        return True

    def should_reload_from_cpu(self, state: PrefixState) -> bool:
        if self.policy == "gpu_only":
            return False
        if self.policy in {"cargo_utility", "cargo_coverage_guarded", "oracle_upper_bound"}:
            saved = self.estimate_saved_ms(state.prefix_tokens)
            gain = saved - self.config.reload_penalty_ms(state.prefix_tokens)
            return gain > self.config.cargo_reload_gain_threshold_ms
        return True

    def touch_lru(self, cache: OrderedDict[str, None], group: str) -> None:
        if group in cache:
            cache.move_to_end(group)

    def get_or_create_state(self, row: pd.Series, request_index: int) -> PrefixState:
        group = str(row["prefix_group"])
        if group not in self.states:
            planned = int(row.get("planned_reuse_count", 0) or 0)
            tokens = int(row.get("actual_prefix_tokens", row.get("prefix_tokens", 0)) or 0)
            self.states[group] = PrefixState(group, tokens, planned, request_index, request_index)
        state = self.states[group]
        state.last_access_index = request_index
        return state

    def add_gpu(self, state: PrefixState) -> tuple[int, float, bool]:
        if state.prefix_group in self.gpu:
            self.touch_lru(self.gpu, state.prefix_group)
            state.location = "gpu"
            return 0, 0.0, False
        tokens = state.prefix_tokens
        total_offload_bytes = 0
        total_evicted_utility = 0.0
        any_damaging = False
        while self.gpu_tokens_used + tokens > self.config.gpu_token_capacity and self.gpu:
            victim = self.choose_gpu_victim()
            offload_bytes, evicted_utility, damaging = self.remove_from_gpu(victim)
            total_offload_bytes += offload_bytes
            total_evicted_utility += evicted_utility
            any_damaging = any_damaging or damaging
        if tokens <= self.config.gpu_token_capacity and self.should_admit_gpu(state):
            self.gpu[state.prefix_group] = None
            self.gpu_tokens_used += tokens
            state.location = "gpu"
        return total_offload_bytes, total_evicted_utility, any_damaging

    def remove_from_gpu(self, group: str) -> tuple[int, float, bool]:
        victim = self.states[group]
        self.gpu.pop(group, None)
        self.gpu_tokens_used = max(0, self.gpu_tokens_used - victim.prefix_tokens)
        victim.evict_count += 1
        evicted_utility = self.oracle_value(victim)
        damaging = evicted_utility > 0
        if self.should_offload_cpu(victim):
            self.add_cpu(victim)
            victim.offload_count += 1
            return self.config.estimate_bytes(victim.prefix_tokens), evicted_utility, damaging
        victim.location = "evicted"
        return 0, evicted_utility, damaging

    def add_cpu(self, state: PrefixState) -> None:
        if state.prefix_group in self.cpu:
            self.touch_lru(self.cpu, state.prefix_group)
            state.location = "cpu"
            return
        tokens = state.prefix_tokens
        while self.cpu_tokens_used + tokens > self.config.cpu_token_capacity and self.cpu:
            victim_group = self.choose_cpu_victim()
            victim = self.states[victim_group]
            self.cpu.pop(victim_group, None)
            self.cpu_tokens_used = max(0, self.cpu_tokens_used - victim.prefix_tokens)
            victim.location = "evicted"
            victim.evict_count += 1
        if tokens <= self.config.cpu_token_capacity:
            self.cpu[state.prefix_group] = None
            self.cpu_tokens_used += tokens
            state.location = "cpu"

    def choose_gpu_victim(self) -> str:
        if self.policy in {"cpu_lru", "gpu_only"}:
            return next(iter(self.gpu))
        if self.policy == "cpu_lfu":
            return min(self.gpu.keys(), key=lambda g: (self.states[g].access_count, self.states[g].last_access_index))
        if self.policy == "size_aware":
            return max(self.gpu.keys(), key=lambda g: (self.states[g].prefix_tokens, -self.states[g].last_access_index))
        if self.policy == "oracle_upper_bound":
            return min(self.gpu.keys(), key=lambda g: (self.oracle_score(self.states[g]), self.states[g].last_access_index))
        return min(self.gpu.keys(), key=lambda g: (self.policy_score(self.states[g]), self.states[g].last_access_index))

    def choose_cpu_victim(self) -> str:
        if self.policy in {"cargo_utility", "cargo_coverage_guarded", "oracle_upper_bound"}:
            return min(self.cpu.keys(), key=lambda g: (self.cpu_score(self.states[g]), self.states[g].last_access_index))
        if self.policy == "cpu_lfu":
            return min(self.cpu.keys(), key=lambda g: (self.states[g].access_count, self.states[g].last_access_index))
        return next(iter(self.cpu))

    def replay_request(self, row: pd.Series, request_index: int) -> RequestReplayResult:
        state = self.get_or_create_state(row, request_index)
        phase = str(row.get("phase", ""))
        pressure_level = int(row.get("pressure_level", 0) or 0)
        request_id = str(row.get("request_id", ""))
        trace_name = str(row.get("trace_name", row.get("workload_name", "unknown")))
        prior_access_count = state.access_count
        is_repeat_online = prior_access_count > 0

        offload_bytes = 0
        reload_bytes = 0
        evicted_utility = 0.0
        damaging_eviction = False
        cache_result = "cold_miss"
        hit_coverage = 0.0

        if state.prefix_group in self.gpu:
            cache_result = "gpu_hit"
            hit_coverage = 1.0
            self.touch_lru(self.gpu, state.prefix_group)
            state.location = "gpu"
        elif state.prefix_group in self.cpu and self.should_reload_from_cpu(state):
            cache_result = "cpu_reload"
            hit_coverage = 1.0
            reload_bytes = self.config.estimate_bytes(state.prefix_tokens)
            self.cpu.pop(state.prefix_group, None)
            self.cpu_tokens_used = max(0, self.cpu_tokens_used - state.prefix_tokens)
            state.reload_count += 1
            offload_bytes, evicted_utility, damaging_eviction = self.add_gpu(state)
        else:
            if self.should_admit_gpu(state):
                offload_bytes, evicted_utility, damaging_eviction = self.add_gpu(state)
            else:
                state.location = "evicted"

        useful_hit = is_repeat_online and hit_coverage >= self.config.useful_coverage_threshold
        if self.policy == "cargo_coverage_guarded" and hit_coverage < self.config.useful_coverage_threshold:
            useful_hit = False

        estimated_ttft = self.estimated_ttft_for(cache_result, state.prefix_tokens)
        recovered = max(0.0, self.cold_baseline_ms - estimated_ttft)
        if cache_result in {"gpu_hit", "cpu_reload"}:
            state.cumulative_saved_ms += recovered

        state.access_count += 1
        state.reuse_count_so_far = max(0, state.access_count - 1)

        return RequestReplayResult(
            policy=self.policy,
            trace_name=trace_name,
            pressure_level=pressure_level,
            capacity_tokens=self.config.gpu_token_capacity,
            request_id=request_id,
            request_index=request_index,
            prefix_group=state.prefix_group,
            phase=phase,
            prefix_tokens=state.prefix_tokens,
            prior_access_count=prior_access_count,
            is_repeat_online=is_repeat_online,
            cache_result=cache_result,
            hit_coverage=hit_coverage,
            useful_hit=useful_hit,
            estimated_ttft_ms=estimated_ttft,
            recovered_ttft_saved_ms=recovered,
            offload_bytes=offload_bytes,
            reload_bytes=reload_bytes,
            evicted_utility=evicted_utility,
            damaging_eviction=damaging_eviction,
            gpu_tokens_used=self.gpu_tokens_used,
            cpu_tokens_used=self.cpu_tokens_used,
        )


def replay_pressure_level(
    pressure_rows: pd.DataFrame,
    policy: PolicyName,
    config: ReplayConfig,
    saved_ms_by_bucket: dict[int, float],
    cold_baseline_ms: float,
    warm_baseline_ms: float,
) -> list[RequestReplayResult]:
    cache = TwoTierCache(policy, config, saved_ms_by_bucket, cold_baseline_ms, warm_baseline_ms)
    rows = pressure_rows.sort_values(["request_index", "request_id"]).reset_index(drop=True)
    return [cache.replay_request(row, index) for index, row in rows.iterrows()]
