# vLLM Prefix Cache Observation Notes

This project observes vLLM 0.20.0 prefix-cache behavior without editing files
inside `.venv`. The observer is enabled only when `VLLM_CACHE_OBSERVER=1` and is
loaded through `src/sitecustomize.py`.

## Relevant vLLM Paths

- `vllm/v1/core/kv_cache_manager.py`
  - `KVCacheManager.get_computed_blocks(request)`: request-level prefix-cache
    lookup. The return value includes cached blocks and hit tokens.
  - `KVCacheManager.allocate_slots(...)`: request allocation context. Evictions
    can happen while new blocks are allocated.
  - `KVCacheManager.free(request)`: request-level free context.
  - `KVCacheManager.cache_blocks(request, num_computed_tokens)`: public cache
    commit wrapper, although the scheduler path often reaches the coordinator
    directly from `allocate_slots`.
- `vllm/v1/core/block_pool.py`
  - `BlockPool.cache_full_blocks(...)`: stores full blocks into the prefix-cache
    hash table.
  - `BlockPool.touch(blocks)`: LRU/ref-count update when cached blocks are hit.
  - `BlockPool.free_blocks(ordered_blocks)`: decreases reference counts and
    returns blocks to the free queue.
  - `BlockPool._maybe_evict_cached_block(block)`: removes a cached block hash
    when a free block is reused.

## Inserted Events

The observer records JSONL events to `VLLM_CACHE_OBSERVER_LOG`.

- `prefix_lookup`: one event for every request-level prefix-cache lookup.
- `prefix_hit`: lookup returned one or more hit tokens.
- `prefix_miss`: lookup returned zero hit tokens.
- `block_store`: a full block was inserted into the prefix-cache hash table.
- `block_touch`: a cached block was reused and its ref count/LRU state changed.
- `block_free`: a request released a block reference.
- `block_evict`: a cached block hash was removed because the block was reused.

Every event includes the fixed fields used by the analysis scripts:

```json
{
  "timestamp": 0.0,
  "pid": 0,
  "event_type": "prefix_hit",
  "internal_request_id": "cmpl-req-00001-0",
  "request_id": "req-00001",
  "hit_tokens": 512,
  "hit_blocks": 32,
  "block_id": 123,
  "block_hash": "...",
  "ref_cnt": 1
}
```

Additional fields such as `prompt_tokens`, `block_ids`, `block_size`,
`pressure_level`, and access counters may appear when available.

## Request Alignment

`src/run_baseline_workload.py` sends each workload row's `request_id` to vLLM
through the OpenAI-compatible `extra_body.request_id` field. vLLM internally
formats completion IDs as `cmpl-{request_id}-0`; the observer strips that prefix
and suffix so event logs can be joined with runner CSV files on `request_id`.

The runner also writes `started_at_unix` and `finished_at_unix`, which lets the
analyzer associate request windows with block eviction events that do not always
carry a request id.

## Commands

Smoke observation:

```bash
source .venv/bin/activate
source env.sh

python archive/process_scripts/build_cache_pressure_workload.py

VLLM_CACHE_OBSERVER_LOG=logs/cache_observation/smoke/cache_events.jsonl \
VLLM_CACHE_OBSERVER_RESET_LOG=1 \
./scripts/start_vllm_server_observed.sh

python src/run_baseline_workload.py \
  --workload data/workloads/baseline_v1_1/baseline_smoke.jsonl \
  --output logs/cache_observation/smoke/observed_baseline_smoke.csv \
  --wait
```

Pressure smoke reuses the smoke workload, but restart the observed server with a
smaller GPU KV-cache budget:

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.35 \
VLLM_CACHE_OBSERVER_LOG=logs/cache_observation/smoke/cache_events.jsonl \
./scripts/start_vllm_server_observed.sh

python src/run_baseline_workload.py \
  --workload data/workloads/cache_observation/cache_pressure_smoke.jsonl \
  --output logs/cache_observation/smoke/cache_pressure.csv \
  --wait
```

Smoke analysis:

```bash
python archive/process_scripts/analyze_cache_events.py
```

Formal observation A uses the existing 5x5 baseline matrix workload:

```bash
VLLM_CACHE_OBSERVER_LOG=logs/cache_observation/formal/cache_events_matrix.jsonl \
VLLM_CACHE_OBSERVER_RESET_LOG=1 \
./scripts/start_vllm_server_observed.sh

python src/run_baseline_workload.py \
  --workload data/workloads/baseline_v1_1/prefix_reuse_matrix.jsonl \
  --output logs/cache_observation/formal/observed_prefix_reuse_matrix.csv \
  --wait
```

Formal observation B generates and runs the cache pressure sweep:

```bash
python archive/process_scripts/build_cache_pressure_workload.py --preset formal

VLLM_GPU_MEMORY_UTILIZATION=0.35 \
VLLM_CACHE_OBSERVER_LOG=logs/cache_observation/formal/cache_events_pressure.jsonl \
VLLM_CACHE_OBSERVER_RESET_LOG=1 \
./scripts/start_vllm_server_observed.sh

python src/run_baseline_workload.py \
  --workload data/workloads/cache_observation/cache_pressure_sweep.jsonl \
  --output logs/cache_observation/formal/cache_pressure_sweep.csv \
  --wait
```

Formal analysis:

```bash
python archive/process_scripts/analyze_cache_events.py --profile formal
```

The analyzer accepts repeated `--events` and `--latency-csv` arguments for
partial or custom runs.

For the report-facing figure set, generate a compact four-figure set from the
formal summary:

```bash
python archive/process_scripts/plot_formal_report_figures.py
```

This writes only the report figures to `figures/cache_observation/formal_report/`:

- `01_ttft_by_prefix_coverage.png`: TTFT distribution by observed prefix-hit
  coverage bucket.
- `02_matrix_prefix_coverage_heatmap.png`: 5x5 matrix validation using warm
  prefix-hit coverage.
- `03_pressure_revisit_capacity_cliff.png`: pressure sweep revisit TTFT,
  coverage, and eviction annotations.
- `04_pressure_hit_quality.png`: any-hit rate versus useful-hit rate and median
  coverage, showing why binary hit rate is misleading.

## Formal Figures

`archive/process_scripts/analyze_cache_events.py --profile formal` writes the joined summary to
`logs/cache_observation/formal/cache_event_summary.csv` and generates these main
figures under `figures/cache_observation/formal/`:

- `ttft_vs_hit_tokens.png`: box plot plus jittered request points by observed
  hit-token bucket.
- `observed_hit_tokens_heatmap.png`: 5x5 prefix length by reuse ratio heatmap
  using median warm observed hit tokens.
- `cache_pressure_sweep_summary.png`: revisit median TTFT bars, median hit-token
  line, and eviction-count annotations by prefix group count.
- `cache_pressure_phase_breakdown.png`: seed/filler/revisit median TTFT and hit
  rate by pressure level.
- `cache_pressure_timeline.png`: simplified debug timeline with phase shading and
  eviction rug marks.
