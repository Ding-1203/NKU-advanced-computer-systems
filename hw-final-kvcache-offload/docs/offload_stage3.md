# Stage 3 Offload Design and Experiment Plan

This stage starts from the frozen CoQA GPU-only prefix-cache baseline and adds two offload tracks:

1. Trace-driven CARGO replay on the real CoQA summaries.
2. LMCache MP-mode real CPU-offload baseline, gated by a strict environment-safety check.

The current project environment must remain stable:

- Python 3.12.13
- PyTorch 2.11.0+cu129
- vLLM 0.20.0+cu129
- Triton 3.6.0
- Local model from `QWEN05B_PATH`

## Frozen Baseline

The frozen GPU-only baseline is the current CoQA final experiment under:

- `logs/cache_observation/coqa_final/`
- `figures/cache_observation/coqa_final/`

The only baseline-figure change in this stage is the Fig. 1 axis alignment: right-axis `speedup = 1.0x` is aligned to left-axis `saved = 0 ms`. The underlying CSV data and conclusions are unchanged.

Regenerate the frozen figures with:

```bash
source .venv/bin/activate
source env.sh
python src/analyze_coqa_final_results.py
```

## Replay Simulator

The replay simulator uses real CoQA summary CSVs instead of synthetic request generation. The cache object is a `prefix_group`: this is intentionally not a block-level tensor reload implementation.

Default inputs:

- `logs/cache_observation/coqa_final/coqa_final_reuse_summary.csv`
- `logs/cache_observation/coqa_final/coqa_final_pressure_summary.csv`

Policies:

- `gpu_only`
- `cpu_lru`
- `cpu_lfu`
- `size_aware`
- `cargo_utility`
- `cargo_coverage_guarded`
- `oracle_upper_bound`

CARGO replay behavior:

- `cargo_utility` and `cargo_coverage_guarded` are historical-only policies. They do not read `planned_reuse_count`, seed/filler role labels, or any future request position.
- Historical CARGO uses prefix length, the fitted TTFT-saved curve, observed access count so far, and estimated reload cost per byte.
- `oracle_upper_bound` is the only policy allowed to read the offline remaining-reuse count; it is a ceiling for interpretation, not a deployable strategy.
- Offline metrics may still use future reuse to label whether an eviction was actually damaging. That information is not fed back into non-oracle policy decisions.
- Keeps the stage scoped to policy contribution; it does not modify vLLM internals.

Important limitation: in the current pressure trace, seed groups and filler groups are both first-time prefixes before the revisit phase. A historical-only strategy cannot know which first-time prefix will return unless the real service exposes a session or application-level reuse hint. Therefore the replay should be interpreted as a conservative policy scaffold plus an oracle ceiling, not as proof that CARGO can predict future reuse from unseen prefixes.

Run replay and generate figures:

```bash
source .venv/bin/activate
source env.sh
python src/offload_replay_simulator.py
python src/analyze_offload_replay.py
```

Outputs:

- `logs/offload_replay/offload_replay_requests.csv`
- `logs/offload_replay/offload_replay_summary.csv`
- `figures/offload_replay/01_replay_pressure_cliff.png`
- `figures/offload_replay/02_policy_recovered_utility.png`
- `figures/offload_replay/03_replay_cost_breakdown.png`

## LMCache Safety Gate

Do not run a bare `pip install lmcache` in the current `.venv`. LMCache must pass a dry-run resolver gate first.

Freeze current environment:

```bash
scripts/freeze_stage3_env.sh
```

Check LMCache install plan without installing:

```bash
scripts/check_lmcache_install_plan.sh
```

If and only if the dry-run does not touch critical packages, uninstall packages, or downgrade the validated stack, install with:

```bash
scripts/install_lmcache_guarded.sh
```

The initial target `lmcache==0.5.0` has a CUDA 12.9 wheel, but its shipped connector expects newer vLLM symbols than the local `vllm==0.20.0+cu129` exposes. The compatible target for this environment is `lmcache==0.4.6` from `v0.4.6-cu129`, with the local cached `vllm==0.20.0+cu129` package tree patched into `.venv-lmcache`.

Current status: the first dry-run showed that the resolver wanted to uninstall or downgrade validated packages such as `numpy`, `protobuf`, `prometheus-client`, and OpenTelemetry packages. After those packages were added to the constraints file, the dry-run became unsatisfiable because `lmcache==0.5.0` requires `prometheus-client<=0.24.1` while the frozen environment has `prometheus-client==0.25.0`. No LMCache install was attempted in the frozen `.venv`, and the existing `.venv` was left unchanged.

The real LMCache experiment should therefore use an isolated environment:

```bash
scripts/install_lmcache_isolated.sh
```

Equivalent manual command:

```bash
uv venv .venv-lmcache --python 3.12
source .venv-lmcache/bin/activate
VERSION=0.4.6
uv pip install "lmcache==${VERSION}" "vllm==0.20.0" "torch==2.11.0+cu129" "triton==3.6.0" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --find-links "https://github.com/LMCache/LMCache/releases/expanded_assets/v${VERSION}-cu129" \
  --index-strategy unsafe-best-match
# Then patch the isolated env with the locally cached vllm-0.20.0+cu129 tree; scripts/install_lmcache_isolated.sh automates this.
```

This permits LMCache's metrics dependency set inside `.venv-lmcache` while preserving the frozen `.venv` baseline.

## LMCache MP-Mode Commands

After `.venv-lmcache` is installed, run LMCache in MP mode with a separate server process. The launch scripts default to `LMCACHE_VENV=.venv-lmcache`.

Start LMCache server:

```bash
scripts/start_lmcache_server.sh
```

Start vLLM with LMCache MP connector:

```bash
scripts/start_vllm_server_lmcache.sh
```

The vLLM config uses `LMCacheMPConnector` plus `kv_connector_module_path = lmcache.integration.vllm.lmcache_mp_connector`, which is required for the LMCache-shipped connector path on vLLM 0.20.x.

Run the two-request smoke after both services are up:

```bash
scripts/run_lmcache_smoke.sh
```

Acceptance for the smoke is an LMCache log containing store and retrieve events for the shared-prefix pair.

Run one real CoQA pressure workload end to end:

```bash
scripts/run_lmcache_pressure_once.sh \
  data/workloads/coqa_final/coqa_final_pressure_g24.jsonl \
  logs/lmcache_real/coqa_pressure_g24.csv
```

The script starts LMCache server, starts vLLM with the MP connector, waits for `/v1/models`, runs the workload, saves the LMCache Prometheus metrics, and then stops both services.

## Current LMCache Real-System Status

The isolated environment that worked on this WSL2 machine is:

- `lmcache==0.4.6`
- local cached `vllm==0.20.0+cu129` patched into `.venv-lmcache`
- `torch==2.11.0+cu129`
- `triton==3.6.0`
- `prometheus-client==0.24.1`

`lmcache==0.5.0` was not compatible with the local vLLM connector API because it expected `KVCacheSpecKind`, which this `vllm==0.20.0+cu129` build does not expose.

The first MP-mode attempt using GPU handle transfer failed under WSL2 with `CUDA error: invalid resource handle`. The current launch path therefore defaults to:

- LMCache server `--transfer-mode non_gpu`
- vLLM env `LMCACHE_FORCE_NON_GPU_TRANSFER=1`

`scripts/install_lmcache_isolated.sh` patches LMCache's `transfer_context.py` so this environment variable forces `DataTransferContext` even when the vLLM-side tensors are CUDA tensors. This is a WSL compatibility workaround, not the desired high-performance path.

Validated smoke result:

- Start LMCache server once.
- Start vLLM with LMCache, send the shared-prefix smoke, stop only vLLM.
- Restart vLLM and send the shared-prefix smoke again.
- LMCache server logs showed `4/4 prefix hits (4 L1, 0 L2)`, confirming cross-process prefix reuse through the external LMCache server.

Validated CoQA pressure results with `scripts/run_lmcache_pressure_once.sh`:

| Groups | Requests OK | Seed median TTFT | Filler median TTFT | Revisit median TTFT | L1 read chunks | L1 write chunks | LMCache token hit rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 96/96 | 385.9 ms | 399.5 ms | 38.8 ms | 1,694 | 7,609 | 18.3% |
| 24 | 144/144 | 413.9 ms | 381.9 ms | 350.0 ms | 2,543 | 11,402 | 18.3% |
| 32 | 192/192 | 394.2 ms | 385.8 ms | 338.1 ms | 3,390 | 15,203 | 18.3% |

The LMCache server logs confirm real external-cache reuse during revisit. The tail of the `g24` and `g32` logs contains near-complete hits such as `96/96 prefix hits (96 L1, 0 L2)`. LMCache L1 eviction remained `0` in these runs, so this experiment stresses vLLM GPU-prefix-cache capacity and reload cost rather than LMCache's own L1 capacity.

This verifies functional CPU-side reuse but does not show a performance win. The non-GPU transfer path adds substantial CPU copy and reconstruction overhead; the real-system result should be interpreted as a deployed LMCache baseline plus a clear overhead diagnosis. For performance claims, the next step is to run the same scripts in a native Linux or otherwise CUDA-IPC-compatible environment and compare against this WSL fallback.
