# Online CARGO Replay Validation

This replay validates the CARGO offload policy on the frozen CoQA trace summaries without using future request information in non-oracle policy decisions. It is separate from the LMCache real-system work.

## Decision Contract

For `cargo_utility` and `cargo_coverage_guarded`, each request is processed in arrival order. The policy decision for request `t` may use only:

- prefix group id, used only to look up state already built from earlier requests;
- prefix length / estimated KV bytes;
- the fitted prefix-cache TTFT-saved curve from the baseline calibration;
- fixed replay hyperparameters such as reload cost and first-access reuse prior;
- completed historical state before request `t`: prior access count, last access index, current GPU/CPU placement.

The policy must not use these future/offline labels for non-oracle decisions:

- `planned_reuse_count`;
- `phase`, including `seed`, `filler`, `revisit`, or `warm`;
- `expected_cache_state` and `is_warm_request`;
- `pressure_group_role`;
- future request positions.

`oracle_upper_bound` is the only policy allowed to use the offline remaining-reuse count. It is included only as an upper bound. Offline metrics may still report damaging evictions with future knowledge, but that label is not fed back into CARGO decisions.

Conventional baselines are intentionally simple: `cpu_lru` evicts least-recently used prefixes, `cpu_lfu` evicts lowest historical access count, and `size_aware` evicts the largest prefix first to preserve more cache entries under a byte capacity.

## Workloads

The replay uses existing CoQA summaries rather than synthetic prompts:

- `coqa_pressure_no_prior_signal`: original capacity-pressure trace. It is a sanity check: seed and filler prefixes are indistinguishable before revisit, so an online policy should not match the oracle.
- `coqa_popularity_skew`: original popularity-skew CoQA trace. It tests whether the policy can use online repeat history.
- `coqa_mixed_length_interleaved`: the same CoQA prefix-length rows, deterministically interleaved by occurrence and prefix bucket to create capacity competition across prefix lengths. The policy still sees requests one by one and does not know future repeats.

## Commands

```bash
source .venv/bin/activate
source env.sh
python src/run_online_cargo_replay.py
python src/analyze_online_cargo_replay.py
```

Outputs:

- `logs/offload_online/online_cargo_replay_requests.csv`
- `logs/offload_online/online_cargo_replay_summary.csv`
- `figures/offload_online/01_no_future_sanity.png`
- `figures/offload_online/02_policy_gain.png`
- `figures/offload_online/03_capacity_sweep.png`
- `figures/offload_online/04_decision_audit.png`

## Current Result

Aggregating `g24 + g32 + g48` capacity settings:

| Trace | CARGO recovered TTFT | Best conventional | Gain |
| --- | ---: | ---: | ---: |
| CoQA popularity skew | 15.20 s | 14.24 s | +6.7% |
| CoQA mixed prefix lengths | 6.93 s | 2.86 s | +142.2% |

The pressure sanity figure shows CARGO remains below the oracle, especially at `g48`, which is the expected behavior for an online policy without future seed/revisit labels.
