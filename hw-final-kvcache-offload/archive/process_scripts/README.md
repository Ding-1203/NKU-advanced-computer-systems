# Archived Process Scripts

These scripts were useful during earlier baseline, smoke, and intermediate CoQA
iterations. They are kept for traceability, but the main reproducible final
experiment path now lives in `src/`:

- `src/download_coqa.py`
- `src/build_coqa_final_workloads.py`
- `src/run_baseline_workload.py`
- `src/analyze_coqa_final_results.py`
- `src/cache_event_logger.py`, `src/vllm_cache_observer.py`, `src/sitecustomize.py`

Archived scripts may still work after adding both `archive/process_scripts` and
`src` to `PYTHONPATH`, but they are not the maintained report-facing workflow.
