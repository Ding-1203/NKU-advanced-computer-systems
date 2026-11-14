# CoQA Workload Integration

CoQA is used as an optional real-data workload source for prefix-cache
experiments. Each CoQA row contains one `story` and a sequence of `questions`.
The workload builder treats the story as the shared prefix group and emits one
request per selected question.

## Download Locally

Use the project environment and the official Hugging Face endpoint if the local
mirror does not expose the dataset:

```bash
source .venv/bin/activate
source env.sh
HF_ENDPOINT=https://huggingface.co python src/download_coqa.py --force
```

The script writes local JSONL exports to:

```text
data/raw/coqa/train.jsonl
data/raw/coqa/validation.jsonl
data/raw/coqa/dataset_info.json
```

`data/raw/` is ignored by git and should not be committed.

## Build Workloads

Default validation story-reuse workload:

```bash
python archive/process_scripts/build_coqa_workloads.py
```

Long-prefix validation workload:

```bash
python archive/process_scripts/build_coqa_workloads.py   --output data/workloads/coqa/coqa_validation_long_story_reuse.jsonl   --workload-name coqa_long_story_reuse   --limit-stories 40   --turns-per-story 4   --min-prefix-tokens 512   --max-prefix-tokens 1536   --max-prompt-tokens 1800   --access-pattern round_robin
```

Larger long-prefix train workload:

```bash
python archive/process_scripts/build_coqa_workloads.py   --split train   --output data/workloads/coqa/coqa_train_long_story_reuse.jsonl   --workload-name coqa_train_long_story_reuse   --limit-stories 40   --turns-per-story 4   --min-prefix-tokens 512   --max-prefix-tokens 1536   --max-prompt-tokens 1800   --access-pattern round_robin
```

`data/workloads/coqa/` is ignored because it contains derived CoQA text.

## Recommended First Run

For a real-data prefix-cache sanity check, use the train long-story workload:

```bash
VLLM_CACHE_OBSERVER_LOG=logs/cache_observation/coqa/cache_events_train_long.jsonl VLLM_CACHE_OBSERVER_RESET_LOG=1 ./scripts/start_vllm_server_observed.sh

python src/run_baseline_workload.py   --workload data/workloads/coqa/coqa_train_long_story_reuse.jsonl   --output logs/cache_observation/coqa/coqa_train_long_story_reuse.csv   --wait
```

The expected pattern is one cold request per story followed by warm requests for
later questions from the same story. With `round_robin`, all stories are seeded
first and then revisited by turn, which is closer to a multi-user service access
pattern than grouped immediate reuse.
