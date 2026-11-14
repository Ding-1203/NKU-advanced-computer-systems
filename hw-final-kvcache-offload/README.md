# vLLM Prefix Cache 与 Online CARGO Offload 实验

本仓库是《高级计算机系统》大作业“大模型推理前缀缓存”的代码与实验材料。项目围绕 vLLM Prefix Cache 的真实服务模式实验展开，并进一步基于 CoQA trace 设计 online CARGO replay，用于验证 utility-aware offload 策略相对 LRU、LFU、size-aware 等常规策略的收益。

项目最终保留三条主线：

1. **vLLM Prefix Cache baseline**：在 OpenAI-compatible server 模式下启动 vLLM，一次加载模型，多请求连续发送，验证相同 CoQA story/document 前缀可以跨请求复用。
2. **Prefix Cache 事件观测**：通过 `sitecustomize.py` + monkey patch 记录 vLLM 内部 prefix lookup、hit/miss、block store/touch/free/evict 事件，并与 runner CSV 对齐。
3. **Online CARGO replay**：基于真实 CoQA 实验 summary 做 two-tier cache replay。除 oracle upper bound 外，策略只使用历史访问状态，不使用 future reuse 信息。

LMCache MP mode 也做了真实系统探索，但由于 WSL2 下 CUDA IPC 出现 `invalid resource handle`，最终只作为功能验证和风险记录，不作为主要性能结论。

---

## 1. 环境

本项目在 WSL2 Ubuntu 22.04 中运行，不建议在 Windows CMD 或 PowerShell 中直接运行。

核心环境如下：

```text
Python: 3.12.13
vLLM: 0.20.0+cu129
PyTorch: 2.11.0+cu129
Triton: 3.6.0
GPU: NVIDIA RTX 4060 Laptop GPU, 8GB
Model: Qwen2.5-0.5B-Instruct
```

进入项目后先激活环境：

```bash
cd ~/advanced-computer-systems/hw-final-kvcache-offload
source .venv/bin/activate
source env.sh
```

`env.sh` 会设置项目路径、Hugging Face 缓存路径和本地模型路径。实验默认使用：

```bash
echo "$QWEN05B_PATH"
```

不要直接使用 Hugging Face 远程模型名启动正式实验，避免重复下载模型。

---

## 2. 目录结构

```text
hw-final-kvcache-offload/
├── src/
│   ├── download_coqa.py                 # 下载/导出 CoQA JSONL
│   ├── build_coqa_final_workloads.py    # 构造最终 CoQA workload
│   ├── run_baseline_workload.py         # 请求 vLLM OpenAI API 并记录 TTFT
│   ├── analyze_coqa_final_results.py    # 生成 CoQA baseline/pressure summary 和图
│   ├── cache_event_logger.py            # cache event JSONL logger
│   ├── vllm_cache_observer.py           # vLLM monkey patch observer
│   ├── sitecustomize.py                 # observer 自动安装入口
│   ├── offload_policy.py                # online replay 策略实现
│   ├── offload_replay_simulator.py      # two-tier cache replay 模拟器
│   ├── run_online_cargo_replay.py       # 运行 online CARGO replay
│   └── analyze_online_cargo_replay.py   # 生成 CARGO replay 图表
├── scripts/
│   ├── start_vllm_server.sh             # Prefix Cache ON 服务
│   ├── start_vllm_server_no_prefix.sh   # Prefix Cache OFF 服务
│   ├── start_vllm_server_observed.sh    # Prefix Cache ON + observer 服务
│   ├── start_lmcache_server.sh          # 可选 LMCache server
│   └── start_vllm_server_lmcache.sh     # 可选 vLLM + LMCache connector
├── docs/                                # 实验设计与阶段记录
├── configs/                             # LMCache constraints 等配置
├── data/                                # 本地数据与 workload，通常不提交大文件
├── logs/                                # 实验日志和 CSV，通常不提交
├── figures/                             # 生成图表，通常不提交
├── report/                              # 实验报告 PDF / TeX
└── archive/process_scripts/             # 早期过程脚本，仅保留追溯价值
```

最终源码包只需要包含 `src/`、`scripts/`、`configs/`、`docs/`、`README.md`、`env.sh` 等关键文件，不需要包含 `.venv/`、`models/`、`logs/`、`figures/`、`lmcache_disk/` 或大型数据缓存。

---

## 3. 数据与 Workload

实验使用 CoQA 构造真实 story-question 前缀复用场景。先下载并导出本地 JSONL：

```bash
python src/download_coqa.py
```

默认输出：

```text
data/raw/coqa/train.jsonl
data/raw/coqa/validation.jsonl
```

构造最终 CoQA workload：

```bash
python src/build_coqa_final_workloads.py
```

默认输出：

```text
data/workloads/coqa_final/coqa_final_reuse.jsonl
data/workloads/coqa_final/coqa_final_pressure.jsonl
data/workloads/coqa_final/coqa_final_pressure_g8.jsonl
data/workloads/coqa_final/coqa_final_pressure_g16.jsonl
data/workloads/coqa_final/coqa_final_pressure_g24.jsonl
data/workloads/coqa_final/coqa_final_pressure_g32.jsonl
data/workloads/coqa_final/coqa_final_pressure_g48.jsonl
```

`coqa_final_reuse.jsonl` 用于 prefix length sweep 和 popularity-skew；`coqa_final_pressure_g*.jsonl` 用于容量 pressure cliff 实验。

---

## 4. 运行 vLLM Baseline

所有 vLLM 实验都采用服务模式：先启动一次 vLLM server，使模型常驻 GPU；然后 runner 连续发送请求。不要每条请求重新初始化模型，否则 Prefix Cache 无法跨请求复用。

### 4.1 Prefix Cache OFF 对照

启动关闭 Prefix Cache 的 vLLM 服务：

```bash
./scripts/start_vllm_server_no_prefix.sh
```

另开一个 WSL 终端运行 workload：

```bash
source .venv/bin/activate
source env.sh

python src/run_baseline_workload.py \
  --workload data/workloads/coqa_final/coqa_final_reuse.jsonl \
  --output logs/coqa_final/coqa_final_reuse_no_prefix.csv \
  --wait
```

完成后停止该 vLLM 服务，再启动下一组服务。

### 4.2 Prefix Cache ON + Observer

启动带 observer 的 Prefix Cache ON 服务：

```bash
VLLM_CACHE_OBSERVER_LOG=logs/cache_observation/coqa_final/cache_events_reuse.jsonl \
VLLM_CACHE_OBSERVER_RESET_LOG=1 \
./scripts/start_vllm_server_observed.sh
```

运行相同 workload：

```bash
python src/run_baseline_workload.py \
  --workload data/workloads/coqa_final/coqa_final_reuse.jsonl \
  --output logs/coqa_final/coqa_final_reuse_prefix_observed.csv \
  --wait
```

runner 会通过 OpenAI SDK 的 `extra_body={"request_id": ...}` 将 workload request id 传给 vLLM，便于后续将 CSV 与 observer event JSONL 对齐。

### 4.3 Capacity Pressure

容量压力实验建议每个 `g` 单独重启一次 observed server，避免前一组 cache 状态污染后一组：

```bash
for g in 8 16 24 32 48; do
  echo "Run one pressure level at a time: g=$g"
done
```

每组服务启动示例：

```bash
VLLM_CACHE_OBSERVER_LOG=logs/cache_observation/coqa_final/cache_events_pressure_g24.jsonl \
VLLM_CACHE_OBSERVER_RESET_LOG=1 \
./scripts/start_vllm_server_observed.sh
```

对应 runner：

```bash
python src/run_baseline_workload.py \
  --workload data/workloads/coqa_final/coqa_final_pressure_g24.jsonl \
  --output logs/coqa_final/coqa_final_pressure_g24_observed.csv \
  --wait
```

把 `g24` 替换为 `g8/g16/g32/g48` 即可。

---

## 5. 生成 CoQA Baseline 图表

在完成 Prefix Cache OFF、Prefix Cache ON 和各 pressure level 的 observed CSV/event JSONL 后，运行：

```bash
python src/analyze_coqa_final_results.py
```

默认读取：

```text
logs/coqa_final/coqa_final_reuse_no_prefix.csv
logs/coqa_final/coqa_final_reuse_prefix_observed.csv
logs/cache_observation/coqa_final/cache_events_reuse.jsonl
logs/coqa_final/coqa_final_pressure_g*_observed.csv
logs/cache_observation/coqa_final/cache_events_pressure_g*.jsonl
```

默认输出 summary：

```text
logs/cache_observation/coqa_final/coqa_final_reuse_summary.csv
logs/cache_observation/coqa_final/coqa_final_pressure_summary.csv
```

默认输出四张报告核心图：

```text
figures/cache_observation/coqa_final/01_coqa_prefix_length_ttft_savings.png
figures/cache_observation/coqa_final/02_coqa_on_off_by_cache_state.png
figures/cache_observation/coqa_final/03_coqa_popularity_utility_pareto.png
figures/cache_observation/coqa_final/04_coqa_pressure_cliff.png
```

报告中的主要 baseline 结论来自这四张图：长 prefix 的 warm 请求 TTFT saved 更明显；cold 请求基本没有收益；少数 hot prefix 贡献主要 utility；GPU-only Prefix Cache 在 g16 到 g24 之间出现明显 capacity cliff。

---

## 6. Online CARGO Replay

Online CARGO replay 基于前一步生成的 CoQA summary CSV，不直接读取 future reuse 信息。除 `oracle_upper_bound` 外，策略只使用当前请求之前的访问次数、最近访问时间、prefix 长度、估计 KV 大小和 saved curve。

运行 replay：

```bash
python src/run_online_cargo_replay.py
```

默认输出：

```text
logs/offload_online/online_cargo_replay_requests.csv
logs/offload_online/online_cargo_replay_summary.csv
```

生成 replay 图表：

```bash
python src/analyze_online_cargo_replay.py
```

默认输出：

```text
figures/offload_online/01_no_future_sanity.png
figures/offload_online/02_policy_gain.png
figures/offload_online/03_capacity_sweep.png
figures/offload_online/04_decision_audit.png
```

其中 `01_no_future_sanity.png` 用于检查 online 策略没有接近 oracle 到不合理程度；`02_policy_gain.png` 和 `03_capacity_sweep.png` 展示 CARGO 相对 LRU/LFU/size-aware 的收益；`04_decision_audit.png` 拆分 GPU hit、CPU reload 和 cold miss。

---

## 7. LMCache 探索

LMCache 相关脚本保留在 `scripts/` 和 `configs/` 中：

```text
scripts/freeze_stage3_env.sh
scripts/install_lmcache_guarded.sh
scripts/install_lmcache_isolated.sh
scripts/start_lmcache_server.sh
scripts/start_vllm_server_lmcache.sh
scripts/run_lmcache_smoke.sh
scripts/run_lmcache_pressure_once.sh
configs/lmcache_constraints_cu129.txt
```
