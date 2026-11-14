# AGENTS.md

本文件用于说明本项目中编码代理需要了解的基础环境和安全须知。

## 项目环境

本项目运行在 **WSL2 Ubuntu 22.04** 中，不在 Windows 原生环境中运行。

运行项目前，请先执行：

```bash
cd ~/advanced-computer-systems/hw-final-kvcache-offload
source .venv/bin/activate
source env.sh
```

当前核心环境：

```
Python: 3.12.13
vLLM: 0.20.0+cu129
PyTorch: 2.11.0+cu129
Model: Qwen2.5-0.5B-Instruct
GPU: NVIDIA RTX 4060 Laptop GPU, 8GB
```

本地模型路径由以下环境变量提供：

```
$QWEN05B_PATH
```

开发和运行时，优先使用本地模型路径，不要直接使用 Hugging Face 远程模型名，避免重复下载模型。

## 基本安全须知

- 未经明确要求，不要重新安装、升级或降级 `vllm`、`torch`、CUDA 相关包或 Python 虚拟环境。
- 未经明确要求，不要删除 `.venv`、`models/`、`lmcache_disk/` 或已有实验日志。
- 未经确认，不要下载大型模型或大型数据集。
- 不要提交大模型文件、缓存文件、虚拟环境或大型实验输出。
- 不要把 API Key、Token、密码或其他私密凭据写入代码或文档。
- 实验日志统一放在 `logs/` 目录下。
- 实验图表统一放在 `figures/` 目录下。


## Git 协作约定

本项目远端仓库为：

```
https://github.com/Ding-1203/kv-cache-offload.git
```

用户已授权编码代理在后续开发中自主执行 `git add` 和 `git commit`，提交应保持主题清晰、粒度适中。

提交前需要检查 `git status`，不要提交 `.venv/`、`models/`、`lmcache_disk/`、大模型文件、缓存文件、私密凭据或大型实验输出。

## WSL / vLLM 注意事项

在 WSL 环境下，vLLM 可能会使用多进程启动方式。

如果 Python 脚本中初始化 vLLM，应使用如下入口保护：

```
from multiprocessing import freeze_support

def main():
    ...

if __name__ == "__main__":
    freeze_support()
    main()
```

不要把 `LLM(...)` 直接写在脚本最外层。

## 开发风格

代码应尽量简单、可复现、便于检查。

建议将不同功能拆分到以下目录：

```
src/
scripts/
configs/
logs/
figures/
```

修改代码时，应尽量保持当前已验证可用的环境不变，避免不必要的依赖变更。