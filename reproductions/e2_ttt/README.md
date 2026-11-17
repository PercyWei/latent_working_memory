# 20261108_E²-TTT 第一阶段复现

创建时间：20261108 17:02:06 Asia/Shanghai

最后修订时间：20261117 11:42:23 Asia/Shanghai

本目录使用 [Rethinking Expressivity and Efficiency in Test-Time Training](https://arxiv.org/abs/2608.21308) 的官方实现和作者公开 checkpoint。公开权重加载、生成和计分检查已完成，尚未完成本轮计划的完整评估。

源码与本地优化边界见 [UPSTREAM.md](UPSTREAM.md)，验证结论见[复现状态](../20261110_ttt_reproduction_status.md)。

## 评估目标与数据

MLP 和 SwiGLU 两个 1.3B 模型均使用下表所列的 **10,095 条样本**进行评估。下载和构造命令见[评估数据准备](../20261108_ttt_data_preparation.md)。数据固定从下表路径读取。

| benchmark | 任务与长度 | 样本数 | 固定数据路径 |
|---|---|---:|---|
| RULER | S-NIAH-1／2；512、1K、2K、4K、8K、16K；每项任务、每个长度 500 条 | 6,000 | `data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b` |
| SQuAD completion | 完整 validation | 2,984 | `data/benchmarks/based-squad/data/validation-00000-of-00001.parquet` |
| SWDE | 完整 validation | 1,111 | `data/benchmarks/based-swde/data/validation-00000-of-00001.parquet` |

论文 `tab:incontext_retrieval_main` 中 **1.3B 参数规模模型**的准确率如下；RULER 参照原文长上下文主图及附录的 S-NIAH 数值表。

| 指标 | MLP | SwiGLU |
|---|---:|---:|
| SWDE accuracy (%) | 37.2 | 47.3 |
| SQuAD accuracy (%) | 45.6 | 44.4 |

## 模型下载

作者公开权重见 [zeyun-zhong/e2-ttt](https://huggingface.co/collections/zeyun-zhong/e2-ttt)，`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
hf download zeyun-zhong/e2-ttt-mlp-1.3B-15B \
  --local-dir ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b

hf download zeyun-zhong/e2-ttt-swiglu-1.3B-15B \
  --local-dir ~/rivermind-data/models/reproductions/e2_ttt/swiglu-1.3b-15b
```

## 环境与运行命令

从服务器项目根目录安装并激活独立评估环境，版本见[评估环境](../20261109_environment_setup.md)：

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh e2_ttt
source ~/rivermind-data/envs/ttt-e2_ttt/bin/activate
```

MLP 和 SwiGLU 分别执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/e2_ttt/scripts/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --output-id mlp-1.3b-15b_20261116 \
  --inference-mode optimized \
  --batch-size 8 \
  --batch-size-by-length 8192=4 16384=2

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/e2_ttt/scripts/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/e2_ttt/swiglu-1.3b-15b \
  --output-id swiglu-1.3b-15b_20261116 \
  --inference-mode optimized \
  --batch-size 8 \
  --batch-size-by-length 8192=4 16384=2
```

| 参数 | 默认值与约定 |
|---|---|
| `--resume` | 沿用原评估协议补齐未完成样本的评估，默认不开启 |
| `--benchmarks` | 默认 `ruler squad_completion swde`，可选择其中一项或多项 |
| `--checkpoint` | 必需，本地完整 checkpoint 目录 |
| `--output-root` | 产物父目录，默认 `artifacts/reproductions/e2_ttt` |
| `--output-id` | 默认 `<checkpoint_name>_<YYYYMMDD-HHMM>` |
| `--inference-mode` | 推理模式，支持 `original`（默认）和 `optimized` |
| `--batch-size` | 每卡样本数上限，默认 `1`；只有 `optimized` 支持大于 1 |
| `--batch-size-by-length` | 按评估长度覆盖每卡上限，未指定的长度使用 `--batch-size`，仅作用于 RULER |
| `--lengths` | 默认 `512 1024 2048 4096 8192 16384`，仅作用于 RULER |

- 评估产物保存在 `<output-root>/eval/<output-id>/<bench>`；新运行在已有同名目录时拒绝覆盖；原评估中断后，指定原 `--output-id` 并追加 `--resume` 可续评估，checkpoint、长度及推理模式保持原计划不变，支持调整 GPU 数量、顺序、`--batch-size` 和 `--batch-size-by-length`，也支持选择原计划内的部分 benchmark。
- RULER 输入为 `input + answer_prefix`，greedy 最多生成 128 tokens；输入加生成预算不得超过所选长度，不截断。
- SQuAD／SWDE 保留原文本 `.strip()` 和换行停止，最多生成 48 tokens；总预算 2048，prompt 超过 2000 tokens 时按原规则左截断。
- 按长度降序排列后相邻组批，每批左 padding 到该批最长输入，最后一批保留实际样本数。各样本独立维护真实位置、attention 与 TTT 状态。

## 来源记录与输出

最终分数分别读取 `<output-root>/eval/<output-id>/<benchmark>/summary.json`。`scripts/eval.sh` 调用 `evaluation/evaluate_parallel.py` 统一调度，每个 worker 加载一次模型，依次处理分配到的多个 benchmark。终端根据已评分 JSONL 显示总体完成数、百分比、吞吐和耗时；进度随各任务评分记录的写出成批更新。

| 文件 | 内容 |
|---|---|
| `evaluation_plan.json`、`launcher_execution.json`、`attempts/` | 父目录内的总评估计划、各次启动设置、代码来源与 worker 日志 |
| `summary.json` | 各 benchmark 的汇总位置及整次运行的完成状态 |
| `<benchmark>/evaluation_plan.json` | 该 benchmark 的固定数据及生成／评分协议 |
| `<benchmark>/summary.json` | 该 benchmark 全部样本完成后的任务／长度分数 |
| `<benchmark>/attempts/` | 各次 worker 的原始逐题结果、原生评分和生成缓存 |
| `<benchmark>/predictions.jsonl`、`<benchmark>/manifest.json` | 该 benchmark 合并后的逐题结果及其原始记录位置 |

## 备用训练入口

本轮使用公开权重进行模型评估。训练设置与备用入口见 [UPSTREAM.md](UPSTREAM.md) 和[官方训练说明](source/e2_ttt/training/README.md)。
