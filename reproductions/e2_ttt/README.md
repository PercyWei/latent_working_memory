# 20261108_E²-TTT 第一阶段复现

创建时间：20261108 17:02:06 Asia/Shanghai

最后修订时间：20261116 11:15:42 Asia/Shanghai

本目录使用 [Rethinking Expressivity and Efficiency in Test-Time Training](https://arxiv.org/abs/2608.21308) 的官方实现和作者公开 checkpoint。公开权重加载、生成和计分检查已完成，尚未完成本轮计划的完整评估。

源码与本地优化边界见 [UPSTREAM.md](UPSTREAM.md)，验证结论见[复现状态](../20261110_ttt_reproduction_status.md)。

## 评估目标与数据

MLP 和 SwiGLU 两个 1.3B 模型均使用下表所列的 **10,095 条样本**进行评估。下载和构造命令见[评估数据准备](../20261108_ttt_data_preparation.md)。

| benchmark | 任务与长度 | 样本数 | 默认数据目录 |
|---|---|---:|---|
| RULER | S-NIAH-1／2；512、1K、2K、4K、8K、16K；每项任务、每个长度 500 条 | 6,000 | `data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b` |
| SQuAD completion | 完整 validation | 2,984 | `data/benchmarks/based-squad` |
| SWDE | 完整 validation | 1,111 | `data/benchmarks/based-swde` |

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

MLP 和 SwiGLU 均使用 `optimized`，分别执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/e2_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --output-dir artifacts/reproductions/e2_ttt/eval/mlp-1.3b-15b_ruler-squad-completion-swde_optimized \
  --inference-mode optimized \
  --batch-size 8

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/e2_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/e2_ttt/swiglu-1.3b-15b \
  --output-dir artifacts/reproductions/e2_ttt/eval/swiglu-1.3b-15b_ruler-squad-completion-swde_optimized \
  --inference-mode optimized \
  --batch-size 8
```

| 参数 | 默认值与约定 |
|---|---|
| `--checkpoint` | 必需，本地完整 checkpoint 目录 |
| `--output-dir` | 必需，首次运行时目录须不存在 |
| `--inference-mode` | `optimized`（默认）或 `original`；后者仅支持 batch size 1 |
| `--batch-size` | 每卡样本数上限，默认 `1`；只有 `optimized` 支持大于 1 |
| `--benchmarks` | `ruler squad_completion swde`，可选择其中一项或多项 |
| `--lengths` | `512 1024 2048 4096 8192 16384`，仅作用于 RULER |
| `--ruler-data` | `data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b` |
| `--squad-data` | `data/benchmarks/based-squad` |
| `--swde-data` | `data/benchmarks/based-swde` |
| `--resume` | 使用原输出目录继续评估，保持 GPU 顺序与其余参数一致 |

- 首次运行的输出目录须不存在。中断后在原命令末尾追加 `--resume`；不同 checkpoint 或推理模式使用独立目录。
- RULER 输入为 `input + answer_prefix`，greedy 最多生成 128 tokens；输入加生成预算不得超过所选长度，不截断。
- SQuAD／SWDE 保留原文本 `.strip()` 和换行停止，最多生成 48 tokens；总预算 2048，prompt 超过 2000 tokens 时按原规则左截断。
- 按长度降序排列后相邻组批，每批左 padding 到该批最长输入，最后一批保留实际样本数。各样本独立维护真实位置、attention 与 TTT 状态。

## 来源记录与输出

最终分数读取输出根目录的 `summary.json`。

| 文件 | 内容 |
|---|---|
| `summary.json` | 所选样本全部完成后的任务／长度分数 |
| `plan.json`、`execution.json` | 运行设置、代码来源、执行状态与耗时 |
| `predictions.jsonl`、`native_harness/` | 单卡逐题预测、评分和原生结果 |
| `manifest.json` | 并行评估各分片的结果位置 |

## 备用训练入口

本轮使用公开权重进行模型评估。训练设置与备用入口见 [UPSTREAM.md](UPSTREAM.md) 和[官方训练说明](upstream/training/README.md)。
