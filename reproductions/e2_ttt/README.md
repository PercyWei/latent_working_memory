# 20261108_E²-TTT 第一阶段复现

创建时间：20261108 17:02:06 Asia/Shanghai

最后修订时间：20261110 21:56:52 Asia/Shanghai

本目录接入 [E²-TTT 官方实现](https://github.com/zeyun-zhong/E2-TTT)及随附 lm-evaluation-harness，保留原始推理路径，另提供本地批量实现。论文为 [Rethinking Expressivity and Efficiency in Test-Time Training](https://arxiv.org/abs/2608.21308)，源码边界见 [UPSTREAM.md](UPSTREAM.md)。公开权重加载、模型生成和计分检查已完成，尚未完成本轮计划的 benchmark 评估；详见[复现状态](../20261110_ttt_reproduction_status.md)。

## 评估范围

MLP 和 SwiGLU 两个 1.3B 模型默认均使用下表所列的 **10,095 条样本**进行评估。下载和构造命令见[评估数据准备](../20261108_ttt_data_preparation.md)。

| benchmark | 任务与长度 | 样本数 | 默认数据目录 |
|---|---|---:|---|
| RULER | S-NIAH-1／2，512、1K、2K、4K、8K、16K；每项每长度 500 条 | 6,000 | `data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b` |
| SQuAD completion | 完整 validation | 2,984 | `data/benchmarks/based-squad` |
| SWDE | 完整 validation | 1,111 | `data/benchmarks/based-swde` |

论文 `tab:incontext_retrieval_main` 的准确率参照如下；RULER 参照原文长上下文主图及附录的 S-NIAH 数值表。

| 指标 | 1.3B MLP | 1.3B SwiGLU |
|---|---:|---:|
| SWDE accuracy (%) | 37.2 | 47.3 |
| SQuAD accuracy (%) | 45.6 | 44.4 |

## 模型下载

下载[作者发布](https://huggingface.co/collections/zeyun-zhong/e2-ttt)的两个完整 checkpoint；`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
hf download zeyun-zhong/e2-ttt-mlp-1.3B-15B \
  --local-dir ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b

hf download zeyun-zhong/e2-ttt-swiglu-1.3B-15B \
  --local-dir ~/rivermind-data/models/reproductions/e2_ttt/swiglu-1.3b-15b
```

## 环境与评估命令

从服务器仓库根目录安装并激活环境；已安装时只需激活。依赖固定于 [requirements-eval.txt](requirements-eval.txt)，版本及安装说明见[评估环境](../20261109_environment_setup.md)。

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh e2_ttt
source ~/rivermind-data/envs/ttt-e2_ttt/bin/activate
```

MLP 和 SwiGLU 分别执行以下命令，使用 `optimized`，每卡 batch 上限为 8。已完成 4K／16K 的代表性对照，速度、显存及生成／评分差异见[批量推理记录](../20261110_e2_ttt_batch_inference.md)。

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
| `--inference-mode` | `original`（默认）或 `optimized`；后者用于本页两个公开 checkpoint 的等长批量推理 |
| `--batch-size` | 每卡上限，默认 `1`；`original` 仅允许 `1`，`optimized` 按截断后的实际 token 数分组 |
| `--benchmarks` | 默认 `ruler squad_completion swde`，可选择其中一项或多项 |
| `--lengths` | RULER 默认 `512 1024 2048 4096 8192 16384`，可选择所需长度 |
| `--ruler-data`／`--squad-data`／`--swde-data` | 默认分别使用评估范围表中的目录，可按需覆盖；相对路径以仓库根目录为准 |
| `--resume` | 使用原输出目录继续评估，保持原运行设置 |

`original` 对照使用 `--inference-mode original --batch-size 1`，并将输出目录改为独立的 `_original` 目录；不同 checkpoint 或推理模式不共用结果目录。

中断后在原命令末尾追加 `--resume`，保持 GPU 顺序、输出目录与其余参数一致。例如继续 MLP：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/e2_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --output-dir artifacts/reproductions/e2_ttt/eval/mlp-1.3b-15b_ruler-squad-completion-swde_optimized \
  --inference-mode optimized \
  --batch-size 8 \
  --resume
```

## 输入限制与结果

- RULER 使用 `input + answer_prefix`，不加 chat 模板，greedy 最多生成 128 tokens；输入加输出预算超限时报错，不截断。
- SQuAD／SWDE 保留原文本 `.strip()` 和换行停止，最多生成 48 tokens；总预算 2048，prompt 超过 2000 tokens 时按原流程左截断。
- `summary.json` 保存任务／长度汇总；逐题预测和原生结果文件见 `manifest.json`。`plan.json`、`execution.json` 保存参数、代码来源、运行状态与耗时。

本轮使用公开权重进行模型评估，无需重新训练。训练设置与备用入口见 [UPSTREAM.md](UPSTREAM.md) 和[官方训练说明](upstream/training/README.md)。
