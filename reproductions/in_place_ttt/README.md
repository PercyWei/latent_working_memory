# 20261108_In-Place TTT 第一阶段复现

创建时间：20261108 17:05:19 Asia/Shanghai

最后修订时间：20261110 20:22:59 Asia/Shanghai

## 评估目标

首批评估第三方 Qwen3-4B 32K TTT 和 baseline 两个模型，采用完整 attention。两个模型均使用下表所列的 **9,100 条样本**进行评估，无需重新训练。

| benchmark | 任务与长度 | 样本数 | 默认数据目录 |
|---|---|---:|---|
| RULER | 13 项任务；4K／8K／16K／32K／64K／128K／256K；每项任务、每个长度 100 条 | 9,100 | `data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b` |

当前权重不对应原文 Table 1 的官方模型。原文 RULER 的 13 项任务平均准确率（%）仅作参照：

| 模型 | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原文 baseline | 96.6 | 94.1 | 92.1 | 88.7 | 74.3 | 74.8 | 41.7 |
| 原文 In-Place TTT | 96.1 | 95.6 | 92.7 | 89.3 | 78.7 | 77.0 | 43.9 |

第三方模型在 32K 上训练，**超过 32K 属于长度外推**；评估保留 checkpoint 的 RoPE 配置，不自动添加 YaRN。作者未公开对应原文权重，见[官方回复](https://github.com/ByteDance-Seed/In-Place-TTT/issues/1)。

## 模型与数据准备

下载第三方 [32K TTT](https://huggingface.co/zhongweixie/inplace-ttt-qwen3-4b-32k-ttt) 和 [32K baseline](https://huggingface.co/zhongweixie/inplace-ttt-qwen3-4b-32k-baseline)。两者为完整 Hugging Face checkpoint，模型卡记录使用 ProLong 训练 6,000 steps。`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
hf download zhongweixie/inplace-ttt-qwen3-4b-32k-ttt \
  --local-dir ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt

hf download zhongweixie/inplace-ttt-qwen3-4b-32k-baseline \
  --local-dir ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-baseline
```

RULER 下载与构造命令见[评估数据准备](../20261108_ttt_data_preparation.md)。

## 环境

从服务器项目根目录安装并激活独立环境；已安装时仅需激活：

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh in_place_ttt
source ~/rivermind-data/envs/ttt-in_place_ttt/bin/activate
```

依赖由 [requirements-eval.txt](requirements-eval.txt) 固定，版本及安装说明见[评估环境](../20261109_environment_setup.md)。

## RULER 评估

以下命令均从仓库根目录执行。

每进程 batch size 固定为 1。TTT 使用仅压紧尾部缓存的路径，baseline 使用原始路径，MLP 均不分块：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/in_place_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --output-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-ttt_ruler \
  --inference-mode optimized \
  --mlp-chunk-size 0

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/in_place_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-baseline \
  --output-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-baseline_ruler \
  --inference-mode original
```

| 参数 | 默认值与约定 |
|---|---|
| `--checkpoint` | 必需，本地完整 Hugging Face checkpoint 目录 |
| `--output-dir` | 必需，首次运行时必须尚不存在 |
| `--inference-mode` | `original`（默认）或 `optimized` |
| `--mlp-chunk-size` | 优化模式默认 `16384`；本轮显式使用 `0`，仅压紧尾部缓存；原始模式不使用 MLP 分块 |
| `--ruler-data` | `data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b` |
| `--lengths` | `4096 8192 16384 32768 65536 131072 262144` |
| `--resume` | 使用原设置和输出目录继续 |

首次运行的输出目录须不存在；中断后保持相同 GPU 顺序、checkpoint、数据、推理参数与输出目录，追加 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/in_place_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --output-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-ttt_ruler \
  --inference-mode optimized \
  --mlp-chunk-size 0 \
  --resume
```

输入为 `input + answer_prefix`，greedy 生成最多 64 tokens，保留 EOS；输入加生成预算不得超过所选长度，不截断。评分沿用原 OpenCompass evaluator，长度分数为 13 项任务得分的等权平均。

本次正式评估不启用 MLP 分块。源码与优化适用范围见 [UPSTREAM.md](UPSTREAM.md) 及[复现状态](../20261110_ttt_reproduction_status.md)，训练与 checkpoint 转换见[官方说明](upstream/README.md)。

## 输出

| 文件 | 内容 |
|---|---|
| `summary.json` | 全部所选数据的任务／长度评分汇总 |
| `merged/predictions.jsonl` | 上述运行命令的逐题预测、评分和完整生成 IDs |
| `plan.json`、`execution.json` | 运行参数、代码来源、执行状态与耗时 |

运行完成后读取输出根目录的 `summary.json`；耗时估算见[运行参数测试](../20261110_ttt_evaluation_tuning.md)。
