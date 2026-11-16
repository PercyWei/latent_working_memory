# 20261108_In-Place TTT 第一阶段复现

创建时间：20261108 17:05:19 Asia/Shanghai

最后修订时间：20261116 21:06:35 Asia/Shanghai

## 评估目标与数据

首批评估第三方 Qwen3-4B 32K TTT 和 baseline 两个模型，采用完整 attention。两个模型均使用下表所列的 **9,100 条样本**进行评估，无需重新训练。

| benchmark | 任务与长度 | 样本数 | 固定数据路径 |
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

RULER 固定从表中路径读取，下载与构造命令见[评估数据准备](../20261108_ttt_data_preparation.md)。

## 环境与运行命令

从服务器项目根目录安装并激活独立环境；已安装时仅需激活：

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh in_place_ttt
source ~/rivermind-data/envs/ttt-in_place_ttt/bin/activate
```

依赖由 [requirements-eval.txt](requirements-eval.txt) 固定，版本及安装说明见[评估环境](../20261109_environment_setup.md)。

每卡逐条评估，batch size 固定为 1。TTT 使用仅压紧尾部缓存的路径，baseline 使用原始路径，MLP 均不分块：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/in_place_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --output-id qwen3-4b-32k-ttt_20261116 \
  --inference-mode optimized

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/in_place_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-baseline \
  --output-id qwen3-4b-32k-baseline_20261116 \
  --inference-mode original
```

| 参数 | 默认值与约定 |
|---|---|
| `--resume` | 沿用原评估协议补齐未完成样本的评估，默认不开启 |
| `--checkpoint` | 必需，本地完整 Hugging Face checkpoint 目录 |
| `--output-root` | 产物父目录，默认 `artifacts/reproductions/in_place_ttt` |
| `--output-id` | 默认 `<checkpoint_name>_<YYYYMMDD-HHMM>`，时间采用 `Asia/Shanghai` |
| `--inference-mode` | `original`（默认）或 `optimized` |
| `--mlp-chunk-size` | 默认 `0`；`original` 仅接受 `0`；`optimized` 设为 `0` 时仅压紧尾部缓存，正值同时启用 MLP 分块，TTT 模型的分块长度须为 TTT chunk 的整数倍 |
| `--lengths` | 默认 `4096 8192 16384 32768 65536 131072 262144` |

- 评估产物保存在 `<output-root>/eval/<output-id>/ruler/`；新运行在已有同名目录时拒绝覆盖。续评指定原 `--output-id` 并追加 `--resume`，checkpoint、长度、推理模式、分块和生成／评分协议保持原计划不变，允许调整 GPU 数量和顺序。
- 输入为 `input + answer_prefix`，greedy 生成最多 64 tokens，保留 EOS；输入加生成预算不得超过所选长度，不截断。评分沿用原 OpenCompass evaluator，长度分数为 13 项任务得分的等权平均。
- 单卡和多卡均经同一启动器调度，每卡一个进程，不提供 batch 参数。同一评估使用独占运行目录锁；已有评估或历史 worker 仍存活时拒绝重复启动。

源码与优化适用范围见 [UPSTREAM.md](UPSTREAM.md) 及[复现状态](../20261110_ttt_reproduction_status.md)，训练与 checkpoint 转换见[官方说明](upstream/README.md)。

## 来源记录与输出

以下文件均位于 `<output-root>/eval/<output-id>/ruler/`。终端显示总体完成数、百分比、吞吐和已耗时；续评按全局样本 ID 重新分配未完成样本。

| 文件 | 内容 |
|---|---|
| `summary.json` | 全部所选样本完成后的任务／长度评分汇总 |
| `evaluation_plan.json` | 固定 checkpoint、数据及生成／评分协议 |
| `launcher_execution.json` | 各次启动状态与执行参数 |
| `runtime.json` | 模型加载信息与运行环境 |
| `attempts/attempt-<编号>/plan.json`、`execution.json` | 本轮计划、来源和执行状态 |
| `attempts/attempt-<编号>/logs/shard-<i>.log`、`assignments/shard-<i>.json` | 各卡日志和样本分配 |
| `attempts/attempt-<编号>/shards/shard-<i>/` | 各卡的计划、执行状态、环境及 `predictions.jsonl` |
| `attempts/attempt-<编号>/merged/` | 全部预测、来源 `manifest.json`、汇总及 OpenCompass `results/` |

最终分数读取 RULER 目录内的 `summary.json`，其中 `merged_summary_path` 指向本轮完整汇总。耗时估算见[运行参数测试](../20261110_ttt_evaluation_tuning.md)。
