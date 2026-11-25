# 20261108_In-Place TTT 第一阶段复现

创建时间：20261108 17:05:19 Asia/Shanghai

最后修订时间：20261125 10:58:32 Asia/Shanghai

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
  --local-dir ~/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt

hf download zhongweixie/inplace-ttt-qwen3-4b-32k-baseline \
  --local-dir ~/models/reproductions/in_place_ttt/qwen3-4b-32k-baseline
```

RULER 固定从表中路径读取，下载与构造命令见[评估数据准备](../20261108_ttt_data_preparation.md)。

## 环境与运行命令

从服务器项目根目录安装并激活独立评估环境，版本见[评估环境](../20261109_environment_setup.md)：

```bash
bash reproductions/setup_eval_env.sh in_place_ttt
source ~/envs/ttt-in_place_ttt/bin/activate
```

自定义位置与激活方式见[评估环境](../README.md#评估环境)。

TTT 和 baseline 均支持 `optimized` 运行，验证范围见[复现状态记录](../20261110_ttt_reproduction_status.md#in-place-ttt)。

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/in_place_ttt/scripts/eval.sh \
  --checkpoint ~/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --output-id qwen3-4b-32k-ttt_20261117 \
  --inference-mode optimized \
  --batch-size 1 \
  --batch-size-by-length 4096=8 8192=4 16384=4 32768=2
```

| 参数 | 默认值与约定 |
|---|---|
| `--resume` | 沿用原评估协议补齐未完成样本的评估，默认不开启 |
| `--checkpoint` | 必需，本地完整 checkpoint 目录 |
| `--output-root` | 产物父目录，默认 `artifacts/reproductions/in_place_ttt` |
| `--output-id` | 默认 `<checkpoint_name>_<YYYYMMDD-HHMM>` |
| `--inference-mode` | 推理模式，支持 `original`（默认）或 `optimized` |
| `--batch-size` | 每卡样本数上限，默认 `1`；只有 `optimized` 支持大于 1 |
| `--batch-size-by-length` | 按评估长度覆盖每卡上限，未指定的长度使用 `--batch-size` |
| `--mlp-chunk-size` | MLP 分块长度（token 数），默认 `0` 不启用；只有 `optimized` 支持正值，TTT 模型要求其为 TTT chunk 的整数倍 |
| `--lengths` | 默认 `4096 8192 16384 32768 65536 131072 262144` |

- 评估产物保存在 `<output-root>/eval/<output-id>/ruler/`；新运行在已有同名目录时拒绝覆盖。续评指定原 `--output-id` 并追加 `--resume`，checkpoint、长度、推理模式、分块和生成／评分协议保持原计划不变，允许调整 GPU 数量、顺序与 batch 设置。
- 输入为 `input + answer_prefix`，greedy 生成最多 64 tokens，沿用原停止条件；各行在首次结束后冻结状态，生成 IDs 截至各自结束位置，保留 EOS。输入加生成预算不得超过所选长度，不截断。评分沿用原 OpenCompass evaluator，长度分数为 13 项任务得分的等权平均。
- 每个 RULER 长度内按实际 token 数稳定降序，相邻组批并左 padding 到批内最长输入，尾批保留实际样本数。每条样本独立维护真实位置、attention KV 和 TTT 状态；padding 不参与记忆读写或推进更新。完整 attention、原 TTT chunk 的读写顺序及卷积边界保持不变。
- `optimized` 使用原生 SDPA 的完整 attention，支持 `default`、`linear`、`yarn`、`llama3` RoPE；不支持 `dynamic` 或 `longrope`。
- 单卡和多卡均经同一启动器调度，每卡一个进程。同一评估使用独占运行目录锁；已有评估或历史 worker 仍存活时拒绝重复启动。

源码与优化适用范围见 [UPSTREAM.md](UPSTREAM.md) 及[复现状态](../20261110_ttt_reproduction_status.md)，训练与 checkpoint 转换见[官方说明](source/in_place_ttt/README.md)。

## LoCoMo 评估

状态与生成的小模型 CPU 对照已通过；完整 LoCoMo 评估尚未运行。

在上述独立环境中运行。数据准备见[公共数据说明](../../benchmarks/README.md)，评估范围、F1 评分和续评规则见[公共 LoCoMo 说明](../../benchmarks/locomo/README.md)。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/in_place_ttt/scripts/eval_locomo.sh \
  --checkpoint ~/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --output-id qwen3-4b-32k-ttt_20261117 \
  --batch-size-per-gpu 2
```

| 参数 | 默认值与约定 |
|---|---|
| `--resume` | 沿用原 ID、checkpoint 和数据，补齐未完成的问题 |
| `--checkpoint` | 必填，本地完整 checkpoint 目录 |
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |
| `--output-root` | 默认 `artifacts/reproductions/in_place_ttt` |
| `--output-id` | 默认 `<checkpoint_name>_<YYYYMMDD-HHMM>` |
| `--batch-size-per-gpu` | 每卡评估批量大小，默认 1 |
| `--mlp-chunk-size` | 默认 0；设为 TTT chunk 的整数倍可减少 MLP 中间激活 |

- 使用 `optimized` 批量推理、BF16 权重和 SDPA。完整输入与生成预算须在模型位置窗口内，超窗时报错。
- 不套用聊天模板，由公共接口构造输入：`QA 系统提示 → 完整对话 → 简答指令与问题 → Answer:`。
- 每段对话只预填一次，各题从这份 KV 和 TTT 状态独立作答，互不继承其他问题或答案。
- 结果保存在 `<output-root>/eval/<output-id>/locomo/`。`summary.json` 汇总总体及各题型的 F1 和题数，`details.jsonl` 保存逐题回答与评分。

## 查看结果

### Benchmark 结果

从项目根目录启动独立看板，当前支持 LoCoMo：

```bash
bash reproductions/in_place_ttt/scripts/dashboard.sh \
  --output-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-ttt_20261117
```

`--output-dir` 指向包含 `locomo/` 的运行目录。看板使用项目根环境，展示总体及各题型 F1、题数，并可按题型与对话查看逐题回答和参考答案。

### 论文复现对照

```bash
uv run reproductions/in_place_ttt/evaluation/compare_results.py \
  --eval-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-ttt_20261117
```

`--eval-dir` 指向 `<output-root>/eval/<output-id>`，报告保存为 `<eval-dir>/comparison.html`。

TTT／baseline 自动匹配论文参照；要求本次 RULER 计划完成，论文仅有 13 项任务平均分。

## 来源记录与输出

以下评估记录位于 `<output-root>/eval/<output-id>/ruler/`，对比报告 `comparison.html` 位于父目录 `<output-id>/`。`scripts/eval.sh` 调用 `evaluation/evaluate_parallel.py` 统一调度，[进度展示](../README.md#进度展示)采用三方法的共同规则；续评按全局样本 ID 重新分配未完成样本。

| 文件 | 内容 |
|---|---|
| `summary.json` | 全部所选样本完成后的任务／长度评分汇总 |
| `evaluation_plan.json` | 固定 checkpoint、数据及生成／评分协议 |
| `launcher_execution.json` | 各次启动状态与执行参数 |
| `runtime.json` | 模型加载信息与运行环境 |
| `attempts/attempt-<编号>/shards/shard-<i>/progress.json` | 当前 worker 的独立进度快照，仅用于展示 |
| `attempts/attempt-<编号>/plan.json`、`execution.json` | 本轮计划、来源和执行状态 |
| `attempts/attempt-<编号>/logs/shard-<i>.log`、`assignments/shard-<i>.json` | 各卡日志和样本分配 |
| `attempts/attempt-<编号>/shards/shard-<i>/` | 各卡的计划、执行状态、环境及 `predictions.jsonl` |
| `attempts/attempt-<编号>/merged/` | 全部预测、来源 `manifest.json`、汇总及 OpenCompass `results/` |
| `comparison.html` | 执行对比脚本后在父目录生成的图表与表格 |

最终分数读取 RULER 目录内的 `summary.json`，其中 `merged_summary_path` 指向本轮完整汇总。此前 A800 的 B1 测量及外推见[历史 B1 测量](../20261110_ttt_reproduction_status.md#历史-b1-测量)。
