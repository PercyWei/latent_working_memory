# 20261108_In-Place TTT 第一阶段复现

创建时间：20261108 17:05:19 Asia/Shanghai

最后修订时间：20261126 17:07:15 Asia/Shanghai

[环境准备](#一环境准备) · [原论文复现](#二原论文复现) · [通用功能](#三通用功能)

## 一、环境准备

推理默认以 BF16 加载模型权重，关闭模型级自动混合精度（autocast）。全注意力使用 FlashAttention 2。

从服务器项目根目录安装并激活独立评估环境，版本见[评估环境](../20261109_environment_setup.md)：

```bash
bash reproductions/setup_eval_env.sh in_place_ttt
source ~/envs/in_place_ttt/bin/activate
```

环境位置、uv 镜像配置与激活方式见[评估环境](../README.md#评估环境)。

## 二、原论文复现

本项目使用第三方 Qwen3-4B 32K TTT 和 32K baseline 两个 checkpoint 评估 RULER。作者未公开论文 Table 1 对应的权重，表中的分数用于对照。[作者说明](https://github.com/ByteDance-Seed/In-Place-TTT/issues/1)

源码与优化适用范围见 [UPSTREAM.md](UPSTREAM.md) 及[复现状态](../20261110_ttt_reproduction_status.md)，训练与 checkpoint 转换见[官方说明](source/in_place_ttt/README.md)。

<a id="模型与数据准备"></a>

### 2.1 模型与数据准备

下载第三方 [32K TTT](https://huggingface.co/zhongweixie/inplace-ttt-qwen3-4b-32k-ttt) 和 [32K baseline](https://huggingface.co/zhongweixie/inplace-ttt-qwen3-4b-32k-baseline)。两者为完整 Hugging Face checkpoint，模型卡记录使用 ProLong 训练 6,000 steps。`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
hf download zhongweixie/inplace-ttt-qwen3-4b-32k-ttt \
  --local-dir ~/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt

hf download zhongweixie/inplace-ttt-qwen3-4b-32k-baseline \
  --local-dir ~/models/reproductions/in_place_ttt/qwen3-4b-32k-baseline
```

两个模型均在 32K 上训练。超过 32K 的评估属于长度外推，沿用 checkpoint 的 RoPE 配置，不额外启用 YaRN。

两个模型共用下表的 **9,100 条 RULER 样本**。下载与构造命令见[评估数据准备](../20261108_ttt_data_preparation.md)。

| benchmark | 任务与长度 | 样本数 | 固定数据路径 |
|---|---|---:|---|
| RULER | 13 项任务；4K／8K／16K／32K／64K／128K／256K；每项任务、每个长度 100 条 | 9,100 | `data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b` |

### 2.2 RULER 评估

按[环境准备](#一环境准备)激活环境后，从项目根目录运行。TTT 和 baseline 均支持 `optimized` 运行，验证范围见[复现状态记录](../20261110_ttt_reproduction_status.md#in-place-ttt)。

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
- `optimized` 的批量路径支持与序列长度无关的 `default`、`linear`、`yarn` 和 `llama3` RoPE。
- 单卡和多卡均经同一启动器调度，每卡一个进程。同一评估使用独占运行目录锁；已有评估或历史 worker 仍存活时拒绝重复启动。

#### 来源记录与输出

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

### 2.3 论文复现对照

下表是论文 Table 1 在 13 项 RULER 任务上的平均准确率（%）。

| 模型 | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原文 baseline | 96.6 | 94.1 | 92.1 | 88.7 | 74.3 | 74.8 | 41.7 |
| 原文 In-Place TTT | 96.1 | 95.6 | 92.7 | 89.3 | 78.7 | 77.0 | 43.9 |

```bash
python reproductions/in_place_ttt/evaluation/compare_results.py \
  --eval-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-ttt_20261117
```

`--eval-dir` 指向 `<output-root>/eval/<output-id>`，报告保存为 `<eval-dir>/comparison.html`。

TTT／baseline 自动匹配论文参照；要求本次 RULER 计划完成，论文仅有 13 项任务平均分。

## 三、通用功能

### 3.1 benchmark 评估

| 完整评估脚本共同参数 | 用途与默认值 |
|---|---|
| `--checkpoint` | 必填，本地完整的 checkpoint 目录 |
| `--benchmark-data` | 输入数据目录，默认路径见各任务小节 |
| `--output-root` | 输出根目录，默认 `artifacts/reproductions/in_place_ttt` |
| `--output-id` | 运行 ID，默认 `<checkpoint_name>_<YYYYMMDD-HHMM>`，续评估时必填 |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |
| `--mlp-chunk-size` | MLP 分块长度（token 数），默认 `0`（关闭）。TTT 模型启用时须取 TTT chunk 的正整数倍 |
| `--resume` | 用于续评估，默认关闭 |

- **运行方式**：在项目根目录使用 In-Place TTT 环境运行。每张可见 GPU 加载一个模型副本，使用 `optimized` 模式推理。[环境准备](#一环境准备) · [数据下载](../../benchmarks/README.md)
- **输入与解码**：将系统提示、会话历史和问题拼成纯文本，在 `Answer:` 后贪心生成回答。
- **结果与续评**：各评估设置的结果分别保存在 `<output-root>/eval/<output-id>/` 下的对应子目录。`--resume` 复用已保存的回答和评分，只补齐未完成的部分。

<a id="locomo-评估"></a>
<a id="31-locomo-评估"></a>

#### 1. LoCoMo

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |

- **评估范围**：覆盖 10 段完整对话中类别 1–4 的 1,540 道题。[LoCoMo 评估协议](../../benchmarks/locomo/README.md)
- **推理方式**：按对话分配 GPU，每段对话历史只预填一次。该对话的各题复用预填好的 KV 缓存，TTT 模型同时复用预填后的快速权重，各题独立作答。
- **回答长度与评分**：每题最多生成 50 个 token，输入与生成预算之和须在 checkpoint 的 32,768 个位置以内。按官方规则逐题计算 F1，再对所有题目取平均。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/in_place_ttt/scripts/eval_locomo.sh \
  --checkpoint ~/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --output-id qwen3-4b-32k-ttt_qa-eval \
  --batch-size-per-gpu 2
```

<a id="32-longmemeval-评估"></a>

#### 2. LongMemEval

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/longmemeval` |
| `--split` | 支持 `longmemeval_s_cleaned`（默认）和 `longmemeval_oracle`  |
| `--max-new-tokens` | 每题最多生成的 token 数，默认 `128` |
| `--max-seq-len` | 输入与生成的 token 总预算，默认取 checkpoint 的 `max_position_embeddings`（两个 32K 模型均为 `32768`） |

- **评估范围**：S 和 Oracle 使用相同的 500 道题。S 提供证据会话和干扰会话，Oracle 只提供证据会话。
- **推理方式**：相同的完整历史归到同一卡，按日期顺序并行预填新历史。各题从对应的正文状态独立续写，结合提问日期并行作答。
- **位置预算**：Oracle 可使用默认的 32K 上限。S 示例将上限设为 `131072`，沿用 checkpoint 的 RoPE 配置做长度外推，外推效果尚未实测。
- **自动评分**：生成全部回答后，脚本调用 `gpt-6-luna`，按官方分题型提示词逐题判定正误，汇总准确率。裁判密钥 `OPENAI_API_KEY` 可在项目根目录的 `.env` 中配置。[裁判配置与评分口径](../../benchmarks/longmemeval/README.md#问答评分)

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/in_place_ttt/scripts/eval_longmemeval.sh \
  --split longmemeval_s_cleaned \
  --max-seq-len 131072 \
  --checkpoint ~/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --output-id qwen3-4b-32k-ttt_qa-eval
```

| 独立评分代码参数 | 用途与默认值 |
|---|---|
| `--run-dir` | 必填，所选设置的结果目录 |
| `--benchmark-data` | 默认沿用生成时记录的数据目录 |
| `--resume` | 用于续评分，默认关闭 |

- **独立评分**：已有完整回答时，可在项目根环境中直接评分，无需重新运行 In-Place TTT 推理。

```bash
uv run -m benchmarks.longmemeval.score \
  --run-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-ttt_qa-eval/longmemeval_s_cleaned
```

<a id="33-结果展示"></a>

### 3.2 结果展示

以上示例共用一个运行目录，三个设置分别保存在 `locomo/`、`longmemeval_s_cleaned/` 和 `longmemeval_oracle/`。在看板中选择已评分的设置，可查看总体和分题型指标（LoCoMo 为 F1，LongMemEval 为准确率）、题数和逐题回答。每个设置的 `summary.json` 保存汇总，`details.jsonl` 保存逐题评分。LongMemEval 的生成回答另存为 `predictions.jsonl`。

```bash
bash reproductions/in_place_ttt/scripts/dashboard.sh \
  --output-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-ttt_qa-eval
```
