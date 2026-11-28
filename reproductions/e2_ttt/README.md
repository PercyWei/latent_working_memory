# 20261108_E²-TTT 第一阶段复现

创建时间：20261108 17:02:06 Asia/Shanghai

最后修订时间：20261128 15:17:55 Asia/Shanghai

[环境准备](#一环境准备) · [原论文复现](#二原论文复现) · [通用功能](#三通用功能)

## 一、环境准备

推理默认以 BF16 加载模型权重，关闭模型级自动混合精度（autocast）。窗口注意力使用 FlashAttention 2。TTT 状态更新和归一化沿用模型实现的计算精度。

从服务器项目根目录安装并激活独立评估环境，版本见[评估环境](../20261109_environment_setup.md)：

```bash
bash reproductions/setup_eval_env.sh e2_ttt
source ~/envs/e2_ttt/bin/activate
```

环境位置、uv 镜像配置与激活方式见[评估环境](../README.md#评估环境)。

## 二、原论文复现

本目录使用 [Rethinking Expressivity and Efficiency in Test-Time Training](https://arxiv.org/abs/2608.21308) 的官方实现和作者公开 checkpoint。

源码与本地优化边界见 [UPSTREAM.md](UPSTREAM.md)，验证结论见[复现状态](../20261110_ttt_reproduction_status.md)。

<a id="模型下载"></a>

### 2.1 模型下载

作者公开权重见 [zeyun-zhong/e2-ttt](https://huggingface.co/collections/zeyun-zhong/e2-ttt)，`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
hf download zeyun-zhong/e2-ttt-mlp-1.3B-15B \
  --local-dir ~/models/reproductions/e2_ttt/mlp-1.3b-15b

hf download zeyun-zhong/e2-ttt-swiglu-1.3B-15B \
  --local-dir ~/models/reproductions/e2_ttt/swiglu-1.3b-15b
```

### 2.2 评估目标与数据

MLP 和 SwiGLU 两个 1.3B 模型均使用下表所列的 **10,095 条样本**进行评估。下载和构造命令见[评估数据准备](../20261108_ttt_data_preparation.md)。数据固定从下表路径读取。

| benchmark | 任务与长度 | 样本数 | 固定数据路径 |
|---|---|---:|---|
| RULER | S-NIAH-1／2；512、1K、2K、4K、8K、16K；每项任务、每个长度 500 条 | 6,000 | `data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b` |
| SQuAD completion | 完整 validation | 2,984 | `data/benchmarks/based-squad/data/validation-00000-of-00001.parquet` |
| SWDE | 完整 validation | 1,111 | `data/benchmarks/based-swde/data/validation-00000-of-00001.parquet` |

### 2.3 复现评估

按[环境准备](#一环境准备)激活评估环境后，在项目根目录运行以下命令。示例使用 MLP checkpoint，评估 SwiGLU 时将 `--checkpoint` 指向对应模型目录。两个模型均支持 `optimized` 推理，验证范围见[复现状态记录](../20261110_ttt_reproduction_status.md#e-ttt)。

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/e2_ttt/scripts/eval.sh \
  --checkpoint ~/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --output-id mlp-1.3b-15b_20261117 \
  --inference-mode optimized \
  --batch-size 32 \
  --batch-size-by-length 512=64 1024=32 2048=32 4096=16 8192=8 16384=8
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

#### 来源记录与输出

最终分数分别读取 `<output-root>/eval/<output-id>/<benchmark>/summary.json`。`scripts/eval.sh` 调用 `evaluation/evaluate_parallel.py` 统一调度，每个 worker 加载一次模型，依次处理分配到的多个 benchmark。[进度展示](../README.md#进度展示)采用三方法的共同规则。

| 文件 | 内容 |
|---|---|
| `evaluation_plan.json`、`launcher_execution.json`、`attempts/` | 父目录内的总评估计划、各次启动设置、代码来源与 worker 日志 |
| `attempts/attempt-<编号>/workers/worker-<i>/progress.json` | 父目录内当前 worker 的独立进度快照，仅用于展示 |
| `summary.json` | 各 benchmark 的汇总位置及整次运行的完成状态 |
| `<benchmark>/evaluation_plan.json` | 该 benchmark 的固定数据及生成／评分协议 |
| `<benchmark>/summary.json` | 该 benchmark 全部样本完成后的任务／长度分数 |
| `<benchmark>/attempts/` | 各次 worker 的原始逐题结果、原生评分和生成缓存 |
| `<benchmark>/predictions.jsonl`、`<benchmark>/manifest.json` | 该 benchmark 合并后的逐题结果及其原始记录位置 |
| `comparison.html` | 执行对比脚本后在总目录生成的图表与表格 |

### 2.4 论文复现对照

论文 `tab:incontext_retrieval_main` 中 **1.3B 参数规模模型**的准确率见下表。RULER 参照原文长上下文主图及附录的 S-NIAH 数值表。

| 指标 | MLP | SwiGLU |
|---|---:|---:|
| SWDE accuracy (%) | 37.2 | 47.3 |
| SQuAD accuracy (%) | 45.6 | 44.4 |

```bash
python reproductions/e2_ttt/evaluation/compare_results.py \
  --eval-dir artifacts/reproductions/e2_ttt/eval/mlp-1.3b-15b_20261117
```

`--eval-dir` 指向 `<output-root>/eval/<output-id>`，报告保存为 `<eval-dir>/comparison.html`。

MLP／SwiGLU 自动匹配论文参照；未评估完成的 benchmark 显示 `—`。

## 三、通用功能

### 3.1 benchmark 评估

| 完整评估脚本共同参数 | 用途与默认值 |
|---|---|
| `--checkpoint` | 必填，本地完整的 checkpoint 目录 |
| `--benchmark-data` | 输入数据目录，默认路径见各任务小节 |
| `--output-root` | 输出根目录，默认 `artifacts/reproductions/e2_ttt` |
| `--output-id` | 运行 ID，默认 `<checkpoint_name>_<YYYYMMDD-HHMM>`，续评估时必填 |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |
| `--resume` | 用于续评估，默认关闭 |

- **运行方式**：在项目根目录使用 E²-TTT 环境运行。每张可见 GPU 加载一个模型副本，使用 `optimized` 模式推理。[环境准备](#一环境准备) · [数据下载](../../benchmarks/README.md)
- **输入与解码**：将系统提示、会话历史和问题拼成纯文本，在 `Answer:` 后贪心生成回答。
- **结果与续评**：各评估设置的结果分别保存在 `<output-root>/eval/<output-id>/` 下的对应子目录。`--resume` 复用已保存的回答和评分，只补齐未完成的部分。

<a id="locomo-评估"></a>
<a id="31-locomo-评估"></a>

#### 1. LoCoMo

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |
| `--no-llm-as-judge` | 跳过 Judge Accuracy，只计算官方 F1 |

- **评估范围**：覆盖 10 段完整对话中类别 1–4 的 1,540 道题。[LoCoMo 评估协议](../../benchmarks/locomo/README.md)
- **推理方式**：按对话分配 GPU，每段对话历史只预填一次。该对话的各题复用预填好的递归状态和窗口注意力缓存，按各自的真实位置独立作答。
- **回答长度与评分**：每题最多生成 50 个 token。按官方规则逐题计算 F1，再对所有题目取平均。保存 F1 后默认计算 Judge Accuracy。[裁判配置与补评](../../benchmarks/README.md#judge-accuracy)
- **验证范围**：LoCoMo 适配已完成静态检查，CUDA 数值对照和全量评估待完成。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/e2_ttt/scripts/eval_locomo.sh \
  --checkpoint ~/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --output-id mlp-1.3b-15b_qa-eval \
  --batch-size-per-gpu 2
```

<a id="32-longmemeval-评估"></a>

#### 2. LongMemEval

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/longmemeval` |
| `--split` | 支持 `longmemeval_s_cleaned`（默认）和 `longmemeval_oracle`  |
| `--max-new-tokens` | 每题最多生成的 token 数，默认 `128` |

- **评估范围**：S 和 Oracle 使用相同的 500 道题。S 提供证据会话和干扰会话，Oracle 只提供证据会话。
- **推理方式**：相同的完整历史归到同一卡，按日期顺序并行预填新历史。各题从对应的正文状态独立续写，结合提问日期并行作答。
- **评估长度**：S 的会话历史通常超过本项目 RULER 复现使用的最长设置（16K）。
- **自动评分**：生成全部回答后，脚本调用 `gpt-6-luna`，按官方分题型提示词逐题判定正误，汇总准确率。裁判密钥 `OPENAI_API_KEY` 可在项目根目录的 `.env` 中配置。[裁判配置与评分口径](../../benchmarks/longmemeval/README.md#问答评分)

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/e2_ttt/scripts/eval_longmemeval.sh \
  --split longmemeval_s_cleaned \
  --checkpoint ~/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --output-id mlp-1.3b-15b_qa-eval
```

| 独立评分代码参数 | 用途与默认值 |
|---|---|
| `--run-dir` | 必填，所选设置的结果目录 |
| `--benchmark-data` | 默认沿用生成时记录的数据目录 |
| `--resume` | 用于续评分，默认关闭 |

- **独立评分**：已有完整回答时，可在项目根环境中直接评分，无需重新运行 E²-TTT 推理。

```bash
uv run -m benchmarks.longmemeval.score \
  --run-dir artifacts/reproductions/e2_ttt/eval/mlp-1.3b-15b_qa-eval/longmemeval_s_cleaned
```

<a id="33-结果展示"></a>

### 3.2 结果展示

以上示例共用一个运行目录，三个设置分别保存在 `locomo/`、`longmemeval_s_cleaned/` 和 `longmemeval_oracle/`。在看板中选择已评分的设置，可查看总体和分题型指标（LoCoMo 为 F1，LongMemEval 为准确率）、题数和逐题回答。每个设置的 `summary.json` 保存汇总，`details.jsonl` 保存逐题评分。LongMemEval 的生成回答另存为 `predictions.jsonl`。

```bash
bash reproductions/e2_ttt/scripts/dashboard.sh \
  --output-dir artifacts/reproductions/e2_ttt/eval/mlp-1.3b-15b_qa-eval
```
