# 20261124_AutoCompressors 复现

创建时间：20261124 11:54:28 Asia/Shanghai

最后修订时间：20261126 15:14:03 Asia/Shanghai

[环境准备](#一环境准备) · [原论文复现](#二原论文复现) · [通用功能](#三通用功能)

## 一、环境准备

推理默认以 BF16 加载模型权重，关闭模型级自动混合精度（autocast）。注意力由官方 Llama 实现调用 FlashAttention 2 计算。

从 Linux x86-64 服务器的项目根目录创建并激活独立评估环境。依赖版本见 [requirements-eval.txt](requirements-eval.txt)。固定版本的官方源码随项目保存在 [source/AutoCompressors](source/AutoCompressors/)，安装器仅负责创建环境和安装依赖。

```bash
bash reproductions/setup_eval_env.sh autocompressors
source ~/envs/autocompressors/bin/activate
```

环境位置、uv 镜像配置与激活方式见[评估环境](../README.md#评估环境)。

## 二、原论文复现

本项目采用 AutoCompressors 的官方实现。Princeton 发布的 AutoCompressor-Llama-2-7b-6k 基于 Llama-2-7B 进行语言建模训练，未做 QA 微调。作者提供了 Llama 的[训练脚本](source/AutoCompressors/run/train_llama.sh)和[困惑度评估脚本](source/AutoCompressors/run/eval_llama.sh)。源码与项目适配见 [UPSTREAM.md](UPSTREAM.md)。

### 2.1 模型下载

下载 [AutoCompressor-Llama-2-7b-6k](https://huggingface.co/princeton-nlp/AutoCompressor-Llama-2-7b-6k) 的完整权重与 tokenizer。`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
hf download princeton-nlp/AutoCompressor-Llama-2-7b-6k \
  --local-dir ~/models/reproductions/autocompressors/AutoCompressor-Llama-2-7b-6k
```

## 三、通用功能

### 3.1 benchmark 评估

| 完整评估脚本共同参数 | 用途与默认值 |
|---|---|
| `--checkpoint` | 必填，本地完整的 checkpoint 目录 |
| `--segment-length` | 每段正文的 token 数上限，默认值见各任务小节 |
| `--benchmark-data` | 输入数据目录，默认路径见各任务小节 |
| `--output-root` | 输出根目录，默认 `artifacts/reproductions/autocompressors` |
| `--output-id` | 运行 ID，默认 `<checkpoint_name>_seg<segment_length>_<YYYYMMDD-HHMM>`，续评估时必填 |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |
| `--resume` | 用于续评估，默认关闭 |

- **运行方式**：在项目根目录使用 AutoCompressors 环境运行。每张可见 GPU 加载一个模型副本。[环境准备](#一环境准备) · [数据下载](../../benchmarks/README.md)
- **历史压缩**：Writer 逐段压缩对话正文，每段（含末段）生成 50 个摘要向量。累积摘要与当前正文合计最多占用 4,096 个位置。
- **输入与解码**：Reader 依次读入 `BOS`、系统提示、全部摘要和问题，在 `Answer:` 后贪心生成回答。Reader 输入与回答预算之和也须在 4,096 个位置内。
- **结果与续评**：各评估设置的结果分别保存在 `<output-root>/eval/<output-id>/` 下的对应子目录。`--resume` 复用已保存的回答和评分，只补齐未完成的部分。

<a id="locomo-评估"></a>
<a id="31-locomo-评估"></a>

#### 1. LoCoMo

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |
| `--segment-length` | 默认 `2048` |

- **评估范围**：覆盖 10 段完整对话中类别 1–4 的 1,540 道题。[LoCoMo 评估协议](../../benchmarks/locomo/README.md)
- **推理方式**：按对话分配 GPU，每段对话历史只压缩一次。默认分段下，完整段的压缩率约为 41 倍。同一对话内按问题长度组批，各题复用摘要与 Reader 前缀缓存，独立作答。
- **回答长度与评分**：每题最多生成 50 个 token。按官方规则逐题计算 F1，再对所有题目取平均。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/autocompressors/scripts/eval_locomo.sh \
  --checkpoint ~/models/reproductions/autocompressors/AutoCompressor-Llama-2-7b-6k \
  --output-id AutoCompressor-Llama-2-7b-6k \
  --batch-size-per-gpu 2
```

<a id="32-longmemeval-评估"></a>

#### 2. LongMemEval

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/longmemeval` |
| `--split` | 支持 `longmemeval_s_cleaned`（默认）和 `longmemeval_oracle` |
| `--segment-length` | 默认 `2304` |
| `--max-new-tokens` | 每题最多生成的 token 数，默认 `128` |

- **评估范围**：S 和 Oracle 使用相同的 500 道题。S 提供证据会话和干扰会话，Oracle 只提供证据会话。
- **推理方式**：按题分配 GPU。每张卡逐题压缩按日期排列的会话历史，Reader 根据摘要、提问日期和问题作答。随着摘要累积，Writer 按剩余位置缩短后续正文段，为本段新增的 50 个摘要向量留出空间。
- **输入长度**：按模型 tokenizer 统计，S 的历史长 120,431–132,553 个 token。使用默认分段长度与回答预算时，500 道 S 题均能放入 Writer 和 Reader 窗口。
- **自动评分**：生成全部回答后，脚本调用 `gpt-6-luna`，按官方分题型提示词逐题判定正误，汇总准确率。裁判密钥 `OPENAI_API_KEY` 可在项目根目录的 `.env` 中配置。[裁判配置与评分口径](../../benchmarks/longmemeval/README.md#问答评分)

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/autocompressors/scripts/eval_longmemeval.sh \
  --split longmemeval_s_cleaned \
  --checkpoint ~/models/reproductions/autocompressors/AutoCompressor-Llama-2-7b-6k \
  --output-id AutoCompressor-Llama-2-7b-6k
```

| 独立评分代码参数 | 用途与默认值 |
|---|---|
| `--run-dir` | 必填，所选设置的结果目录 |
| `--benchmark-data` | 默认沿用生成时记录的数据目录 |
| `--resume` | 用于续评分，默认关闭 |

- **独立评分**：已有完整回答时，可在项目根环境中直接评分，无需重新运行 AutoCompressors 推理。

```bash
uv run -m benchmarks.longmemeval.score \
  --run-dir artifacts/reproductions/autocompressors/eval/AutoCompressor-Llama-2-7b-6k/longmemeval_s_cleaned
```

<a id="33-结果展示"></a>

### 3.2 结果展示

`--output-dir` 指向运行目录 `<output-root>/eval/<output-id>`。各任务评分完成后，`summary.json` 保存汇总指标，`details.jsonl` 保存逐题回答、参考答案及评分。看板可在同一运行目录中切换任务。

```bash
bash reproductions/autocompressors/scripts/dashboard.sh --output-dir artifacts/reproductions/autocompressors/eval/AutoCompressor-Llama-2-7b-6k
```

| 设置目录 | 结果 |
|---|---|
| `locomo/` | 总体与分题型 F1，以及各题型的题数 |
| `longmemeval_s_cleaned/`、`longmemeval_oracle/` | 生成回答写入 `predictions.jsonl`。评分后可查看总体与分题型准确率以及每题的裁判判定 |

LoCoMo 看板可按题型和对话筛选，并显示模型、分段长度、每段对话的正文与摘要 token 数，以及按对话平均的摘要 token 数和保留比例。保留比例为摘要 token 数除以正文 token 数。
