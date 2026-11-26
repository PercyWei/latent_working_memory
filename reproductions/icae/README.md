# 20260904_ICAE 复现

创建时间：20260904 10:13:57 Asia/Shanghai

最后修订时间：20261126 17:07:15 Asia/Shanghai

[环境准备](#一环境准备) · [原论文复现](#二原论文复现) · [通用功能](#三通用功能)

## 一、环境准备

推理默认以 BF16 加载模型权重，关闭模型级自动混合精度（autocast）。注意力使用 FlashAttention 2。

从服务器的项目根目录创建并激活独立评估环境。完整依赖见 [requirements-eval.txt](requirements-eval.txt)。

```bash
bash reproductions/setup_eval_env.sh icae
source ~/envs/icae-v2/bin/activate
```

环境位置、uv 镜像配置与激活方式见[评估环境](../README.md#评估环境)。

## 二、原论文复现

本项目采用 [In-context Autoencoder（ICAE）](https://arxiv.org/abs/2307.06945) 的官方 v2 实现。作者基于 Mistral-7B-Instruct-v0.2 发布了预训练与指令微调两个阶段的权重。源码与项目适配见 [UPSTREAM.md](UPSTREAM.md#icae-v2)。

两个阶段均冻结基座模型与 Reader，训练 Writer 的 LoRA 和记忆嵌入。

| 阶段 | 训练目标 |
|---|---|
| `pretrained` | 将正文压缩为记忆，再由 Reader 重建原文；官方代码也支持混入后文续写目标 |
| `finetuned` | 从预训练权重继续训练，使用“正文、指令或问题、答案”，让 Reader 根据记忆完成任务 |

### 2.1 模型下载

下载 [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) 基座与 tokenizer，以及 [ICAE v2](https://huggingface.co/sggetao/icae) 两个阶段的公开权重。`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
# Mistral 基座和 tokenizer
hf download mistralai/Mistral-7B-Instruct-v0.2 \
  --local-dir ~/models/Mistral-7B-Instruct-v0.2

# ICAE v2 预训练与指令微调权重
hf download sggetao/icae mistral_7b_pretrained_icae.safetensors mistral_7b_ft_icae.safetensors \
  --local-dir ~/models/reproductions/icae
```

## 三、通用功能

### 3.1 benchmark 评估

| 完整评估脚本共同参数 | 用途与默认值 |
|---|---|
| `--stage` | 必填，与 checkpoint 对应的阶段，可选 `pretrained` 或 `finetuned` |
| `--model-path` | 必填，本地基座模型目录 |
| `--checkpoint` | 必填，所选阶段的本地 checkpoint 文件（`.safetensors`） |
| `--benchmark-data` | 输入数据目录，默认路径见各任务小节 |
| `--output-root` | 输出根目录，默认 `artifacts/reproductions/icae` |
| `--output-id` | 运行 ID，默认 `<checkpoint_stem>_mem<fixed_mem_size>_cr<mean_compression_rate>_<YYYYMMDD-HHMM>`，续评估时必填 |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |
| `--resume` | 用于续评估，默认关闭 |
| `--fixed-mem-size` | 每块的记忆槽数，取正整数，默认 `128`，须与 checkpoint 匹配 |
| `--mean-compression-rate` | 每个记忆槽对应的平均输入 token 数，取正整数，默认 `4` |

- **运行方式**：在项目根目录使用 ICAE 环境运行。每张可见 GPU 加载一个模型副本。[环境准备](#一环境准备) · [数据下载](../../benchmarks/README.md)
- **历史压缩**：Writer 按 `ceil(输入 token 数 / (每块记忆槽数 × 平均压缩率))` 确定块数，再将历史尽量均分到各块。默认每块最多把 512 个输入 token 压成 128 个记忆 token。
- **输入与解码**：Reader 按 Mistral 聊天模板读入系统提示、记忆和问题，在 `[/INST]` 后贪心生成回答。指令微调阶段在记忆后插入 `[FT]`。Reader 输入与回答预算之和须在 32,768 个位置内。
- **结果与续评**：各评估设置的结果分别保存在 `<output-root>/eval/<output-id>/` 下的对应子目录。`--resume` 复用已保存的回答和评分，只补齐未完成的部分。

<a id="locomo-评估"></a>
<a id="31-locomo-评估"></a>

#### 1. LoCoMo

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |

- **评估范围**：覆盖 10 段完整对话中类别 1–4 的 1,540 道题。[LoCoMo 评估协议](../../benchmarks/locomo/README.md)
- **推理方式**：按对话分配 GPU，每段对话历史只压缩一次。同一对话下的各题复用记忆与 Reader 前缀缓存，独立作答。
- **回答长度与评分**：每题最多生成 50 个 token。按官方规则逐题计算 F1，再对所有题目取平均。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/icae/scripts/eval_locomo.sh \
  --stage finetuned \
  --model-path ~/models/Mistral-7B-Instruct-v0.2 \
  --checkpoint ~/models/reproductions/icae/mistral_7b_ft_icae.safetensors \
  --output-id mistral_7b_ft_icae_mem128_cr4 \
  --batch-size-per-gpu 2
```

<a id="32-longmemeval-评估"></a>

#### 2. LongMemEval

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/longmemeval` |
| `--split` | 支持 `longmemeval_s_cleaned`（默认）和 `longmemeval_oracle` |
| `--max-new-tokens` | 每题最多生成的 token 数，默认 `128` |

- **评估范围**：S 和 Oracle 使用相同的 500 道题。S 提供证据会话和干扰会话，Oracle 只提供证据会话。
- **推理方式**：相同的完整历史归到同一卡。Writer 并行压缩本批的新历史，Reader 从各题对应的记忆出发，结合提问日期并行作答。
- **输入长度**：按 Mistral tokenizer 统计，S 的历史长 113,994–125,468 个 token。使用默认记忆槽数、压缩率和回答预算时，S 和 Oracle 的全部题目均能放入 Reader 窗口。
- **自动评分**：生成全部回答后，脚本调用 `gpt-6-luna`，按官方分题型提示词逐题判定正误，汇总准确率。裁判密钥 `OPENAI_API_KEY` 可在项目根目录的 `.env` 中配置。[裁判配置与评分口径](../../benchmarks/longmemeval/README.md#问答评分)

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/icae/scripts/eval_longmemeval.sh \
  --split longmemeval_s_cleaned \
  --stage finetuned \
  --model-path ~/models/Mistral-7B-Instruct-v0.2 \
  --checkpoint ~/models/reproductions/icae/mistral_7b_ft_icae.safetensors \
  --output-id mistral_7b_ft_icae_mem128_cr4
```

| 独立评分代码参数 | 用途与默认值 |
|---|---|
| `--run-dir` | 必填，所选设置的结果目录 |
| `--benchmark-data` | 默认沿用生成时记录的数据目录 |
| `--resume` | 用于续评分，默认关闭 |

- **独立评分**：已有完整回答时，可在项目根环境中直接评分，无需重新运行 ICAE 推理。

```bash
uv run -m benchmarks.longmemeval.score \
  --run-dir artifacts/reproductions/icae/eval/mistral_7b_ft_icae_mem128_cr4/longmemeval_s_cleaned
```

<a id="33-结果展示"></a>

### 3.2 结果展示

`--output-dir` 指向运行目录 `<output-root>/eval/<output-id>`。各任务评分完成后，`summary.json` 保存汇总指标，`details.jsonl` 保存逐题回答、参考答案及评分。看板可在同一运行目录中切换任务。

```bash
bash reproductions/icae/scripts/dashboard.sh --output-dir artifacts/reproductions/icae/eval/mistral_7b_ft_icae_mem128_cr4
```

| 设置目录 | 结果 |
|---|---|
| `locomo/` | 总体与分题型 F1，以及各题型的题数 |
| `longmemeval_s_cleaned/`、`longmemeval_oracle/` | 生成回答写入 `predictions.jsonl`。评分后可查看总体与分题型准确率以及每题的裁判判定 |

LoCoMo 看板可按题型和对话筛选，并显示 ICAE 阶段、基座、记忆槽数、目标压缩率及按对话平均的记忆 token 数和保留比例。保留比例为记忆 token 数除以正文 token 数。
