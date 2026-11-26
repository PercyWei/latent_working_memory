# 20261125_SDCC 复现

创建时间：20261125 21:08:04 Asia/Shanghai

最后修订时间：20261126 17:07:15 Asia/Shanghai

[环境准备](#一环境准备) · [原论文复现](#二原论文复现) · [通用功能](#三通用功能)

## 一、环境准备

推理默认以 BF16 加载模型权重，关闭模型级自动混合精度（autocast）。编码器使用双向注意力，解码器使用因果注意力，两者均使用 FlashAttention 2。

从服务器的项目根目录创建并激活独立评估环境。依赖版本见 [requirements-eval.txt](requirements-eval.txt)。

```bash
bash reproductions/setup_eval_env.sh sdcc
source ~/envs/sdcc/bin/activate
```

环境位置、uv 镜像配置与激活方式见[评估环境](../README.md#评估环境)。

## 二、原论文复现

本项目采用 SDCC 的官方半动态压缩实现。选用的公开 checkpoint 以 Qwen3-4B-Instruct-2507 作为编码器和解码器的基座，分别加载训练好的 LoRA。编码器根据正文选择压缩倍率，对连续 token 的隐藏状态做均值池化，再将结果投影为解码器的输入嵌入。[SDCC 论文](https://arxiv.org/abs/2603.25926) · [源码与权重说明](UPSTREAM.md)

原论文在 2,048 tokens 以内的阅读理解样本上评估。作者提供了半动态 SFT 脚本和训练数据，从源码启动训练还需补齐脚本引用的 `sft_ctxcomp_train_config` 模块。[官方训练脚本](source/sdcc/SFT/sft_ctxcomp_semi_dynamic.py) · [训练数据](https://huggingface.co/datasets/yuyijiong/context_qa_sum_qwen3_synthetic)

### 2.1 模型下载

选择编码器与解码器使用同一基座的半动态 checkpoint，并准备与权重配套的基座和 tokenizer。下例使用 Qwen3-4B-Instruct-2507。使用官方同基座的 Qwen3-0.6B 权重时，对应替换基座和 checkpoint 即可。

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir ~/models/Qwen3-4B-Instruct-2507

hf download yuyijiong/qwen3-semi-dynamic-soft-context-compress \
  --include 'semi-dynamic/*-enc=Qwen3-4B-Instruct-2507-lora-dec=Qwen3-4B-Instruct-2507-lora/*' \
  --local-dir ~/models/reproductions/sdcc
```

## 三、通用功能

### 3.1 benchmark 评估

| 完整评估脚本共同参数 | 用途与默认值 |
|---|---|
| `--model-path` | 必填，本地基座模型目录 |
| `--checkpoint` | 必填，本地完整的 checkpoint 目录 |
| `--compress-ratio-scale` | 压缩倍率选择器的偏移量，默认 `0`。增大时倾向于使用更高的压缩倍率 |
| `--benchmark-data` | 输入数据目录，默认路径见各任务小节 |
| `--output-root` | 输出根目录，默认 `artifacts/reproductions/sdcc` |
| `--output-id` | 运行 ID，默认 `<checkpoint_name>_scale<compress-ratio-scale>_<YYYYMMDD-HHMM>`，续评估时必填 |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |
| `--resume` | 用于续评估，默认关闭 |

- **运行方式**：在项目根目录使用 SDCC 环境运行。每张可见 GPU 各加载一套编码器和解码器。[环境准备](#一环境准备) · [数据下载](../../benchmarks/README.md)
- **历史压缩**：编码器根据历史末尾 EOS 的隐藏状态选择压缩倍率。示例 checkpoint 支持 2×、4×、8×、16×、32×，长度为 `L` 的输入以倍率 `r` 压缩后得到 `ceil(L / r)` 个记忆 token。
- **输入与解码**：解码器使用基座聊天模板，`system` 消息包含 QA 提示，`user` 消息依次包含 `Context:`、压缩记忆和问题。解码器从 `assistant` 起始标记贪心生成回答。含 EOS 的历史须放入编码器窗口，记忆前缀、问题和回答预算须放入解码器窗口。
- **结果与续评**：各评估设置的结果分别保存在 `<output-root>/eval/<output-id>/` 下的对应子目录。`--resume` 复用已保存的回答和评分，只补齐未完成的部分。

<a id="locomo-评估"></a>
<a id="31-locomo-评估"></a>

#### 1. LoCoMo

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |

- **评估范围**：覆盖 10 段完整对话中类别 1–4 的 1,540 道题。按 Qwen3-4B-Instruct-2507 tokenizer 统计，每段对话正文为 13,479–25,580 个 token。[LoCoMo 评估协议](../../benchmarks/locomo/README.md)
- **推理方式**：按对话分配 GPU，每段对话只选择一次压缩倍率并生成一份记忆。各题复用解码器的同一份前缀 KV 缓存，独立作答。
- **回答长度与评分**：每题最多生成 50 个 token。按官方规则逐题计算 F1，再对所有题目取平均。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/sdcc/scripts/eval_locomo.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --checkpoint ~/models/reproductions/sdcc/semi-dynamic/ctxcomp-semi-dynamic-mean_pooling-doclen=1200to64-ratios=0.5_0.03125-enc=Qwen3-4B-Instruct-2507-lora-dec=Qwen3-4B-Instruct-2507-lora \
  --output-id qwen3-4b_sdcc_scale0 \
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
- **推理方式**：相同的完整历史归到同一卡。编码器并行处理本批的新历史，每份历史独立选择倍率并生成记忆。解码器根据各题的记忆、提问日期和问题并行作答。
- **自动评分**：生成全部回答后，脚本调用 `gpt-6-luna`，按官方分题型提示词逐题判定正误，汇总准确率。裁判密钥 `OPENAI_API_KEY` 可在项目根目录的 `.env` 中配置。[裁判配置与评分口径](../../benchmarks/longmemeval/README.md#问答评分)

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/sdcc/scripts/eval_longmemeval.sh \
  --split longmemeval_s_cleaned \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --checkpoint ~/models/reproductions/sdcc/semi-dynamic/ctxcomp-semi-dynamic-mean_pooling-doclen=1200to64-ratios=0.5_0.03125-enc=Qwen3-4B-Instruct-2507-lora-dec=Qwen3-4B-Instruct-2507-lora \
  --output-id qwen3-4b_sdcc_scale0 \
  --batch-size-per-gpu 1
```

| 独立评分代码参数 | 用途与默认值 |
|---|---|
| `--run-dir` | 必填，所选设置的结果目录 |
| `--benchmark-data` | 默认沿用生成时记录的数据目录 |
| `--resume` | 用于续评分，默认关闭 |

- **独立评分**：已有完整回答时，可在项目根环境中直接评分，无需重新运行 SDCC 推理。

```bash
uv run -m benchmarks.longmemeval.score \
  --run-dir artifacts/reproductions/sdcc/eval/qwen3-4b_sdcc_scale0/longmemeval_s_cleaned
```

<a id="33-结果展示"></a>

### 3.2 结果展示

`--output-dir` 指向运行目录 `<output-root>/eval/<output-id>`。各任务评分完成后，`summary.json` 保存汇总指标，`details.jsonl` 保存逐题回答、参考答案及评分。看板可在同一运行目录中切换任务。

```bash
bash reproductions/sdcc/scripts/dashboard.sh \
  --output-dir artifacts/reproductions/sdcc/eval/qwen3-4b_sdcc_scale0
```

| 设置目录 | 结果 |
|---|---|
| `locomo/` | 总体与分题型 F1，以及各题型的题数 |
| `longmemeval_s_cleaned/`、`longmemeval_oracle/` | 生成回答写入 `predictions.jsonl`。评分后可查看总体与分题型准确率以及每题的裁判判定 |

LoCoMo 看板可按题型和对话筛选，并显示各对话的压缩倍率、正文与记忆 token 数及按对话平均的保留比例。保留比例为记忆 token 数除以原始正文 token 数。
