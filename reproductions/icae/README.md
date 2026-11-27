# 20260904_ICAE 复现

创建时间：20260904 10:13:57 Asia/Shanghai

最后修订时间：20261127 14:50:15 Asia/Shanghai

[环境准备](#一环境准备) · [原论文复现](#二原论文复现) · [通用功能](#三通用功能)

## 一、环境准备

推理默认以 BF16 加载模型权重，关闭模型级自动混合精度（autocast）。注意力使用 FlashAttention 2。

在 Linux x86-64 GPU 服务器的项目根目录运行以下命令，创建或更新独立的 ICAE v2 环境。Qwen3 自行训练和 Mistral 公开权重评估共用该环境。

- **GPU 依赖**：Python 3.11、PyTorch `2.8.0+cu128`、FlashAttention `2.8.3`。安装脚本使用与 Torch 2.8、CUDA 12 和 C++11 ABI（`TRUE`）匹配的预编译 wheel。
- **训练依赖**：Transformers `4.57.3`、PEFT `0.18.0`、Accelerate `1.10.1` 和 Datasets `3.6.0`。[完整依赖](requirements-eval.txt)

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

### 2.2 自行训练 ICAE v2

- **基座与权重**：下例使用 Qwen3-4B-Instruct-2507 的基座与 tokenizer。指令微调加载第一步在同一 Qwen 基座上训练的 ICAE 权重，LoCoMo 评估加载第二步的微调权重。训练与评估共用 LoRA rank 512。
- **训练数据**：每个阶段分别准备训练集和验证集，以下示例使用 JSONL，每行一个样本。预训练示例为 `{"text":"The Nile flows into the Mediterranean Sea."}`；指令微调示例为 `{"input":"The Nile flows into the Mediterranean Sea.","prompt":"Where does the Nile flow?","answer":"The Mediterranean Sea."}`。
- **并行与批量**：每张可见 GPU 启动一个进程，每卡每个微批处理一篇正文。全局有效批量等于 GPU 数乘以 `--gradient-accumulation-steps`。两个阶段均在训练结束时计算验证集 loss。

| 训练脚本参数 | 用途与默认值 |
|---|---|
| `--model-path` | 必填，本地基座模型与 tokenizer 目录 |
| `--train-file`、`--validation-file` | 必填，当前阶段的训练集、验证集文件路径，支持 JSON 和 JSONL |
| `--output-dir` | 必填，当前阶段的训练输出目录 |
| `--checkpoint` | 指令微调必填，预训练 ICAE 的 `.safetensors` 文件 |
| `--fixed-mem-size`、`--mean-compression-rate` | 每块记忆槽数和目标压缩率，默认 `128`、`4`。槽数为 2 的幂；微调及评估沿用预训练设置 |
| `--model-max-length` | Writer 的正文 token 上限，默认 `5120`。预训练随机截取超长训练文本；预训练验证和指令微调使用正文开头。Reader 输入仍须放入基座窗口 |
| `--lm-ratio` | 仅预训练，采样后文续写目标的概率，默认 `0`（重建原文）；验证始终使用重建目标 |
| `--num-train-epochs`、`--max-steps` | 默认训练 `1` 轮；`--max-steps` 默认为 `-1`，设为正数时按指定优化步数训练 |
| `--learning-rate`、`--gradient-accumulation-steps` | 学习率和梯度累积步数，默认 `5e-5`、`1` |
| `--save-steps`、`--seed` | 两次训练断点间的优化步数和随机种子，默认 `500`、`42` |

**预训练**

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir ~/models/Qwen3-4B-Instruct-2507

CUDA_VISIBLE_DEVICES=0,1 bash reproductions/icae/scripts/pretrain.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --train-file data/reproductions/icae/pretrain/train.jsonl \
  --validation-file data/reproductions/icae/pretrain/dev.jsonl \
  --output-dir artifacts/reproductions/icae/train/qwen3_4b_instruct_2507_pretrained_icae_mem128_cr4 \
  --gradient-accumulation-steps 4
```

**指令微调**

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/icae/scripts/instruction_finetune.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --checkpoint artifacts/reproductions/icae/train/qwen3_4b_instruct_2507_pretrained_icae_mem128_cr4/model.safetensors \
  --train-file data/reproductions/icae/finetune/train.jsonl \
  --validation-file data/reproductions/icae/finetune/dev.jsonl \
  --output-dir artifacts/reproductions/icae/train/qwen3_4b_instruct_2507_ft_icae_mem128_cr4 \
  --gradient-accumulation-steps 4
```

- **训练产物**：最终权重在 `<output-dir>/model.safetensors`，包含 Writer 的 LoRA 和记忆及控制标记嵌入。训练断点位于 `checkpoint-<step>/`。中断后用相同参数和输出目录重新启动会从最新断点恢复优化器和步数；新实验或新阶段请使用各自的输出目录。
- **接入评估**：将最终文件或中间断点中的 `model.safetensors` 直接传给评估脚本的 `--checkpoint`。预训练权重选 `--stage pretrained`，指令微调权重选 `--stage finetuned`，两者均使用训练时的基座和压缩设置。例如：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/icae/scripts/eval_locomo.sh \
  --stage finetuned \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --checkpoint artifacts/reproductions/icae/train/qwen3_4b_instruct_2507_ft_icae_mem128_cr4/model.safetensors
```

## 三、通用功能

### 3.1 benchmark 评估

| 完整评估脚本共同参数 | 用途与默认值 |
|---|---|
| `--resume` | 默认关闭；在原评估命令后添加即可从同一结果目录续评 |
| `--stage` | 必填，与 checkpoint 对应的阶段，可选 `pretrained` 或 `finetuned` |
| `--model-path` | 必填，本地基座模型目录 |
| `--checkpoint` | 必填，所选阶段的本地 checkpoint 文件（`.safetensors`） |
| `--benchmark-data` | 输入数据目录，默认路径见各任务小节 |
| `--output-root` | 输出根目录，默认 `artifacts/reproductions/icae` |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |
| `--fixed-mem-size` | 每块的记忆槽数，取正整数，默认 `128`，须与 checkpoint 匹配 |
| `--mean-compression-rate` | 每个记忆槽对应的平均输入 token 数，取正整数，默认 `4` |

- **运行方式**：在项目根目录使用 ICAE 环境运行。每张可见 GPU 加载一个模型副本。[环境准备](#一环境准备) · [数据下载](../../benchmarks/README.md)
- **历史压缩**：Writer 按 `ceil(输入 token 数 / (每块记忆槽数 × 平均压缩率))` 确定块数，再将历史尽量均分到各块。默认每块最多把 512 个输入 token 压成 128 个记忆 token。
- **输入与解码**：Reader 按基座 tokenizer 的聊天模板读入系统提示、记忆和问题，从模板的回答起点贪心生成。指令微调阶段在记忆后插入 `[FT]`。Reader 输入与回答预算之和须在基座的位置窗口内。
- **产物保存**：评估结果按 checkpoint 和评估设置归档，写入 `<output_root>/eval/<checkpoint_name>/mem<fixed_mem_size>_cr<mean_compression_rate>_<task>/`。`<checkpoint_name>` 取 `--checkpoint` 指向的文件名，去掉 `.safetensors` 后缀，`<task>` 标识 benchmark，具体路径见各任务小节。

<a id="locomo-评估"></a>
<a id="31-locomo-评估"></a>

#### 1. LoCoMo

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |

- **评估范围**：覆盖 10 段完整对话中类别 1–4 的 1,540 道题。[LoCoMo 评估协议](../../benchmarks/locomo/README.md)
- **推理方式**：按对话分配 GPU，每段对话历史只压缩一次。同一对话下的各题复用记忆与 Reader 前缀缓存，独立作答。
- **回答长度与评分**：每题最多生成 50 个 token。按官方规则逐题计算 F1，再对所有题目取平均。
- **产物保存**：评估结果写入 `<output_root>/eval/<checkpoint_name>/mem<fixed_mem_size>_cr<mean_compression_rate>_locomo/`。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/icae/scripts/eval_locomo.sh \
  --stage finetuned \
  --model-path ~/models/Mistral-7B-Instruct-v0.2 \
  --checkpoint ~/models/reproductions/icae/mistral_7b_ft_icae.safetensors \
  --batch-size-per-gpu 4
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
- **产物保存**：评估结果写入 `<output_root>/eval/<checkpoint_name>/mem<fixed_mem_size>_cr<mean_compression_rate>_<split>/`。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/icae/scripts/eval_longmemeval.sh \
  --split longmemeval_s_cleaned \
  --stage finetuned \
  --model-path ~/models/Mistral-7B-Instruct-v0.2 \
  --checkpoint ~/models/reproductions/icae/mistral_7b_ft_icae.safetensors \
  --batch-size-per-gpu 4
```

| 独立评分代码参数 | 用途与默认值 |
|---|---|
| `--run-dir` | 必填，所选设置的结果目录 |
| `--benchmark-data` | 默认沿用生成时记录的数据目录 |
| `--resume` | 用于续评分，默认关闭 |

- **独立评分**：已有完整回答时，可在项目根环境中直接评分，无需重新运行 ICAE 推理。

```bash
uv run -m benchmarks.longmemeval.score \
  --run-dir artifacts/reproductions/icae/eval/mistral_7b_ft_icae/mem128_cr4_longmemeval_s_cleaned
```

#### 3. QASPER

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/qasper` |
| `--split` | 支持 `dev`（默认）和 `test` |
| `--max-new-tokens` | 每题最多生成的 token 数，默认 `128` |

- **评估范围**：dev 包含 281 篇论文、1,005 道题，test 包含 416 篇论文、1,451 道题。[QASPER 评估协议](../../benchmarks/qasper/README.md)
- **推理方式**：按论文分配 GPU。每篇论文只压缩一次，各题复用记忆和 Reader 前缀缓存，独立作答。
- **评分口径**：在本地逐题计算官方 Answer F1。每题取各参考答案中的最高分，再对所有题目取平均。
- **产物保存**：评估结果写入 `<output_root>/eval/<checkpoint_name>/mem<fixed_mem_size>_cr<mean_compression_rate>_qasper_<split>/`。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/icae/scripts/eval_qasper.sh \
  --split dev \
  --stage finetuned \
  --model-path ~/models/Mistral-7B-Instruct-v0.2 \
  --checkpoint ~/models/reproductions/icae/mistral_7b_ft_icae.safetensors \
  --batch-size-per-gpu 4
```

<a id="33-结果展示"></a>

### 3.2 结果展示

将 `--output-dir` 指向 `<output_root>/eval/<checkpoint_name>/`。看板默认一次展示该 checkpoint 下所有已评分的设置，按 benchmark 及其设置或划分分组，在同一任务内比较不同压缩参数。各设置目录中的 `summary.json` 保存汇总指标，`details.jsonl` 保存逐题回答、参考答案及评分。

```bash
bash reproductions/icae/scripts/dashboard.sh --output-dir artifacts/reproductions/icae/eval/mistral_7b_ft_icae
```

LoCoMo 记忆表列出各设置的 ICAE 阶段、基座、每块记忆槽数和目标压缩率，并按对话等权汇总记忆 token 数和保留比例。展开明细可查看每段对话的正文与记忆 token 数及保留比例。

页面底部可对照同一道题在各压缩设置下的回答和评分。先选择任务，再选择问题。LoCoMo 可按题型和对话筛选，LongMemEval 可按题型筛选，QASPER 可按论文筛选。
