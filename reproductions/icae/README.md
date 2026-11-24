# 20260904_ICAE 复现

创建时间：20260904 10:13:57 Asia/Shanghai

最后修订时间：20261124 14:58:37 Asia/Shanghai

本目录使用 [In-context Autoencoder（ICAE）](https://arxiv.org/abs/2307.06945) 的官方 v2 实现，在 LoCoMo 上分别评估基于 Mistral-7B-Instruct-v0.2 的预训练与指令微调权重。源码与项目适配见 [UPSTREAM.md](UPSTREAM.md#icae-v2)。

## 评估数据

两个阶段共用 LoCoMo 的 10 段完整对话。每道题使用所属对话的全部历史，数据下载命令见[评估数据准备](../../src/latent_working_memory/benchmarks/README.md)。

| benchmark | 题型 | 题数 | 默认数据路径 |
|---|---|---:|---|
| LoCoMo | 多跳（1）、时间（2）、开放域（3）、单跳（4） | 1,540 | `data/benchmarks/locomo/locomo10.json` |

## 模型下载

从 [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) 和 [sggetao/icae](https://huggingface.co/sggetao/icae) 的 `main` 分支下载基座、tokenizer 和两份 ICAE v2 权重。`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
# Mistral 基座和 tokenizer
hf download mistralai/Mistral-7B-Instruct-v0.2 \
  --include 'model*.safetensors' model.safetensors.index.json config.json generation_config.json \
    'tokenizer*' special_tokens_map.json \
  --local-dir ~/rivermind-data/models/Mistral-7B-Instruct-v0.2

# ICAE v2 预训练与指令微调权重
hf download sggetao/icae mistral_7b_pretrained_icae.safetensors mistral_7b_ft_icae.safetensors \
  --local-dir ~/rivermind-data/models/reproductions/icae
```

## 环境与运行命令

从 Linux x86-64 服务器的项目根目录创建并激活独立评估环境。评估使用单张支持 BF16 和 FlashAttention 2 的 CUDA GPU，依赖版本见 [requirements-eval.txt](requirements-eval.txt)。

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh icae
source ~/rivermind-data/envs/icae-v2/bin/activate
```

两次评估共用基座，各自写入独立目录。

```bash
# 预训练权重
CUDA_VISIBLE_DEVICES=0 bash reproductions/icae/scripts/eval_locomo.sh \
  --stage pretrained \
  --model-path ~/rivermind-data/models/Mistral-7B-Instruct-v0.2 \
  --checkpoint ~/rivermind-data/models/reproductions/icae/mistral_7b_pretrained_icae.safetensors \
  --output-root artifacts/reproductions/icae \
  --output-id mistral-7b_pretrained-4x

# 指令微调权重
CUDA_VISIBLE_DEVICES=0 bash reproductions/icae/scripts/eval_locomo.sh \
  --stage finetuned \
  --model-path ~/rivermind-data/models/Mistral-7B-Instruct-v0.2 \
  --checkpoint ~/rivermind-data/models/reproductions/icae/mistral_7b_ft_icae.safetensors \
  --output-root artifacts/reproductions/icae \
  --output-id mistral-7b_finetuned-4x
```

| 参数 | 默认值与约定 |
|---|---|
| `--resume` | 默认关闭。指定原 `--output-id` 并沿用原权重和数据，仅补齐未完成的题目 |
| `--stage` | 必填。预训练选 `pretrained`，指令微调选 `finetuned` |
| `--model-path` | 必填。本地 Mistral-7B-Instruct-v0.2 基座和 tokenizer 目录 |
| `--checkpoint` | 必填。与所选阶段对应的官方 `.safetensors` 文件，保留下载时的文件名 |
| `--benchmark-data` | 包含 `locomo10.json` 的目录，默认 `data/benchmarks/locomo` |
| `--output-root` | 产物父目录，默认 `artifacts/reproductions/icae` |
| `--output-id` | 单个目录名，首次运行须使用新名称。省略时按权重文件名（去扩展名）和上海时间生成 `<checkpoint>_<YYYYMMDD-HHMM>` |

- 压缩器只读取对话正文，将完整历史均分为若干段，每段最多 512 个输入 token，压成 128 个记忆槽。超过官方示例 5,120 tokens 的对话也按此规则处理。
- 同一对话的各题复用记忆槽和回答端的前缀 KV 缓存。回答端依次读取 QA 系统提示、记忆槽和问题。指令微调阶段在记忆后加入官方 `[FT]` 标记，预训练阶段直接用冻结基座回答。
- 贪心解码最多生成 50 个 token。按 LoCoMo 官方规则计算逐题 F1，再对题目取平均，分别报告总体和各题型分数。

## 查看结果

两次评估完成后，从项目根目录运行以下命令，在终端对照两个阶段的总体和分类 F1（百分制）。

```bash
jq '{stage: .settings.stage, questions: .samples, F1: (.score * 100),
     by_category: [.groups[] | {category: .category_name, F1: (.score * 100)}]}' \
  artifacts/reproductions/icae/eval/mistral-7b_{pretrained,finetuned}-4x/locomo/summary.json
```

## 来源记录与输出

[评估入口](evaluation/eval_locomo.py) 将以下文件写入 `<output-root>/eval/<output-id>/locomo/`。

| 文件 | 内容 |
|---|---|
| `evaluation_config.json` | 权重与数据标识、输入与评分协议、启动时的 Git commit 和工作树状态 |
| `details.jsonl` | 逐题追加的问题、参考答案、模型回答、F1 和该对话的记忆用量 |
| `summary.json` | 全部题目完成后的总体与分类 F1、题数、各对话的记忆用量及平均保留比例 |

## 备用训练入口

ICAE v2 的官方训练代码见[预训练](source/icae/code/icae_v2/pretrain.py)和[指令微调](source/icae/code/icae_v2/instruction_finetune.py)。运行前需准备训练依赖，并将脚本中的占位路径替换为实际的训练集和验证集路径。
