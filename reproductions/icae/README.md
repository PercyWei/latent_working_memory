# 20260904_ICAE 复现

创建时间：20260904 10:13:57 Asia/Shanghai

最后修订时间：20261125 10:58:32 Asia/Shanghai

本目录使用 [In-context Autoencoder（ICAE）](https://arxiv.org/abs/2307.06945) 的官方 v2 实现，在 LoCoMo 上分别评估基于 Mistral-7B-Instruct-v0.2 的预训练与指令微调权重。源码与项目适配见 [UPSTREAM.md](UPSTREAM.md#icae-v2)。

两个阶段均冻结基座模型与 Reader，训练 Writer 的 LoRA 和记忆嵌入：

| 阶段 | 训练目标 |
|---|---|
| `pretrained` | 将正文压缩为记忆，再由 Reader 重建原文；官方代码也支持混入后文续写目标 |
| `finetuned` | 从预训练权重继续训练，使用“正文、指令或问题、答案”，让 Reader 根据记忆完成任务 |

## 评估数据

两个阶段共用 LoCoMo 的 10 段完整对话。每道题使用所属对话的全部历史，数据下载命令见[评估数据准备](../../benchmarks/README.md)。

| benchmark | 题型 | 题数 | 默认数据路径 |
|---|---|---:|---|
| LoCoMo | 多跳（1）、时间（2）、开放域（3）、单跳（4） | 1,540 | `data/benchmarks/locomo/locomo10.json` |

## 模型下载

下载 [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) 基座与 tokenizer，以及 [ICAE v2](https://huggingface.co/sggetao/icae) 两个阶段的公开权重。`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
# Mistral 基座和 tokenizer
hf download mistralai/Mistral-7B-Instruct-v0.2 \
  --local-dir ~/models/Mistral-7B-Instruct-v0.2

# ICAE v2 预训练与指令微调权重
hf download sggetao/icae mistral_7b_pretrained_icae.safetensors mistral_7b_ft_icae.safetensors \
  --local-dir ~/models/reproductions/icae
```

## 环境准备

从 Linux x86-64 服务器的项目根目录创建并激活独立评估环境。评估使用支持 BF16 和 FlashAttention 2 的 CUDA GPU，依赖版本见 [requirements-eval.txt](requirements-eval.txt)。

```bash
bash reproductions/setup_eval_env.sh icae
source ~/envs/icae-v2/bin/activate
```

自定义位置与激活方式见[评估环境](../README.md#评估环境)。

## LoCoMo 评估

在上述独立环境中运行。评估范围、F1 评分和续评规则见[公共 LoCoMo 说明](../../benchmarks/locomo/README.md)。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/icae/scripts/eval_locomo.sh \
  --stage pretrained \
  --model-path ~/models/Mistral-7B-Instruct-v0.2 \
  --checkpoint ~/models/reproductions/icae/mistral_7b_pretrained_icae.safetensors \
  --output-id mistral_7b_pretrained_icae_mem128_cr4_20261125 \
  --batch-size-per-gpu 2
```

| 参数 | 默认值与约定 |
|---|---|
| `--resume` | 沿用原 ID、checkpoint、数据、记忆槽数和压缩率，补齐未完成的问题 |
| `--stage` | 必填，支持 `pretrained` 和 `finetuned` |
| `--model-path` | 必填，本地基座模型目录 |
| `--checkpoint` | 必填，ICAE v2 格式的 `.safetensors` 权重文件 |
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |
| `--output-root` | 默认 `artifacts/reproductions/icae` |
| `--output-id` | 默认 `<checkpoint_name>_mem<fixed_mem_size>_cr<mean_compression_rate>_<YYYYMMDD-HHMM>` |
| `--batch-size-per-gpu` | 每卡评估批量大小，默认 1 |
| `--fixed-mem-size` | 每块记忆槽数，取正整数，默认 128 |
| `--mean-compression-rate` | 平均压缩率参数，取正整数，默认 4 |

- 只压缩完整对话正文，QA 系统提示与问题不参与压缩。沿用官方分段规则：先按 `ceil(输入 token 数 / (每块记忆槽数 × 压缩率))` 确定块数，再将正文尽量均匀分段。默认每块最多 512 个输入 token 压成 128 个记忆 token，实际压缩率受取整影响；Reader 输入与生成预算须在基座位置窗口内。
- 通过公共接口，按基座 tokenizer 的聊天模板构造 Reader 输入。以 Mistral 为例，输入结构为：`<s>[INST] → QA 系统提示 → 记忆 →（[FT]）→ 简答指令与问题 → [/INST]`。其中 `[FT]` 由 ICAE 适配在 `finetuned` 阶段插入。
- 每段对话只压缩一次，各题复用记忆和 Reader 前缀 KV，独立作答，互不继承其他问题或答案。
- 结果保存在 `<output-root>/eval/<output-id>/locomo/`。记忆槽数和压缩率写入评估配置，续评时严格核对。`summary.json` 汇总总体及各题型的 F1 和题数，`details.jsonl` 保存逐题回答与评分。额外指标为按对话平均的记忆 token 数和保留比例；保留比例为记忆 token 数除以正文 token 数。

## 查看结果

```bash
bash reproductions/icae/scripts/dashboard.sh \
  --output-dir artifacts/reproductions/icae/eval/mistral_7b_ft_icae_20261125
```

`--output-dir` 指向 `<output-root>/eval/<output-id>`。页面展示总体及各题型 F1、题数和记忆用量，并可按题型与对话筛选逐题回答、参考答案和 F1。阶段、基座名称、每块记忆槽数与目标压缩率读取评估配置。

## 备用训练入口

ICAE v2 的官方训练代码见[预训练](source/icae/code/icae_v2/pretrain.py)和[指令微调](source/icae/code/icae_v2/instruction_finetune.py)。运行前需准备训练依赖，并将脚本中的占位路径替换为实际的训练集和验证集路径。

本项目的 [Reader 模板解析](formatting.py)已检查 Mistral 与 Qwen3-Instruct 的输入格式；官方训练脚本和当前独立评估环境仍面向 Mistral，尚未提供 Qwen3 训练入口。
