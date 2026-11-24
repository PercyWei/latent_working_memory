# 20261124_AutoCompressors 复现

创建时间：20261124 11:54:28 Asia/Shanghai

最后修订时间：20261124 11:59:50 Asia/Shanghai

本目录用 Princeton 发布的 `AutoCompressor-Llama-2-7b-6k` 在 LoCoMo 上做开放问答。作者使用 15B 个 RedPajama tokens 继续训练 Llama-2-7B，在 6,144 tokens 的序列上分四步训练摘要向量。当前使用该语言建模 checkpoint，评估阶段没有 QA 微调。[官方模型卡](https://huggingface.co/princeton-nlp/AutoCompressor-Llama-2-7b-6k)

## 环境与权重

在项目根目录执行。评估环境使用 Linux x86-64、Python 3.11、PyTorch 2.1.0 CUDA 11.8 和 BF16 FlashAttention 2.3.5，适用于 A100/A800。安装脚本将官方源码检出到 `source/AutoCompressors/`，创建方法独立的 `.venv/`。

```bash
bash reproductions/autocompressors/scripts/setup.sh

hf download \
  princeton-nlp/AutoCompressor-Llama-2-7b-6k \
  --revision e32c98b484e24e66cead4218a7fee1e0e5dd616f \
  --local-dir ~/rivermind-data/models/reproductions/autocompressors/AutoCompressor-Llama-2-7b-6k
```

## LoCoMo 评估

准备 `data/benchmarks/locomo/locomo10.json` 后运行。每个进程使用一张 GPU，每段对话写入一次，回答端只预填一次共享前缀，再逐题贪心生成最多 50 个 token。结果采用项目共用的 LoCoMo 官方四类逐题 F1。

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/autocompressors/scripts/eval_locomo.sh \
  --model-path ~/rivermind-data/models/reproductions/autocompressors/AutoCompressor-Llama-2-7b-6k \
  --benchmark-data data/benchmarks/locomo \
  --segment-length 2048 \
  --output-dir artifacts/reproductions/autocompressors/eval/llama-2-7b_segment2048/locomo
```

| 参数 | 含义 |
|---|---|
| `--model-path` | 完整本地 checkpoint 目录，包含权重、配置和 tokenizer |
| `--benchmark-data` | LoCoMo 数据目录，默认 `data/benchmarks/locomo` |
| `--segment-length` | 每段原文最多多少个 token，默认 2,048，与作者公布的困惑度评估设置一致 |
| `--output-dir` | 独立运行目录。保存 `evaluation_config.json`、逐题 `details.jsonl` 和 `summary.json` |
| `--resume` | 在相同数据、本地 checkpoint 和分段设置下继续未完成的问题 |

**记忆预算。** 长度为 `N` 的正文按 token 顺序切分，末段保留实际长度。每段连同之前累积的软记忆送入官方模型，新增 50 个摘要向量。全部对话最终占 `50 × ceil(N / segment_length)` 个槽，回答时保留的原文为 0。段长为 2,048 时，一个完整段的倍率为 40.96×；实际整篇倍率计入末段。LoCoMo 的长对话可跨越训练时的四段。每次写入的旧记忆、原文与新槽，以及每次读出的完整前缀和回答预算，都须能放入公开配置的 4,096 个位置。过短的段会使累积记忆超窗，评估在该处报告长度。

**问答输入。** Llama-2 基座使用普通文本。回答时依次提供 BOS 和项目的 QA system 提示、整段对话的累积摘要、LoCoMo 简答指令与问题、`Answer:`。只有对话正文参与写入。修改段长会改变压缩预算，应在新的输出目录运行。目录职责和上游版本见 [UPSTREAM.md](UPSTREAM.md)。
