# 20261124_AutoCompressors 复现

创建时间：20261124 11:54:28 Asia/Shanghai

最后修订时间：20261125 11:45:30 Asia/Shanghai

本目录使用 AutoCompressors 的官方实现和 Princeton 发布的 [AutoCompressor-Llama-2-7b-6k](https://huggingface.co/princeton-nlp/AutoCompressor-Llama-2-7b-6k) 权重，在 LoCoMo 上进行开放问答评估。该权重基于 Llama-2-7B 进行语言建模训练，未做 QA 微调。源码与项目适配见 [UPSTREAM.md](UPSTREAM.md)。

## 评估数据

评估 LoCoMo 的 10 段完整对话。每道题使用所属对话的全部历史，数据下载命令见[评估数据准备](../../benchmarks/README.md)。

| benchmark | 题型 | 题数 | 默认数据路径 |
|---|---|---:|---|
| LoCoMo | 多跳（1）、时间（2）、开放域（3）、单跳（4） | 1,540 | `data/benchmarks/locomo/locomo10.json` |

## 模型下载

下载 [AutoCompressor-Llama-2-7b-6k](https://huggingface.co/princeton-nlp/AutoCompressor-Llama-2-7b-6k) 的完整权重与 tokenizer。`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
hf download princeton-nlp/AutoCompressor-Llama-2-7b-6k \
  --local-dir ~/models/reproductions/autocompressors/AutoCompressor-Llama-2-7b-6k
```

## 环境准备

从 Linux x86-64 服务器的项目根目录创建并激活独立评估环境。评估使用支持 BF16 和 FlashAttention 2 的 CUDA GPU，依赖版本见 [requirements-eval.txt](requirements-eval.txt)。固定版本的官方源码随项目保存在 [source/AutoCompressors](source/AutoCompressors/)，安装器仅负责创建环境和安装依赖。

```bash
bash reproductions/setup_eval_env.sh autocompressors
source ~/envs/autocompressors/bin/activate
```

自定义位置与激活方式见[评估环境](../README.md#评估环境)。

## LoCoMo 评估

在上述独立环境中运行。`CUDA_VISIBLE_DEVICES` 指定使用的 GPU，按对话分配任务。评估范围、F1 评分与续评规则见[公共 LoCoMo 说明](../../benchmarks/locomo/README.md)。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/autocompressors/scripts/eval_locomo.sh \
  --checkpoint ~/models/reproductions/autocompressors/AutoCompressor-Llama-2-7b-6k \
  --output-id AutoCompressor-Llama-2-7b-6k_seg2048_20261125 \
  --batch-size-per-gpu 2
```

| 参数 | 默认值与约定 |
|---|---|
| `--resume` | 指定原 ID、checkpoint、数据和分段设置，补齐未完成的问题 |
| `--checkpoint` | 必填，本地完整 checkpoint 目录，包含权重、配置和 tokenizer |
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |
| `--output-root` | 默认 `artifacts/reproductions/autocompressors` |
| `--output-id` | 默认 `<checkpoint_name>_seg<segment_length>_<YYYYMMDD-HHMM>` |
| `--batch-size-per-gpu` | 每卡评估批量大小，默认 1 |
| `--segment-length` | 每段正文的 token 数上限，默认 2,048 |

- 只压缩完整对话正文，QA 系统提示与问题不参与压缩。默认每 2,048 个正文 token 生成 50 个摘要向量，完整段约为 41× 压缩；末段不足该长度时仍生成 50 个摘要向量。
- 通过公共接口直接拼接 Reader 输入，不套用聊天模板：`BOS → QA 系统提示 → 记忆 → 简答指令与问题 → Answer:`。
- 每段对话只压缩一次，各题复用摘要和 Reader 前缀 KV。同一对话内按问题长度组批，每道题使用独立缓存分支作答。
- 摘要随段数累积，写入与回答均须满足 4,096 位置窗口，回答时还需预留生成预算；超窗时报错。
- 结果保存在 `<output-root>/eval/<output-id>/locomo/`，标准产物见公共说明。额外记录各段对话的正文与记忆 token 数，以及按对话平均的记忆 token 数和保留比例；保留比例为摘要 token 数除以正文 token 数。

## 查看结果

```bash
bash reproductions/autocompressors/scripts/dashboard.sh \
  --output-dir artifacts/reproductions/autocompressors/eval/llama-2-7b_segment2048
```

看板的 `--output-dir` 指向包含 `locomo/` 的运行目录。页面展示总体及各题型 F1、题数和记忆用量，并可按题型与对话筛选逐题回答、参考答案和 F1。模型名称与分段长度读取评估配置。
