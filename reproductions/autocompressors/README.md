# 20261124_AutoCompressors 复现

创建时间：20261124 11:54:28 Asia/Shanghai

最后修订时间：20261126 10:26:09 Asia/Shanghai

[环境准备](#一环境准备) · [原论文复现](#二原论文复现) · [通用功能（LoCoMo）](#三通用功能)

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

<a id="locomo-评估"></a>

### 3.1 LoCoMo 评估

评估 LoCoMo 的 10 段完整对话。每道题使用所属对话的全部历史，数据下载命令见[评估数据准备](../../benchmarks/README.md)。

| benchmark | 题型 | 题数 | 默认数据路径 |
|---|---|---:|---|
| LoCoMo | 多跳（1）、时间（2）、开放域（3）、单跳（4） | 1,540 | `data/benchmarks/locomo/locomo10.json` |

`CUDA_VISIBLE_DEVICES` 指定使用的 GPU，按对话分配任务。评估范围、F1 评分与续评规则见[公共 LoCoMo 说明](../../benchmarks/locomo/README.md)。

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

### 3.2 结果展示

结果保存在 `<output-root>/eval/<output-id>/locomo/`。除[公共评估产物](../../benchmarks/locomo/README.md#产物规范)外，还记录各段对话的正文与记忆 token 数，以及按对话平均的记忆 token 数和保留比例。保留比例为摘要 token 数除以正文 token 数。

```bash
bash reproductions/autocompressors/scripts/dashboard.sh \
  --output-dir artifacts/reproductions/autocompressors/eval/AutoCompressor-Llama-2-7b-6k_seg2048_20261125
```

看板的 `--output-dir` 指向包含 `locomo/` 的运行目录。页面展示总体及各题型 F1、题数和记忆用量，并可按题型与对话筛选逐题回答、参考答案和 F1。模型名称与分段长度读取评估配置。
