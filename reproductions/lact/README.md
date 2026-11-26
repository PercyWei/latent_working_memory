# 20261108_LaCT 第一阶段复现

创建时间：20261108 17:03:39 Asia/Shanghai

最后修订时间：20261126 10:26:09 Asia/Shanghai

[环境准备](#一环境准备) · [原论文复现](#二原论文复现) · [通用功能（LoCoMo）](#三通用功能)

## 一、环境准备

推理默认以 BF16 加载模型权重，关闭模型级自动混合精度（autocast）。注意力使用 FlashAttention 2。三个公开的 760M checkpoint（Momentum、Muon PreNorm 和 Muon PostNorm）的快速权重配置均为 `fp32_states=false`。

从服务器项目根目录安装并激活 LaCT 独立评估环境。

```bash
bash reproductions/setup_eval_env.sh lact
source ~/envs/lact/bin/activate
```

依赖版本见[环境配置记录](../20261109_environment_setup.md)，环境位置、uv 镜像和激活方式见[评估环境](../README.md#评估环境)。

## 二、原论文复现

本目录使用 LaCT 官方语言模型实现和作者公开的 760M checkpoint，复现论文的 RULER 长上下文检索评估。

[Test-Time Training Done Right](https://arxiv.org/abs/2505.23884v1) · [源码与本地优化说明](UPSTREAM.md) · [复现状态](../20261110_ttt_reproduction_status.md)

<a id="模型下载"></a>

### 2.1 模型下载

首批评估使用 Momentum／RoPE／PreNorm，可选的两个 Muon／NoPE checkpoint 分别采用 PreNorm 和 PostNorm。三个模型均由作者按论文设置重新训练。权重来自 [YunjinZhang/lact_llm](https://huggingface.co/YunjinZhang/lact_llm/tree/main)，下载工具见 [`hf` CLI 安装说明](../README.md#模型下载约定)。

```bash
# Momentum／RoPE／PreNorm
hf download YunjinZhang/lact_llm \
  --include 'lact-momentum-rope-prenorm-nheads4-chunk2048-760m/*' \
  --local-dir ~/models/reproductions/lact

# 可选：Muon／NoPE 的 PreNorm 和 PostNorm
hf download YunjinZhang/lact_llm \
  --include 'lact-muon-*/*' \
  --local-dir ~/models/reproductions/lact
```

公开目录名中的头数与模型实际配置有差异。加载时采用 checkpoint 自带的配置，Momentum 有两个 fast-weight heads，两个 Muon 各有四个。[公开 checkpoint 权重说明](UPSTREAM.md#公开语言模型权重)

### 2.2 RULER 评估

三个模型共用 tokenizer、特殊 token 和下表的 RULER 数据。[下载和构造命令](../20261108_ttt_data_preparation.md)

| benchmark | 任务与长度 | 样本数 | 固定数据路径 |
|---|---|---:|---|
| RULER | S-NIAH-1/2/3、MK-NIAH-1、MQ-NIAH、MV-NIAH；4K／8K／16K／32K；每项任务、每个长度 500 条 | 12,000 | `data/benchmarks/ruler-v1/reproductions/lact-760m` |

在本项目的[评估环境](#一环境准备)中运行。Momentum 和 Muon 均支持 `optimized` 推理。[验证范围](../20261110_ttt_reproduction_status.md#lact)

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/lact/scripts/eval.sh \
  --checkpoint ~/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --output-id lact-momentum-rope-prenorm-nheads4-chunk2048-760m_20261116 \
  --inference-mode optimized \
  --batch-size-by-length 4096=64 8192=32 16384=32 32768=16
```

| 参数 | 默认值与约定 |
|---|---|
| `--resume` | 沿用原评估协议补齐未完成样本的评估，默认不开启 |
| `--checkpoint` | 必需，本地完整 checkpoint 目录 |
| `--output-root` | 产物父目录，默认 `artifacts/reproductions/lact` |
| `--output-id` | 默认 `<checkpoint_name>_<YYYYMMDD-HHMM>` |
| `--inference-mode` | 推理模式，支持 `original`（默认）和 `optimized` |
| `--batch-size` | 每卡样本数上限，默认 `1`；只有 `optimized` 支持大于 1 |
| `--batch-size-by-length` | 按评估长度覆盖每卡上限，未指定的长度使用 `--batch-size` |
| `--lengths` | 默认 `4096 8192 16384 32768` |

- 输入为 `input + answer_prefix`，greedy 生成最多 128 tokens，保留 EOS。输入与生成预算之和须在所选长度以内，评分沿用 RULER。
- 每批按最长输入左填充，末批使用实际样本数。`input_tokens` 记录每条样本的真实输入长度，生成结果从填充后的批次宽度开始截取。每条样本的 TTT 状态独立，只按真实 token 的 chunk 进度批量读出和提交。[验证范围与 Muon 数值限制](../20261110_ttt_reproduction_status.md)
- 新运行选择尚未使用的 `--output-id`。中断时用原 `--output-id` 加 `--resume` 续评。续评复用已保存的逐样本预测，沿用原 checkpoint、长度和推理模式，可调整 GPU 数量、顺序及批量设置。

#### 来源记录与输出

`scripts/eval.sh` 调用 `evaluation/evaluate_parallel.py` 调度各卡。评估记录位于 `<output-root>/eval/<output-id>/ruler/`，其中 `summary.json` 的 `merged_summary_path` 指向本轮完整汇总。[进度展示规则](../README.md#进度展示)

| 文件 | 内容 |
|---|---|
| `summary.json` | 所选样本全部完成后的任务／长度分数 |
| `evaluation_plan.json` | 固定 checkpoint、数据及生成／评分协议 |
| `launcher_execution.json` | 各次启动状态与执行参数 |
| `attempts/attempt-<编号>/shards/shard-<i>/progress.json` | 当前 worker 的独立进度快照，仅用于展示 |
| `attempts/attempt-<编号>/shards/shard-<i>/predictions.jsonl` | 本轮各卡实际完成的逐样本预测与评分 |
| `attempts/attempt-<编号>/merged/predictions.jsonl` | 本轮完成后按样本顺序汇总的全部预测 |

### 2.3 论文复现对照

论文 Table 8 和 Table 9 给出的 760M 模型平均准确率如下，单位为 %。论文默认采用四个 fast-weight heads，另报告了两头 Momentum 的结果，便于和公开权重对照。[Table 8](https://arxiv.org/html/2505.23884v1#A3.T8) · [Table 9](https://arxiv.org/html/2505.23884v1#A3.T9)

| 指标 | 更新方式 | 4K | 8K | 16K | 32K |
|---|---|---:|---:|---:|---:|
| S-NIAH-1/2/3 平均准确率 | Momentum（两头） | 70.9 | 35.2 | 14.9 | 7.7 |
| S-NIAH-1/2/3 平均准确率 | Momentum（四头） | 89.9 | 64.3 | 40.9 | 33.1 |
| S-NIAH-1/2/3 平均准确率 | Muon（四头） | 77.9 | 60.5 | 40.0 | 35.1 |
| MK-NIAH-1／MQ-NIAH／MV-NIAH 平均准确率 | Momentum（两头） | 40.3 | 25.1 | 14.4 | 7.7 |
| MK-NIAH-1／MQ-NIAH／MV-NIAH 平均准确率 | Momentum（四头） | 41.7 | 29.7 | 18.5 | 9.1 |
| MK-NIAH-1／MQ-NIAH／MV-NIAH 平均准确率 | Muon（四头） | 42.9 | 32.1 | 17.1 | 7.5 |

完成六项任务、四个长度的 RULER 评估后，运行以下命令生成论文对照报告。Momentum 对照两头和四头的论文结果，两个 Muon 变体均对照论文的四头 Muon。

```bash
python reproductions/lact/evaluation/compare_results.py \
  --eval-dir artifacts/reproductions/lact/eval/lact-momentum-rope-prenorm-nheads4-chunk2048-760m_20261116
```

`--eval-dir` 指向包含 `ruler/` 的运行目录 `<output-root>/eval/<output-id>`。打开该目录下的 `comparison.html` 查看图表与逐任务分数。

## 三、通用功能

<a id="locomo-评估"></a>

### 3.1 LoCoMo 评估

使用[模型下载](#21-模型下载)中的公开 checkpoint，评估 LoCoMo 的 10 段完整对话及题型 1–4 的 1,540 道问题。[数据下载](../../benchmarks/README.md) · [LoCoMo 评分与续评规则](../../benchmarks/locomo/README.md)

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/lact/scripts/eval_locomo.sh \
  --checkpoint ~/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --output-id lact-momentum-rope-prenorm-nheads4-chunk2048-760m_20261116 \
  --batch-size-per-gpu 2
```

| 参数 | 默认值与约定 |
|---|---|
| `--resume` | 沿用原 ID、checkpoint 和数据，补齐未完成的问题 |
| `--checkpoint` | 必填，本地完整 checkpoint 目录 |
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |
| `--output-root` | 默认 `artifacts/reproductions/lact` |
| `--output-id` | 默认 `<checkpoint_name>_<YYYYMMDD-HHMM>` |
| `--batch-size-per-gpu` | 每卡评估批量大小，默认 1 |

- 使用 `optimized` 增量推理。
- 公共接口按普通文本续写格式拼接 `QA 系统提示 → 完整对话 → 简答指令与问题 → Answer:`。
- 每段对话只预填一次，各题从这份正文状态独立作答，互不继承其他问题或答案。

### 3.2 结果展示

结果保存在 `<output-root>/eval/<output-id>/locomo/`。`summary.json` 汇总总体及各题型的 F1 和题数，`details.jsonl` 保存逐题回答与评分。

```bash
bash reproductions/lact/scripts/dashboard.sh \
  --output-dir artifacts/reproductions/lact/eval/lact-momentum-rope-prenorm-nheads4-chunk2048-760m_20261116
```

`--output-dir` 指向包含 `locomo/` 的运行目录。页面展示总体及各题型 F1、题数，并可按题型与对话查看逐题回答和参考答案。
