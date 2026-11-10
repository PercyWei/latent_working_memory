# 20261108_In-Place TTT 第一阶段复现

创建时间：20261108 17:05:19 Asia/Shanghai

最后修订时间：20261110 11:34:26 Asia/Shanghai

## 当前状态与目标

官方代码、独立评估环境和本地 RULER 入口已接通。第三方 Qwen3-4B 32K TTT／baseline 均已完成 4K–256K 的代表性 GPU 生成与评分检查，完整评估尚未运行。首批评估采用完整 attention，每个 checkpoint 共 **9,100 条**：13 项任务、4K／8K／16K／32K／64K／128K／256K 七个长度、每项每长度 100 条。

实现分为 `upstream/`（未修改的官方来源）、`scripts/`（本地加载、生成、评分和记录）、`optimized/`（独立的尾部缓存压紧及 MLP 分块）。默认 `original` 使用官方模型和 OpenCompass 生成路径；`optimized` 主要降低显存，速度基本持平。模型来源与代码差异见 [UPSTREAM.md](UPSTREAM.md)，验证及优化结论统一见[复现状态记录](../20261110_ttt_reproduction_status.md)。

原文 Table 1 的 Qwen3-4B 两阶段继续预训练结果仅作参照，当前第三方权重不是该表的官方模型。RULER 为 13 项任务平均准确率（%）：

| 长度 | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|---|
| 原文 baseline | 96.6 | 94.1 | 92.1 | 88.7 | 74.3 | 74.8 | 41.7 |
| 原文 In-Place TTT | 96.1 | 95.6 | 92.7 | 89.3 | 78.7 | 77.0 | 43.9 |

第三方模型在 32K 上训练，超过 32K 属于长度外推；入口保留 checkpoint 的 RoPE 配置，不自动添加 YaRN。原文两阶段训练为 32K／约 20B tokens，再 128K／约 15B tokens 并扩展 YaRN。作者未公开对应原文权重与内部语料，详见[官方回复](https://github.com/ByteDance-Seed/In-Place-TTT/issues/1)。

## 模型与数据准备

下载第三方 [32K TTT](https://huggingface.co/zhongweixie/inplace-ttt-qwen3-4b-32k-ttt) 和 [32K baseline](https://huggingface.co/zhongweixie/inplace-ttt-qwen3-4b-32k-baseline)。两者为完整 Hugging Face checkpoint；模型卡记录使用 ProLong 训练 6000 steps。`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
hf download zhongweixie/inplace-ttt-qwen3-4b-32k-ttt \
  --local-dir ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt

hf download zhongweixie/inplace-ttt-qwen3-4b-32k-baseline \
  --local-dir ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-baseline
```

RULER 的下载、构造命令和完整规模见[评估数据准备](../20261108_ttt_data_preparation.md)。默认读取 `data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b/`，不在评估时生成数据或下载模型。

## 环境

在服务器项目根目录安装并激活独立环境；已安装时仅需激活：

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh in_place_ttt
source ~/rivermind-data/envs/ttt-in_place_ttt/bin/activate
```

依赖由 [requirements-eval.txt](requirements-eval.txt) 固定，版本及环境说明见[评估环境](../20261109_environment_setup.md)。本轮评估使用 Python 3.11、PyTorch 2.8.0 cu128、FlashAttention 2.8.3、Transformers 4.57.3、OpenCompass 0.5.5，不安装训练专用的 VeOmni。官方训练与 DCP 导出入口保留于 `upstream/`，使用时另按[官方说明](upstream/README.md)准备完整训练环境。

## RULER 评估

以下命令从仓库根目录执行，使用已激活环境的 `python`。显式选择物理 GPU 0；入口也接受单卡 1，进程内均为 `cuda:0`。

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/in_place_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --inference-mode original \
  --output-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-ttt-ruler

CUDA_VISIBLE_DEVICES=0 bash reproductions/in_place_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-baseline \
  --inference-mode original \
  --output-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-baseline-ruler
```

可选显存优化路径使用独立目录。以下命令同时启用尾部缓存压紧和 MLP 分块；分块的数值限制见下文。

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/in_place_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --inference-mode optimized --mlp-chunk-size 16384 \
  --output-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-ttt-ruler-optimized

CUDA_VISIBLE_DEVICES=0 bash reproductions/in_place_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-baseline \
  --inference-mode optimized --mlp-chunk-size 16384 \
  --output-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-baseline-ruler-optimized
```

| 参数 | 约定 |
|---|---|
| `--checkpoint` | 必需，本地完整 Hugging Face checkpoint 目录 |
| `--output-dir` | 必需，首次运行时必须尚不存在 |
| `--inference-mode` | `original`（默认）或 `optimized` |
| `--mlp-chunk-size` | 优化模式默认 `16384`；`0` 关闭 MLP 分块，仅压紧 TTT 尾部缓存；原始模式有效值为 `0` |
| `--ruler-data` | 默认 `data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b` |
| `--lengths` | 默认 `4096 8192 16384 32768 65536 131072 262144` |
| `--resume` | 使用相同设置和输出目录继续，跳过已有预测并重新汇总 |

若只需要 TTT 尾部缓存压紧、保留原 MLP 计算形状，显式使用 `--inference-mode optimized --mlp-chunk-size 0`，并指定独立输出目录。这是更保守的显存优化选项。

中断后在原命令末尾追加 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/in_place_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --inference-mode original \
  --output-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-ttt-ruler \
  --resume
```

续评核对模式、分块大小、backend、运行配置与依赖，禁止混用不同实现的输出目录。仅自动移除末行未写完的 JSON，其他记录或配置错误直接报错。

### 生成与评分

每条输入为 `input + answer_prefix`，采用 base completion、greedy、最多 64 个生成 token，保留 checkpoint EOS，不强制生成满额。输入加生成预算超过该档长度时报错，不截断。每题创建独立 cache，TTT 状态不跨样本传递；两种模式均为单进程、batch size 1。

入口严格核对权重加载结果，使用原 `extract_non_reasoning_content` 和 OpenCompass 五类原生 evaluator；长度汇总为 13 项原生分数的等权平均。运行采用 SDPA／PyTorch FlashAttention。入口默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，显式环境值原样保留，降低长输入分配器碎片风险。

### 结果与来源

- `plan.json`：参数、数据、checkpoint 配置及初始 Git commit／dirty/status、上游 revision 和实现路径。
- `runtime.json`：实际模型／cache／MLP 类、优化开关、分块大小、backend、dtype、依赖与生成配置。
- `execution.json`：每次执行的来源、退出状态、耗时、峰值显存和复用样本数；逐题 `execution_attempt` 关联对应记录。
- `predictions.jsonl`：逐题预测、答案、输入长度、原生分数及完整生成 IDs。
- `results/<length>/<task>.json`：原生任务评分；`summary.json` 仅在全部所选数据完成后生成，日志保存于 `evaluation.log`。

### 优化适用范围

两个 checkpoint 各 14 条真实输入的原始／优化生成 IDs、文本与评分一致，但 MLP 分块在部分合成输入上存在 logits 和状态差异，不能保证任意输入等价。仅尾部压紧保留原 MLP 形状。验证结论、显存与估时统一见[复现状态记录](../20261110_ttt_reproduction_status.md)。`scripts/validate_optimization.py` 保留用于重新进行原始／优化配对检查，不属于正式 benchmark 入口。
