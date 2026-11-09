# 20261108_In-Place TTT 第一阶段复现

创建时间：20261108 17:05:19 Asia/Shanghai

最后修订时间：20261109 11:45:56 Asia/Shanghai

## 当前状态

已迁入官方实现与训练、导出、推理及 RULER 配置，保留原 VeOmni／OpenCompass 工作流，尚未运行 GPU 训练或论文指标评估。来源和已确认差异见 [UPSTREAM.md](UPSTREAM.md)。本次 GPU 评估使用 `zcy-memory` 当前获准的物理 GPU 0；入口允许显式选择单卡 0 或 1。

`upstream/` 保持官方代码原样；本地 `eval.sh` 读取预构造 RULER JSONL，顺序调用原 OpenCompass 模型 wrapper、预测后处理和评分器。未引入统一训练框架或新的记忆目标。

## 复现目标

当前先评估第三方发布的 Qwen3-4B 32K TTT checkpoint 及其无 TTT 对照。论文 Table 1 的 Qwen3-4B-Base 两阶段继续预训练结果作为原文参照；第三方权重不属于该表对应的官方模型。RULER 为平均准确率（%），不能与单个 NIAH 子任务分数混用。

| 长度 | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|---|
| 原文 baseline | 96.6 | 94.1 | 92.1 | 88.7 | 74.3 | 74.8 | 41.7 |
| 原文 In-Place TTT | 96.1 | 95.6 | 92.7 | 89.3 | 78.7 | 77.0 | 43.9 |
| 本次结果 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |

原文继续预训练使用 AdamW，学习率 `5e-6`、weight decay `0.1`，先在 32,768 长度训练约 20B tokens，再在 131,072 长度训练约 15B tokens，并进行 YaRN 扩展；Conv kernel size 为 5。作者在[官方 issue 回复](https://github.com/ByteDance-Seed/In-Place-TTT/issues/1)中说明，内部数据政策限制了原训练语料和论文模型权重的发布。

当前仓库的 Qwen3-8B 推荐配置仅用于接通官方工作流，不能替代上述 4B 实验。500M／1.5B 从头训练目标分别需要 20B／60B tokens，作者语料同样未公开，本阶段没有用不同语料生成替代结果。

本轮评估数据的任务、规模、共用范围及完整下载与构造命令见 [评估数据准备](../20261108_ttt_data_preparation.md)。

## 模型下载

首批下载第三方 `zhongweixie` 发布的 [32K TTT](https://huggingface.co/zhongweixie/inplace-ttt-qwen3-4b-32k-ttt) 和 [32K baseline](https://huggingface.co/zhongweixie/inplace-ttt-qwen3-4b-32k-baseline)。两个仓库均包含完整权重、配置和 tokenizer；模型卡记录两者都使用 ProLong 训练 6000 steps。下载前确保已有 `hf` CLI，安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
hf download zhongweixie/inplace-ttt-qwen3-4b-32k-ttt \
  --local-dir ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt

hf download zhongweixie/inplace-ttt-qwen3-4b-32k-baseline \
  --local-dir ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-baseline
```
## 环境

本轮 checkpoint 评估使用 [requirements-eval.txt](requirements-eval.txt) 固定依赖。在服务器项目根目录安装并激活：

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh in_place_ttt
source ~/rivermind-data/envs/ttt-in_place_ttt/bin/activate
```

版本选择与安装记录见 [评估环境](../20261109_environment_setup.md)。以下上游完整环境说明用于训练依赖参考，不是本轮评估环境的安装步骤。


原始安装说明见 [upstream/README.md](upstream/README.md)。需要 Linux NVIDIA CUDA 环境、Python 3.11、PyTorch 2.8.0（CUDA 12.8）、FlashAttention 2.8.3、Transformers 4.57.3，以及指定 revision 的 VeOmni。使用方法自己的环境保留上游依赖；不要对本项目根环境执行上游 `pip uninstall` 指令。

上游目录的 `pyproject.toml` 是 VeOmni 包定义，但不包含 VeOmni 源码；训练时应按官方说明安装固定 revision 的 VeOmni，而不是把当前目录当作完整 VeOmni 包安装。本轮独立评估环境采用 OpenCompass 0.5.5，不安装 VeOmni。

## 官方训练与导出入口

以下命令从仓库根目录开始。`model_path` 和 `data_path` 必须替换为实际资源；数据应满足上游 `plaintext`、`iterable`、`content_split` 契约。该命令只验证推荐配置的一步运行，不用于报告论文指标。

```bash
project_root="$(git rev-parse --show-toplevel)"
cd reproductions/in_place_ttt/upstream
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export NNODES=1
export PYTHONPATH="$PWD"

bash train.sh tasks/train_torch.py configs/pretrain/qwen3_longct.yaml \
  --model.model_path /absolute/path/to/Qwen3-8B-Base \
  --data.train_path /absolute/path/to/processed_training_data \
  --train.output_dir "$project_root/artifacts/reproductions/in_place_ttt/train/upstream-step1" \
  --train.max_steps 1 \
  --train.use_wandb false
```

训练前必须激活上述方法环境；`train.sh` 由可见 GPU 数确定进程数。原入口还会在当前工作目录生成 `log.txt`，该文件属于运行产物，不纳入源码。

训练完成后使用官方转换入口，将 DCP checkpoint 导出为 Hugging Face 目录：

```bash
python scripts/merge_dcp_to_hf.py \
  --load-dir /absolute/path/to/dcp_checkpoint \
  --save-dir /absolute/path/to/exported_hf_checkpoint \
  --model-assets-dir /absolute/path/to/base_model
```

实际训练或导出前，需把输入数据、模型、完整训练预算及输出路径保存到相应运行计划。

## RULER 评估入口

先激活包含上述依赖的 In-Place TTT 方法环境，再从仓库根目录运行 `eval.sh`。它使用当前环境的 `python`，默认物理 GPU 0，允许显式选择单卡 0 或 1，拒绝其他卡号与多卡列表。逻辑设备始终为 `cuda:0`。

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/in_place_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --output-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-ttt-ruler

CUDA_VISIBLE_DEVICES=0 bash reproductions/in_place_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-baseline \
  --output-dir artifacts/reproductions/in_place_ttt/eval/qwen3-4b-32k-baseline-ruler
```

| 参数 | 约定 |
|---|---|
| `--checkpoint` | 必需，本地 Hugging Face checkpoint 目录 |
| `--output-dir` | 必需，新建的本次运行输出目录 |
| `--ruler-data` | 默认 `data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b`；可指定实际构造目录 |
| `--lengths` | 整数列表，默认 `4096 8192 16384 32768 65536 131072 262144` |

读取 `<ruler-data>/<length>/<task>/test.jsonl`，每个长度包含完整 13 项任务，每项恰好 100 条。默认共 9,100 条，TTT／baseline 读取同一批数据。入口不会现场生成样本、读取原始 RULER 素材或下载模型。

模型输入严格为 `input + answer_prefix`，通过原 `HuggingFaceBaseModel` 的 base completion 路径编码，不应用 chat／thinking 模板。每条生成上限为 64，采用 greedy 生成；预测先经过原配置中的 `extract_non_reasoning_content`，再调用 OpenCompass 原生 NIAH／VT／CWE／FWE／QA evaluator。按各任务完整预测列表评分，长度汇总为 13 项原生分数的等权平均，逐题分数不再次平均替代原评分。

运行先导入官方 `inference_model` 注册 TTT 模型，并核对实际模型类来自该模块；TTT 与非 TTT 行为由 checkpoint 自身配置决定。单个进程、batch size 1、逐题调用原 `generate`；上游每次调用创建新的 cache，TTT 状态不在样本间传递。

运行 `max_seq_len` 取本次最大长度，支持 256K。每条输入在生成前按真实 tokenizer 完整编码，要求 `input_tokens + 64 <= length`；超限直接报错，避免原 wrapper 的截断路径丢弃内容。保持 checkpoint 原有 `max_position_embeddings` 与 `rope_scaling`，不自动添加 YaRN；第三方 32K checkpoint 的长长度结果属于外推评估。

产物：`plan.json`（实际参数）、`runtime.json`（模型类、位置设置、依赖版本）、`predictions.jsonl`（逐题原预测、后处理预测、答案、输入长度和原生分数）、`results/<length>/<task>.json`（原生任务评分）、`summary.json`（全量完成后的长度汇总）、`evaluation.log` 和 `execution.json`。日志与逐题结果增量写入，`summary.json` 仅在全部所选数据完成后生成。

这些入口尚未通过 GPU 评估，不能据此声称已完成第三方权重或论文指标复现。

## 已执行检查

本地环境为 macOS arm64、PyTorch 2.6.0，无 CUDA；以下检查不代表论文效果复现。

| 检查 | 实际结果 |
|---|---|
| 官方源码迁入 | 40 个文件与固定 revision 对应文件一致，全部可被 Git 正常跟踪 |
| 上游 Python 语法解析 | 26 个文件通过 |
| 上游 `train.sh`、`eval.sh` shell 语法 | `bash -n` 通过 |
| 本地评估入口 | Ruff、Python／shell 语法通过；服务器独立环境依赖检查及完整入口导入通过，尚未加载 checkpoint 或运行 GPU |
| 直接导入训练模型 | 根环境缺少 `einops`，导入失败；未更改上游代码绕过依赖 |
| Qwen3MLP 训练／推理计算核 CPU 诊断 | 直接编译上游原始 AST 类，使用真实 PyTorch／einops／opt_einsum；chunk=4，长度 4、8、10、12，输出最大误差 `5.96e-8`，训练反传梯度有限 |

CPU 诊断仅覆盖上游 `Qwen3MLP` 的分块矩阵计算，包括完整 chunk 和尾部 padding，不覆盖 Transformer、KV cache、逐 token 解码、FlashAttention、VeOmni 或 OpenCompass。诊断临时依赖为 `einops==0.8.1`、`opt_einsum==3.4.0`，安装在 `/tmp/in-place-ttt-cpu-deps-20261108/`，没有修改根项目依赖。

下一项运行是在获准 GPU 空闲时，验证第三方 TTT 与 baseline 的完整模型加载及推理。官方论文指标复现仍受原训练语料、原模型权重及实验配置缺失的限制。
