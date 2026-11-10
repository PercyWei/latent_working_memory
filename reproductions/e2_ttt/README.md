# 20261108_E²-TTT 第一阶段复现

创建时间：20261108 17:02:06 Asia/Shanghai

最后修订时间：20261110 11:34:26 Asia/Shanghai

本目录保留 [E²-TTT 官方实现](https://github.com/zeyun-zhong/E2-TTT)及随附 lm-evaluation-harness，使用原模型与评估逻辑。论文为 [Rethinking Expressivity and Efficiency in Test-Time Training](https://arxiv.org/abs/2608.21308)；源码边界与 revision 见 [UPSTREAM.md](UPSTREAM.md)。

## 当前状态与评估范围

第一阶段的源码与评估入口迁移已完成，官方 MLP／SwiGLU 1.3B checkpoint 均通过完整加载、真实样本生成及原生计分检查。尚未完成完整 benchmark，本地推理优化未实施。验证及优化分析结论见[复现状态](../20261110_ttt_reproduction_status.md)。

默认每个 checkpoint 独立评估以下完整范围，共 **10,095 条**：

| benchmark | 任务与长度 | 样本数 | 默认数据目录 |
|---|---|---:|---|
| RULER | S-NIAH-1／2，512、1K、2K、4K、8K、16K；每项每长度 500 条 | 6,000 | `data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b` |
| SQuAD completion | 完整 validation | 2,984 | `data/benchmarks/based-squad` |
| SWDE | 完整 validation | 1,111 | `data/benchmarks/based-swde` |

下载与构造命令见[评估数据准备](../20261108_ttt_data_preparation.md)。论文 `tab:incontext_retrieval_main` 的准确率参照如下，RULER 对照原文长上下文主图及附录的 S-NIAH 数值表。

| 指标 | 1.3B MLP | 1.3B SwiGLU |
|---|---:|---:|
| SWDE accuracy (%) | 37.2 | 47.3 |
| SQuAD accuracy (%) | 45.6 | 44.4 |

## 模型下载

官方两个完整 checkpoint 均包含权重、配置和 tokenizer，见[作者 collection](https://huggingface.co/collections/zeyun-zhong/e2-ttt)。`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
hf download zeyun-zhong/e2-ttt-mlp-1.3B-15B \
  --local-dir ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b

hf download zeyun-zhong/e2-ttt-swiglu-1.3B-15B \
  --local-dir ~/rivermind-data/models/reproductions/e2_ttt/swiglu-1.3b-15b
```

## 环境与评估命令

评估依赖固定于 [requirements-eval.txt](requirements-eval.txt)：Python 3.12、PyTorch 2.8.0 cu128、FlashAttention 2.8.3、FLA 0.4.1、Transformers 4.57.1。安装与版本说明见[评估环境](../20261109_environment_setup.md)。在服务器仓库根目录安装并激活：

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh e2_ttt
source ~/rivermind-data/envs/ttt-e2_ttt/bin/activate
```

环境已安装时只需激活。从仓库根目录分别运行两个 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/e2_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --output-dir artifacts/reproductions/e2_ttt/eval/mlp-1.3b-15b

CUDA_VISIBLE_DEVICES=0 bash reproductions/e2_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/e2_ttt/swiglu-1.3b-15b \
  --output-dir artifacts/reproductions/e2_ttt/eval/swiglu-1.3b-15b
```

首次运行的输出目录须不存在。中断后保持相同 checkpoint、任务、数据、长度和输出目录，追加 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/e2_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --output-dir artifacts/reproductions/e2_ttt/eval/mlp-1.3b-15b \
  --resume
```

入口使用方法环境的 `python`，单进程运行，`CUDA_VISIBLE_DEVICES` 仅接受单卡 `0` 或 `1`，进程内设备为 `cuda:0`。默认评估 `ruler squad_completion swde`；可用 `--benchmarks` 选择任务，`--lengths` 选择 RULER 长度，`--ruler-data`／`--squad-data`／`--swde-data` 覆盖数据目录。相对数据路径以仓库根目录为准。

## 评估协议与输出

- RULER 使用 `input + answer_prefix`，不加 chat 模板，greedy 生成最多 128 tokens；按原 HFLM 编码检查总预算，超限报错，不截断输入。
- SQuAD／SWDE 读取目录内的 `data/validation-00000-of-00001.parquet`，保留原 `.strip()`、换行停止符、48-token 输出上限与 `contains_score`。总预算 2048，prompt 超过 2000 tokens 时按原 HFLM 左截断。
- 每次生成使用独立 cache，保持原模型、task、harness 与 `upstream/training/eval.sh` 不变。Bootstrap 设为 0，仅省略标准误，不改变准确率。
- 续评严格核对 `plan.json`，通过原生 `CachingLM` 的 `generation_cache.sqlite` 复用已保存响应，重新生成缺失请求并重建完整结果。
- `predictions.jsonl` 保存逐样本记录，`native_harness/` 保存原始汇总；`summary.json` 按任务／长度汇总，仅全部任务完成后设置 `completed=true`。`evaluation.log`、`execution.json` 记录运行日志、状态、耗时和峰值显存。

## 训练备用入口

当前评估不需要重新训练。公开 1.3B 模型为 24 层、hidden size 2048、16 heads，训练长度 2240、28,672 steps、global batch 256，约 16.4B tokens，发布名称统称 15B。原多卡入口与完整训练环境分别见 [train_1B.sh](upstream/training/train_1B.sh)、[environment.yml](upstream/environment.yml)。

保留的 340M 单卡入口固定物理 GPU 1、microbatch 4、梯度累积 64，保持官方 global batch 256 和 28,672 steps。训练数据及处理见[官方 Data 流程](upstream/training/README.md#data)，尚未准备或执行训练。从仓库根目录、已激活官方训练环境后运行：

```bash
bash reproductions/e2_ttt/scripts/train_340m_gpu1.sh \
  reproductions/e2_ttt/upstream/training/configs/e2_ttt_mlp_340M.json \
  artifacts/reproductions/e2_ttt/train/mlp-340m-15b
```

SwiGLU 替换为 `e2_ttt_swiglu_340M.json` 并使用独立输出目录。训练与公开 checkpoint 的评估结果分别记录。
