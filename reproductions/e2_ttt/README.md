# 20261108_E²-TTT 第一阶段复现

创建时间：20261108 17:02:06 Asia/Shanghai

最后修订时间：20261109 12:12:01 Asia/Shanghai

本目录保留 [E²-TTT 官方实现](https://github.com/zeyun-zhong/E2-TTT)，使用其原始模型、flame 训练流程与随附 lm-evaluation-harness；不接入本项目的统一框架。论文为 [Rethinking Expressivity and Efficiency in Test-Time Training](https://arxiv.org/abs/2608.21308)。

## 当前状态

官方源码已迁入，revision 与许可证见 [UPSTREAM.md](UPSTREAM.md)。原始标量核与验证函数的 CPU 数值检查及模型／训练源码语法检查已通过，见[执行记录](20261108_phase1_record.md)。本地数据评估入口已接入，两个 1.3B checkpoint 均已在物理 GPU 0 完成真实样本生成验证。

每个 checkpoint 验证 SQuAD completion 2 条、SWDE 2 条，以及 RULER 六个长度各 2 条，共 16 条；MLP 与 SwiGLU 合计 32 条，均无运行异常。峰值显存分别约 9.96 GiB 和 10.12 GiB。该检查覆盖完整模型加载及生成，完整 benchmark 尚未完成，不能据此报告论文指标复现；运行记录见 [GPU 验证与评估](../20261109_checkpoint_gpu_evaluation.md)。

## 复现目标

当前评估官方公开的 **1.3B MLP 和 1.3B SwiGLU**，与原论文的 RULER 长上下文主结果保持相同模型规模。所选 benchmark 为 SQuAD completion、SWDE 和 RULER；各自沿用官方评估范围与样本数。

下表为论文 `tab:incontext_retrieval_main` 中的原文参照，不是本次测量。SQuAD、SWDE 保留原生 HFLM 的总长度上限 2048：为最多 48 个生成 token 预留空间，prompt 超过 2000 tokens 时左截断。

| 指标 | 1.3B MLP | 1.3B SwiGLU |
|---|---:|---:|
| SWDE accuracy (%) | 37.2 | 47.3 |
| SQuAD accuracy (%) | 45.6 | 44.4 |

RULER 完整执行原文的 S-NIAH-1 和 S-NIAH-2，长度为 512、1K、2K、4K、8K、16K；官方生成器默认每项、每个长度 500 条样本。结果对照原文长上下文主图及附录的 S-NIAH 数值表。

两个模型都在[作者的 Hugging Face collection](https://huggingface.co/collections/zeyun-zhong/e2-ttt)中公开。340M 版本保留作可选的小规模运行参考。

本轮评估数据的任务、规模、共用范围及完整下载与构造命令见 [评估数据准备](../20261108_ttt_data_preparation.md)。

## 模型下载

首批准备官方 1.3B MLP 和 SwiGLU 两个完整 checkpoint，均包含权重、配置和 tokenizer。下载前确保已有 `hf` CLI，安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
hf download zeyun-zhong/e2-ttt-mlp-1.3B-15B \
  --local-dir ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b

hf download zeyun-zhong/e2-ttt-swiglu-1.3B-15B \
  --local-dir ~/rivermind-data/models/reproductions/e2_ttt/swiglu-1.3b-15b
```

## 环境与评估

本轮 checkpoint 评估使用 [requirements-eval.txt](requirements-eval.txt) 固定依赖。在服务器项目根目录安装并激活：

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh e2_ttt
source ~/rivermind-data/envs/ttt-e2_ttt/bin/activate
```

版本选择与安装记录见 [评估环境](../20261109_environment_setup.md)。以下上游完整环境说明用于训练依赖参考，不是本轮评估环境的安装步骤。


目标运行环境是 Linux x86-64、NVIDIA CUDA；保留官方 Python 3.12、PyTorch 2.8.0、CUDA 12.8 wheel、flash-linear-attention 0.4.1、Transformers 4.57.1 与 torchtitan 固定 revision。完整依赖见 [environment.yml](upstream/environment.yml)。本项目根环境只用于源码检查，不替代这套运行环境。

在 GPU 服务器上从项目根目录建立官方环境：

```bash
cd reproductions/e2_ttt/upstream
conda env create -f environment.yml
conda activate e2ttt
pip install -e .
pip install -e lm-evaluation-harness
```

在上述方法环境中，从项目根目录运行本地评估入口：

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/e2_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --output-dir artifacts/reproductions/e2_ttt/eval/mlp-1.3b-15b
```

中断后，使用相同 checkpoint、任务、数据、长度和输出目录追加 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/e2_ttt/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --output-dir artifacts/reproductions/e2_ttt/eval/mlp-1.3b-15b \
  --resume
```

恢复时核对已有 `plan.json`，除创建时间外的设置须相同。每个输出目录使用原生 `CachingLM` 的 `generation_cache.sqlite` 保存生成响应，原 HFLM 在每条生成完成后通过 cache hook 写入。恢复会复用已保存响应、重新生成未缓存请求，并重建完整逐样本结果与汇总；不将旧汇总当作已完成任务直接跳过。

SwiGLU 使用 `~/rivermind-data/models/reproductions/e2_ttt/swiglu-1.3b-15b`，并指定独立的新输出目录。入口直接使用已激活环境的 `python`，默认物理 GPU 0、单进程，不依赖 Accelerate 启动配置。`CUDA_VISIBLE_DEVICES` 仅接受单卡 `0` 或 `1`，拒绝其他分配；当前使用空闲的物理 GPU 0。进程内的 `cuda:0` 对应所选物理 GPU。

默认评估 `ruler squad_completion swde`，只读取本地文件：

| 参数 | 默认数据目录 | 执行范围 |
|---|---|---|
| `--ruler-data` | `data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b` | 两项 S-NIAH，六个长度，每项每长度 500 条 |
| `--squad-data` | `data/benchmarks/based-squad` | `data/validation-00000-of-00001.parquet` 的完整 validation |
| `--swde-data` | `data/benchmarks/based-swde` | `data/validation-00000-of-00001.parquet` 的完整 validation |

可用 `--benchmarks ruler` 等选择所需任务，使用 `--lengths 4096 8192 16384` 选择 RULER 长度；默认长度为 512、1024、2048、4096、8192、16384。数据目录参数均可显式覆盖；默认路径相对于项目根目录。

RULER 直接拼接每行 `input + answer_prefix`，不加 chat 模板、不重复添加前缀。沿用原 harness 的生成与子串计分，预留 128 个输出 token；用实际 HFLM 编码检查 prompt 加输出预算不超过该档长度，超限时报错，禁止截断。不现场生成样本或访问 essays 数据集。

SQuAD／SWDE 的本地 task 继承原 task，保留 `.strip()`、换行停止符、最多 48 个生成 token 及原 `contains_score`。总序列上限固定 2048，实际 prompt 上限为 2000，超过时按原 HFLM 左截断；不采用训练长度 2240 作为默认评估上限。

首次执行新建 `--output-dir`，续评复用同一目录：`plan.json` 记录 checkpoint、数据、预算和运行依赖；`predictions.jsonl` 保存原 harness 逐样本记录及任务名；`native_harness/` 保存各次原始汇总；`summary.json` 按任务和 RULER 长度汇总百分比准确率及样本数；`evaluation.log` 与 `execution.json` 保存执行日志、退出状态、本次调用耗时和 CUDA allocated／reserved 峰值显存。每完成一项真实检索任务或一个 RULER 长度即保存结果，仅全部任务完成后设置 `completed=true`。不计算 bootstrap 标准误，准确率与原计分相同。

本地入口保留模型及原始 `upstream/training/eval.sh` 不变，使用随附 harness 的 Python API 接入本地数据。服务器评估环境已安装，通过依赖检查、本地入口及模型／harness／任务模块导入，并完成上述 32 条真实样本的 GPU 生成验证；原生评分导出与完整 benchmark 结果另行记录。

## 原训练设置与备用入口

当前使用公开 1.3B checkpoint 评估。其原始训练长度为 2240、24 层、hidden size 2048、16 heads；训练 28,672 steps，global batch 256，实际约 16.4B tokens，论文和仓库命名统称约 15B。原始训练入口为 `upstream/training/train_1B.sh`，该脚本固定多卡配置，不能直接作为本轮单卡评估命令。

下面保留已迁入的 340M 单卡训练入口及原设置，作为备用参考。

340M 原始模型为 24 层、hidden size 1024、8 heads；SWA window 与 TTT chunk 均为 512，训练 sequence length 为 2048。采用 `HuggingFaceFW/fineweb-edu` 的 `sample-100BT`，tokenizer 为 `fla-hub/transformer-1.3B-100B`。

原始训练预算为 28,672 optimizer steps、global batch 256，共 15,032,385,536 tokens；学习率 `1e-3`、warmup 1024 steps、cosine 最小比例 `0.1`、seed 42。论文报告 340M 训练约 132 H100 GPU-hours；该耗时不是当前服务器的预算估计。

官方 `train_340M.sh` 固定 `NGPU=4`、每卡 batch 4、梯度累积 16。备用训练入口固定物理 GPU 1，提供 [train_340m_gpu1.sh](scripts/train_340m_gpu1.sh)：单卡、每卡 batch 4、梯度累积 64，保持 global batch 256 与原始 token 预算，仍调用原 `training/train.sh`。硬件和归约次序与原实验不同，实际显存、速度与训练结果需要在服务器上验证。

准备数据时使用[官方 Data 流程](upstream/training/README.md#data)。该设置会读取 `sample-100BT`，需要数百 GB 缓存；本次尚未下载或处理训练数据。

从项目根目录、已激活官方环境后执行：

```bash
bash reproductions/e2_ttt/scripts/train_340m_gpu1.sh \
  reproductions/e2_ttt/upstream/training/configs/e2_ttt_mlp_340M.json \
  artifacts/reproductions/e2_ttt/train/mlp-340m-15b
```

SwiGLU 使用 `upstream/training/configs/e2_ttt_swiglu_340M.json` 和独立输出目录。原始 `train.sh` 保存训练日志和 DCP checkpoint，并在训练结束后转换为 Hugging Face 格式；原始配置关闭 W&B，本阶段不新增在线跟踪项目。训练复现结果和公开 checkpoint 的评估结果分别记录。
