# 20261108_LaCT 第一阶段复现

创建时间：20261108 17:03:39 Asia/Shanghai

最后修订时间：20261109 12:11:28 Asia/Shanghai

本目录接入 [Test-Time Training Done Right](https://arxiv.org/abs/2505.23884v1) 的官方语言模型实现，保留其与 sliding-window attention 并行的 SwiGLU fast weights、负内积写入、Momentum / Muon 与逐次权重归一化。已接入本地 RULER checkpoint 评估入口，完成源码静态检查、原始写入函数的 CPU 前后向检查，以及公开 Momentum checkpoint 在 4K、32K 真实样本上的完整生成；尚未完成正式 RULER 评估或论文指标复现。

## 代码与运行入口

- `upstream/lact_llm/`：官方完整语言模型源码、五份模型配置和环境要求。
- `upstream/minimal_implementations/`：官方简化实现，供对照阅读。
- `upstream_flame/`：官方指定的 flame 训练框架源码。
- `configs/760m_muon_paper_single_gpu.toml`：760M Muon 的单卡运行配置，保留公开命令的全局 token batch 和步数。
- `scripts/train_760m_muon.sh`：在 `zcy-memory` 的单张物理 GPU 上调用原生 flame trainer。
- `eval.sh`、`scripts/evaluate.py`：在单张物理 GPU 上读取本地六项 RULER JSONL，使用官方 LaCT 模型生成并计分。
- [UPSTREAM.md](UPSTREAM.md)：固定源码 revision、文件范围和必要接入改动。
- [源码迁入记录](20261108_execution.md)：源码接入与本地 CPU 检查。
- [GPU 验证与评估记录](../20261109_checkpoint_gpu_evaluation.md)：完整模型运行状态与正式评估进度。

本地不引入统一训练框架，不更改官方模型或更新函数。flame 的 custom-model 注册仅负责使 Transformers 能够按 `lact_swiglu` 配置加载模型。

## 目标与论文参照

当前先评估作者公开的 760M Momentum、RoPE、PreNorm checkpoint；Muon 变体按需补充。下面列出原论文的训练设置与指标供参照，公开 checkpoint 是作者按相同训练设置重训发布的模型，不代表论文表格对应的同一次训练产物。

| 项目 | 原论文设置 |
|---|---|
| 模型 | 24 层，hidden size 1536，TTT heads 4，低秩初始状态 rank 32 |
| 序列 / TTT chunk / SWA | 32768 / 2048 / 2048 tokens |
| 训练集 | Long-Data-Collections，约 68.8B tokens，Mixstra tokenizer、vocab size 32000 |
| 数据配比 | 41.4% General Data、58.6% Instruction Data |
| 训练预算 | 论文称 40B tokens；公开命令 40960 optimizer steps、全局 batch 32 条 32768-token 序列 |
| 外层优化 | AdamW，lr 0.001，1024-step warmup，cosine decay，seed 42 |
| 原硬件 | 32 × A100-40GB，约 20 小时 |
| 长文本 loss | Book-3，760M 模型评估 2.5B tokens，绘制按位置的 NTP loss |
| 检索 | RULER，4K / 8K / 16K / 32K |

论文 `tab:ruler-single` 中 760M Momentum 的 S-NIAH-1/2/3 平均准确率为 **89.9 / 64.3 / 40.9 / 33.1**，Muon 为 **77.9 / 60.5 / 40.0 / 35.1**（依次对应 4K / 8K / 16K / 32K，单位 %）。`tab:ruler-multi` 中 MK-NIAH、MQ-NIAH、MV-NIAH 的平均值，Momentum 为 **41.7 / 29.7 / 18.5 / 9.1**，Muon 为 **42.9 / 32.1 / 17.1 / 7.5**。这些是原文参照，不是本次运行结果。

本轮评估数据的任务、规模、共用范围及完整下载与构造命令见 [评估数据准备](../20261108_ttt_data_preparation.md)。公开的 Momentum／RoPE／PreNorm、Muon／NoPE／PostNorm 和 Muon／NoPE／PreNorm 三个 760M checkpoint 的 tokenizer 与特殊 token 配置一致，共用一份 12000 条数据。构造命令使用 Momentum 目录作为 tokenizer 来源，两个 Muon 变体无需另行生成数据；三个模型分别在同一数据集上评估。

## 模型下载

作者在[官方 issue 回复](https://github.com/a1600012888/LaCT/issues/2#issuecomment-3282879455)中发布了 [YunjinZhang/lact_llm](https://huggingface.co/YunjinZhang/lact_llm/tree/main)，说明使用 32K 序列、Long-Data-Collections 和 40B tokens 重训了若干 760M 模型。下载前确保已有 `hf` CLI，安装方式见[复现总说明](../README.md#模型下载约定)。

首批下载 Momentum、RoPE、PreNorm 版本：
```bash
hf download YunjinZhang/lact_llm \
  --include 'lact-momentum-rope-prenorm-nheads4-chunk2048-760m/*' \
  --local-dir ~/rivermind-data/models/reproductions/lact
```
可选：同时下载 Muon、NoPE 的 PreNorm 和 PostNorm 两个变体：
```bash
hf download YunjinZhang/lact_llm \
  --include 'lact-muon-*/*' \
  --local-dir ~/rivermind-data/models/reproductions/lact
```

## 本地 RULER 评估

先按 [评估数据准备](../20261108_ttt_data_preparation.md) 准备 `data/benchmarks/ruler-v1/reproductions/lact-760m/`。确认 `zcy-memory` 的物理 GPU 0 空闲、激活 LaCT 的原生 CUDA 环境后，从项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/lact/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --output-dir artifacts/reproductions/lact/eval/momentum-rope-prenorm-ruler
```

`eval.sh` 使用当前方法环境的 `python`，设置官方模型与项目源码路径，仅接受 `CUDA_VISIBLE_DEVICES=0` 或 `1`，未指定时默认 `0`。进程内设备为 `cuda:0`，实际物理设备选择写入 `plan.json`。`--checkpoint` 和 `--output-dir` 必需，首次运行的输出目录必须尚不存在。可选参数如下：

| 参数 | 默认值 |
|---|---|
| `--ruler-data` | `data/benchmarks/ruler-v1/reproductions/lact-760m` |
| `--lengths` | `4096 8192 16384 32768`，可选其中的子集 |
| `--resume` | 续评已有输出目录，默认关闭 |

中断后使用相同设置和输出目录续评：

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/lact/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --output-dir artifacts/reproductions/lact/eval/momentum-rope-prenorm-ruler \
  --resume
```

续评核对 checkpoint 配置、数据路径、长度、任务、生成和精度等设置，跳过已保存的样本 ID，并追加新的预测；时间和运行设备信息可以变化。仅删除最后一条未写完的 JSON 记录；已完成记录的损坏、重复 ID 或样本不匹配均报错。

两个 Muon 模型使用对应的 checkpoint 目录和新的输出目录，数据目录共用。评估要求每个长度的六项 `test.jsonl` 各有 500 条样本，不在入口内生成或截断样本。每条输入按 `input + answer_prefix` 拼接，再由 checkpoint tokenizer 添加默认 special tokens；完整输入与 128-token 输出预算之和必须不超过长度档位。采用 greedy、单样本生成，参考 `outputs` 仅用于计分。

**官方生成路径没有 fast-weight cache。** `prepare_inputs_for_generation` 和模型 `forward` 均强制 `use_cache=False`。本入口保留这一行为，每生成一个 token 都将完整前缀送入原模型，重新执行各层的 chunk 写入及读出；没有补写原实现不更新的当前尾段。生成只计算最后位置的 vocabulary logits（`logits_to_keep=1`），但模型的完整前缀计算仍然存在。因此该入口可用于模型质量评估，耗时不能解释为高效增量解码的性能。

参数以 float32 加载，计算使用 CUDA bfloat16 autocast，实际 dtype 记录在 `plan.json`。关闭外层 autograd；原始 TTT 的负内积梯度已实现为显式矩阵运算，仍会执行记忆更新。入口不修改 checkpoint 的 Muon、PreNorm、RoPE、fused-kernel 或状态精度配置。

checkpoint 仅从本地加载，任何 missing、unexpected、mismatched keys 或加载错误均使评估失败，不接受随机初始化的缺失参数。三个发布目录的名称与实际 head 数相反：Momentum 的配置及权重为 **2 heads**，两个 Muon 为 **4 heads**。已读取 safetensors 元数据确认，具体形状见 [UPSTREAM.md](UPSTREAM.md)；加载时按 checkpoint 自带配置构造，不根据目录名改写。

输出包含：

- `plan.json`：checkpoint、实际模型配置、数据目录、任务与样本数、生成与精度设置。
- `evaluation.log`：加载、逐样本进度及失败信息。
- `predictions.jsonl`：逐样本输出、参考、原始百分比得分、token 数与耗时；ID 使用行序号，另存可能重复的源 `index`。
- `summary.json`：每任务、每长度的分数，以及对应论文表格的三项 S-NIAH 平均、MK-NIAH-1／MQ-NIAH／MV-NIAH 平均和六项平均。
- `execution.json`：各次执行的成功或失败状态、耗时及显存峰值，保留累计耗时和历次最大峰值。

评分复用项目保存的 NVIDIA RULER `postprocess_pred` 与 `string_match_all`；最终汇总从全部已保存的原始预测重新计分，逐样本不取整，汇总百分比保留两位小数。只有全部所选样本完成才写入 `summary.json`；部分预测文件不代表完整评估结果。汇总耗时包含历次已记录执行的加载、编译、生成和失败耗时。

已完成 shell、Python 语法和 Ruff 静态检查，并在服务器安装独立评估环境。公开 Momentum checkpoint 已严格加载，在物理 GPU 0 上完成 4K、32K 各一个真实样本的完整 128-token 生成；正式 12000 条 RULER 评估尚未完成。这些样本仅确认运行路径，不计入正式 benchmark 分数；结果与资源开销见 [GPU 验证与评估记录](../20261109_checkpoint_gpu_evaluation.md)，依赖信息见 [评估环境](../20261109_environment_setup.md)。

## 环境与数据

本轮 checkpoint 评估使用 [requirements-eval.txt](requirements-eval.txt) 固定依赖。在服务器项目根目录安装并激活：

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh lact
source ~/rivermind-data/envs/ttt-lact/bin/activate
```

版本选择与安装记录见 [评估环境](../20261109_environment_setup.md)。以下上游完整环境说明用于训练依赖参考，不是本轮评估环境的安装步骤。


上游完整训练环境需要 Linux x86-64、NVIDIA CUDA，以及 `upstream/lact_llm/requirement.txt` 中的依赖：PyTorch 2.6.x、torchvision 0.21.x、FlashAttention 2.7.4.post1、FLA 等。flame README 另指定 `torchtitan@0b44d4c`。训练框架尚未安装或验证。

若后续进行训练，另建训练环境后按以下入口安装：

```bash
python -m pip install -r reproductions/lact/upstream/lact_llm/requirement.txt
python -m pip install ./reproductions/lact/upstream_flame
python -m pip install 'torchtitan @ git+https://github.com/pytorch/torchtitan.git@0b44d4c'
```

官方没有固定 FLA 与 Transformers 的精确版本。本轮评估分别固定为 0.4.1、4.57.3，属于本项目选定版本，已用于上述 Momentum checkpoint 的 GPU 生成。

若后续开展从头训练，需要准备原 Long-Data-Collections 的 Arrow 文件与原 tokenizer。原仓库提供示例路径，没有提供对应数据构造脚本、精确数据 revision 和 Book-3 / RULER 的完整评估入口。不要用 FineWeb-Edu 替代后，将结果与原 LaCT 论文分数直接对应。模型及 tokenizer 放在 `~/rivermind-data/models/`，训练数据放在项目 `data/` 下，实际路径由下面两个环境变量指定。

## 单卡训练入口

待 `zcy-memory` 的物理 GPU 0 空闲，并完成环境及数据准备后，从项目根目录运行：

```bash
export DATA_FILES='/absolute/path/to/long-data-collections/*.arrow'
export TOKENIZER_PATH='/absolute/path/to/original-mixstra-tokenizer'
CUDA_VISIBLE_DEVICES=0 bash reproductions/lact/scripts/train_760m_muon.sh
```

脚本仅接受物理 GPU 0 或 1，未指定时默认 0，固定一个进程。默认输出为 `artifacts/reproductions/lact/760m-muon-paper/train/`；需要新运行目录时设置 `OUTPUT_DIR`。训练配置使用 microbatch 1、梯度累积 32，使全局 batch 与原 32 卡、每卡 batch 1 的设置相同；这只保持 token batch，不保证浮点归约、数据分片或训练耗时与原实验一致。精确预算为 40960 × 32 × 32768 = 42,949,672,960 tokens，保留公开命令而不自行将它改成另一个“40B”设置。

同一个入口可以传入原生 flame 参数。例如，只运行一个实际训练 step：

```bash
CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR='/absolute/path/to/artifacts/reproductions/lact/760m-muon-one-step/train' \
  bash reproductions/lact/scripts/train_760m_muon.sh --training.steps 1
```

该命令仍执行原 32K 序列和 32 次梯度累积，不能作为低成本语法检查。当前未执行此 GPU 命令。

flame 保存原生 DCP checkpoint、配置和 TensorBoard 本地指标；配置关闭 W&B，不创建外部实验项目。最终 checkpoint 的 HF 转换使用 `upstream_flame/flame/utils/convert_dcp_to_hf.py`。本轮先准备 RULER 评估；Book-3 的精确来源与划分尚待落实。

## 当前待解决事项

1. 在准备好的本地 RULER 数据上完成正式评估，记录公开重训模型的实际分数与原论文参照的差异。
2. 需要评估两个 Muon checkpoint 时，分别完成其严格加载和完整生成验证。
3. 对照当前官方代码与论文 v1 的 Muon 实现差异；详见 [UPSTREAM.md](UPSTREAM.md)。
4. 若后续开展训练，再落实原训练数据、tokenizer 和训练预算。
