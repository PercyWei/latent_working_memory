# v3：token memory 容量分配对照

实现依据：`notes/v3/20260926_step1_damage_guided_capacity_experiment.md`。首版实现五个比较对象及动态方法的**局部写入版本**，用于检验“按信息损失程度扩容”能否在相近容量下改善旧信息保留。

## 模型与方法

统一使用 **Qwen/Qwen3-4B-Instruct-2507，每块 64 slots**。

| 共用部分 | 实现 |
|---|---|
| 写入 | 新文本末尾添加可训练 gist embeddings，取其最终隐藏状态作为记忆 |
| 可训练参数 | 编码 LoRA（rank 128、alpha 32，attention/MLP projections）与 gist embeddings |
| 初始化 | gist embeddings 使用零均值高斯分布，标准差 0.02 |
| 读取 | 同一冻结基座关闭 LoRA；训练时仍保留读取损失对记忆的梯度 |
| 数值与位置 | 基座 bfloat16、LoRA/embeddings float32；普通因果位置编码，LoRA dropout 为 0 |
| 训练框架 | 复用 verl `BaseEngine`、优化器和 replicated DDP；一个外层 forward 并行展开一个 microbatch 的完整轨迹 |

| 方法标识 | 写入范围与容量 | 训练流程 |
|---|---|---|
| `icae_single` | 完整原文一次压缩为一个块 | AE＋LM → 全历史 QA |
| `icae_multi` | 每段独立压缩，拼接所有块 | 多段 AE＋LM → 全历史 QA |
| `autocompressors` | 新段与累计记忆共同压缩，始终追加 | 随机分段的 next-token LM |
| `memory_change` | 覆盖末块或追加新块；按归一化表示变化决定 | 单段 AE＋LM → 动作预热 QA → 策略 QA |
| `information_loss` | 同样的局部写入；按历史可读性损失决定 | 与上一组相同，后两阶段各自训练 |

动态追加输入为 `X_t`，覆盖输入为 `[A, X_t]`，其中 `A` 是当前最后一个块；更早的块不传入写入器。两种动作共享 LoRA 与 gist embeddings。完整历史动作标记、限制 attention、双 LoRA 三种写入版本留待后续架构对照。

冻结读取端、统一 LoRA 与统一初始化均属于本实验的共同设置。三个基线遵循相应的记忆组织与训练目标，在此共同设置下进行对照。

## 两种扩容规则

记忆变化分数对每个 slot 做 RMS 归一化，计算覆盖候选与旧末块的相对 Frobenius 距离；`I >= threshold_i` 时追加。实际保存的向量不归一化。

信息损失使用数据中固定的历史门控 QA：

```text
L0   = 更新前记忆的平均答案 NLL
Lrw  = 覆盖候选记忆的平均答案 NLL
Lapp = 追加候选记忆的平均答案 NLL
d = Lrw - L0; g = Lrw - Lapp
追加 ⇔ g > threshold_g 或 (d > threshold_d 且 g > eta)
```

- 每题先平均答案 token，再平均题目；门控无梯度，不筛题。
- 未选中候选不计算辅助训练损失；任务 QA 不包含门控题。
- 信息损失法在评估时也使用门控参考答案，输出明确标为 **offline oracle**。
- `0.1` 等默认阈值仅用于启动配置，尚未校准。每组策略训练的阈值写入 checkpoint；容量对照需要分别训练多个阈值配置。

## 数据与损失

| 数据 | 规范格式 | 处理 |
|---|---|---|
| AE＋LM | 基础构造器的 `TextSample`，目录内 `train/dev/test.jsonl` | 使用当前基座分词，按输入长度筛选，不截断文本或目标；记录真实 AE/continuation 比例 |
| QA | FactQA 发布目录的 `preparation.json` 与 `train/dev/test.jsonl` | 校验已有来源、事实与题池隔离；保留原文、分段、更新点及 `usage` |

- QA 加载支持现行构造流程的 **6–10 段**，不会强制重切为旧笔记中的八段，也不会使用预训练长度筛选参数裁剪轨迹。
- 训练入口读取并验证数据后记录实际 token 长度与题数。AE＋LM 的 train/dev 在长度筛选后必须仍包含两种任务。1000 条正式 QA 尚未在本机，本次测试使用合成小样本。
- 阶段衔接保存并核对 AE＋LM 来源文档与去重簇，QA 不能与其重叠。
- ICAE-single 需要覆盖完整历史长度的预训练样本；ICAE-multi 的预训练样本按 `segment_tokens` 独立写入后联合读取。现有短文本数据不能替代完整长度的基线预训练。

| 目标 | 实现细节 |
|---|---|
| AE / LM | 分别重建输入、预测未写入的续写；目标末尾添加 EOS。比例由输入数据决定，统计随 run 保存 |
| ICAE QA | 完整记忆上读取全部任务题，按实际题数平均 |
| 动态 QA | 每个更新点读取 `task_new_qa_ids + task_old_qa_ids`，按实际题数平均，再平均更新点 |
| 动作预热 | 每次更新以 0.5 概率追加；由 seed、epoch、trajectory_id 确定，两组动作日程相同 |
| 策略训练 | 采用各自门控；选中路径保留完整跨步梯度，一条轨迹内不更新参数、不 detach 记忆 |
| AutoCompressors LM | 随机分段，累计记忆参与预测及写入；默认每两段为一个 BPTT 子块，之后 detach 累计记忆，参数在整条样本结束后更新 |

AutoCompressors 对 AE 样本使用输入全文，对 continuation 样本使用输入和续写的 token 流。子块内保留跨分段的下一 token 预测，子块首 token 不计损失；按目标 token 数平均。冻结读取端的损失通过当前子块内的记忆写入回传，不使用 QA 微调。当前随机分段范围为 768–1024 tokens，尾段可以更短。

每个 global batch 按实际样本数平均，包括不足一批的尾批。配置项 `micro_batch_size_per_gpu` 控制每卡一次并行处理的完整样本数，`gradient_accumulation_steps` 控制累积次数；全局 batch 根据两者与 GPU 数的乘积计算并记录。默认每卡 microbatch 为 1、累积 4 次，双卡全局 batch 为 8；双卡也可用 microbatch 2、累积 2 次保持全局 batch 为 8。

批量写入采用独立行的右侧 padding 与 attention mask，不跨样本读取信息；动态分支按轨迹独立决策。`qa_batch_size` 控制每条轨迹一次读取的题数，批量调用合并各活跃轨迹的题目，但保留原有每题、更新点和轨迹的损失权重。多个样本共享调用的计时按参与样本分摊，调用/题目数仍按每条样本的逻辑工作量记录。

原生 Transformer 梯度检查点保持关闭；长目标的冻结输出层 CE 单独分块重算，以减少词表 logits 的显存占用。实际 4B 长轨迹的显存与吞吐需在 GPU 上测量。

## 启动

终端和公司 GPU 网站使用 [run_gpu.sh](scripts/run_gpu.sh)，默认仓库为 `/dfs/data/latent_working_memory`，SwanLab project 为已确认的 `latent-working-memory-v3`。`--mode smoke/pilot/full` 控制试跑程度，自动衔接训练阶段、最终评估和对照结果；参数与启动命令见 [GPU 任务说明](scripts/README.md)。

当前服务器的仓库位于 `~/percyw/latent_working_memory` 时，设置 `LWM_REPO_DIR="$HOME/percyw/latent_working_memory"`。启动时显式传入 `--gpus 4,5`，训练使用两个进程，最终评估使用 GPU 4。`--gpus` 接受非重复的非负物理卡号，默认仍为 `0,1`。

已有预训练 checkpoint 时，动态方法可用 `--stage auto --init-checkpoint <预训练文件>` 连续完成 warmup → policy → 最终评估；省略 `--init-checkpoint` 则从预训练开始。批处理参数使用 `--micro-batch-size-per-gpu` 与 `--gradient-accumulation-steps`，同样支持下方直接调用训练模块的命令。

以下是单独调用训练与评估模块的方式。

所有命令在项目根目录运行；相对路径以该目录为基准。六份预设位于 `configs/v3/`，先填写实际数据路径。默认只保存本地记录；未指定 SwanLab project。

**共享基础预训练：**

```bash
CUDA_VISIBLE_DEVICES=4,5 uv run --frozen torchrun --standalone --nproc_per_node=2 \
  -m latent_working_memory.v3.train \
  --config configs/v3/dynamic_pretrain.json \
  --dataset-dir /absolute/path/to/fineweb-text-samples \
  --device cuda
```

输出 `training-result.json` 给出最后一个 checkpoint 的准确路径。分别加载它进行两组预热：

```bash
CUDA_VISIBLE_DEVICES=4,5 uv run --frozen torchrun --standalone --nproc_per_node=2 \
  -m latent_working_memory.v3.train \
  --config configs/v3/memory_change_warmup.json \
  --dataset-dir /absolute/path/to/fineweb-factqa \
  --init-checkpoint /absolute/path/to/pretrain/checkpoints/step-NNNNNN.pt \
  --device cuda
```

将预设替换为 `information_loss_warmup.json`，运行另一组。之后各自进入策略训练；每个阶段使用新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=4,5 uv run --frozen torchrun --standalone --nproc_per_node=2 \
  -m latent_working_memory.v3.train \
  --config configs/v3/memory_change_warmup.json --stage policy \
  --dataset-dir /absolute/path/to/fineweb-factqa \
  --init-checkpoint /absolute/path/to/warmup/checkpoints/step-NNNNNN.pt \
  --output-dir artifacts/v3/step1-token-memory/train/memory-change-policy \
  --device cuda
```

ICAE 两组使用各自预设预训练，再通过 `--stage qa`、QA 数据目录及相应 `--init-checkpoint` 切换阶段。AutoCompressors 使用 `autocompressors_lm.json`，直接进行 LM 训练。

续训沿用原配置、原输出目录、卡数和批处理设置：

```bash
CUDA_VISIBLE_DEVICES=4,5 uv run --frozen torchrun --standalone --nproc_per_node=2 \
  -m latent_working_memory.v3.train --config /absolute/path/to/run/config.json \
  --resume /absolute/path/to/run/checkpoints/step-NNNNNN.pt --device cuda
```

checkpoint 保存 LoRA、gist embeddings、optimizer、训练游标、各 rank RNG 与配置/数据身份。新阶段只继承可训练权重。预训练基座不重复保存在 checkpoint；加载时沿用已记录的模型 revision。

## 评估与产物

默认评估最后一个 checkpoint，不改写训练阈值。所有方法在同一批轨迹的最终记忆上回答评估题；按末段新事实、更早旧事实以及段距离分别统计。

```bash
CUDA_VISIBLE_DEVICES=4 uv run --frozen python -m latent_working_memory.v3.evaluate \
  --checkpoint /absolute/path/to/run/checkpoints/step-NNNNNN.pt \
  --dataset-dir /absolute/path/to/fineweb-factqa \
  --output-dir artifacts/v3/step1-token-memory/eval/memory-change-policy \
  --split test --device cuda --max-new-tokens 64

uv run --frozen python -m latent_working_memory.v3.compare \
  /absolute/path/to/eval-one/summary.json /absolute/path/to/eval-two/summary.json \
  --output-dir artifacts/v3/step1-token-memory/compare/capacity
```

| 产物 | 内容 |
|---|---|
| `config.json` / `run.json` | 已解析配置、模型 revision、来源范围、真实数据统计与运行身份 |
| `metrics.jsonl` | optimizer step 对应的训练、开发集损失、容量、整步耗时及 CUDA 峰值显存 |
| `checkpoints/step-*.pt` | 可续训 checkpoint；跨阶段加载同一规范格式 |
| `trajectories.jsonl` | 动作轨迹、门控分数、容量、计时、逐题答案与 NLL/EM/F1 |
| `summary.json` | 问答质量、按段距离分层、平均/最终容量与读写成本 |
| `points.json` / `points.csv` | 同一题池的质量—容量比较点，保留阈值与 oracle 标记 |

写入计时包含实际生成的所有候选；门控题实例数与任务读取分开。CUDA 计时显式同步。ICAE-single 的平均 slots 指唯一一次完整压缩状态；其他方法平均所有更新点的保存容量。生成采用 greedy decoding；提示及答案格式由配置固定。

设置 `training.swanlab_project` 和显式 `group` 可启用训练/开发集增量记录；使用已有或已获确认的项目。最终评估默认本地，可通过 `--log-to-swanlab` 追加到原训练 run。未记录的评估指标不会补零。

## 验证与边界

```bash
uv run --frozen pytest -q tests/v3
uv run --frozen ruff check src/latent_working_memory/v3 tests/v3
```

测试使用本地随机初始化 tiny Llama/Qwen3，覆盖可微冻结读取、分支梯度、历史输入范围、门控隔离、数据契约、BPTT 截断及 CPU DDP。它们验证实现行为，不代表真实 QA 质量或 4B GPU 性能；正式运行前仍需接入实际数据并测量完整轨迹显存。

参考实现：[ICAE](https://github.com/getao/icae)、[AutoCompressors](https://github.com/princeton-nlp/AutoCompressors)、[verl Model Engine](https://verl.readthedocs.io/en/latest/workers/model_engine.html)。
