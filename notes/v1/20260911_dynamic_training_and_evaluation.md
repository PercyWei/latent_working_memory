# 20260911_动态训练与 QA 评估

创建时间：20260911 15:23:05 UTC+08:00  
最后修订时间：20260912 23:20:25 UTC+08:00

本文记录动态训练与 QA 评估的执行流程。记忆容量、micro epoch 规模、阶段比例及截断跨度通过训练配置确定。

## 1. 准备数据与模型

本文使用以下名称：

- **原始文章**：SQuAD 1.1 中按文章组织的原始数据，包含段落及其问题答案对，用于构造训练或评估样本。
- **训练/评估文本**：从原始文章中选取、实际逐段输入模型的连续正文，由若干完整段落组成。
- **段落**：原始文章中的一个正文段落，附带若干问题及对应参考答案。
- **训练/评估样本**：一份训练或评估文本、关联的问题答案对及本次使用的记忆容量 K 等设置。

训练时，按顺序传入段落，积累初始文本并完成首次压缩；随后在每个段落末更新记忆，再利用更新后的记忆回答指定问题。

先按文章来源确定 train/dev/test，再在各划分内构造训练与评估文本。文章筛选使用训练模型对应 tokenizer 的长度记录，数据来源与统计见[动态数据构建](20260911_dynamic_data_construction.md)。

模型从预训练 checkpoint 初始化，创建独立的 AdamW optimizer。动态阶段训练写入特征投影、Writer、记忆读取投影和 Reader LoRA；语言模型基座与容量网络保持冻结。

一次训练轨迹指从记忆初始化到读完整份文本的处理过程，期间 K 保持固定。

## 2. epoch 与 micro epoch

一个 **epoch** 由 $5m$ 个 micro epoch 组成，五种记忆容量 $K\in\{64,128,256,512,1024\}$ 各出现 m 次，使各容量在训练过程中保持均衡的出现次数。

每个 epoch 开始时，将五种容量各重复 m 次，随机打乱后确定这 $5m$ 个 micro epoch 对应的 K 及执行顺序。

每个 **micro epoch** 使用分配的 K，按目标压缩率 $r\in\{2,4,8\}$ 的占比筛选候选文章、构造并选取训练文本，再遍历样本完成训练，包含多次 optimizer step。

记忆容量与目标压缩率分别选择，各容量共用同一套随训练进度调整的目标压缩率占比策略。以训练 10000 个 epoch 为例：

| Epoch | r=2 | r=4 | r=8 |
|---|---:|---:|---:|
| 1–3000 | 60% | 30% | 10% |
| 3001–7000 | 30% | 40% | 30% |
| 7001–10000 | 10% | 30% | 60% |

表中比例按一个 micro epoch 内实际选用的训练文本数量计算，具体数值由训练配置确定。各目标压缩率的文本数量按整数分配，允许取整差异。

以下流程以一个 micro epoch 为基本单元展开。

## 3. 为当前 micro epoch 构造训练文本

使用当前 micro epoch 的 K，对每个目标压缩率 r 分别确定训练文本长度范围：

$$
L_{\min}=\lceil0.9rK\rceil,\qquad L_{\max}=\lfloor1.5rK\rfloor.
$$

候选文章长度须大于 $L_{\max}$。打乱候选文章顺序，按每篇文章的原始顺序累计完整段落，每份训练文本取满足长度上限的最长连续段落序列，然后从下一段继续构造。同一篇文章可产生多份训练文本，互不重叠。

单段超过长度上限时跳过该段，从其后重新构造训练文本。文章末尾剩余的文本按相同条件筛选。有效训练文本须同时满足：

- 总长度位于 $[L_{\min},L_{\max}]$。
- 首次压缩后至少有一次动态更新。
- 后续动态更新具有可用的 QA 监督。

其中 r 为目标压缩率，用于确定训练文本的长度范围；实际最终压缩率为 $\rho=L/K$，允许范围为 $[0.9r,1.5r]$。例如 r=8 对应的实际最终压缩率为 7.2–12。

按各目标压缩率对应的数量选取有效训练文本，合并为当前 micro epoch 的训练样本。选取过程检查原始段落范围，使不同目标压缩率选中的文本也互不重叠；有效文本不足时提示调整样本数量。每份文本保留来源文章、段落范围、目标压缩率和问题标识，独立开始记忆轨迹。

## 4. 遍历当前 micro epoch 的样本

混合并打乱当前 micro epoch 中不同目标压缩率的训练样本，依次组成 batch。各样本共用当前记忆容量 K。

每卡 microbatch 为 1 个训练样本，通过多卡与梯度累积组成全局 batch B。每个 micro epoch 按打乱后的顺序遍历完整 batch，舍弃末尾不足 B 个的样本。每次参数更新使用 B 个样本。

micro epoch 顺序、候选文章顺序、样本顺序和问题抽样使用由 seed、epoch、micro epoch 及训练实例标识确定的随机源。各梯度传播实验共用数据构造与抽样设置。

## 5. 初始化 memory

每个训练样本从空记忆开始，按顺序积累完整段落。在累计源文本长度首次超过 $1.5K$ 的段落末，执行首次压缩：

$$
M_0=C_\theta(X_{\mathrm{init}};K).
$$

$X_{\mathrm{init}}$ 为初始累计文本，$M_0$ 包含 K 个记忆向量。初始化输入须满足编码器上下文预算。

QA 监督从后续动态更新开始。初始化文本中的问题进入历史候选池，首次压缩通过后续 QA 损失获得梯度。

## 6. 逐段更新并计算 QA 损失

剩余段落依次传入，每次根据旧记忆和当前段落特征生成 K 个新记忆向量：

$$
H_t[i]=P_{\mathrm{in}}(\mathrm{LM}_{\mathrm{frozen}}(X_t)[i])+\mathrm{PE}(S_t+i),
\qquad M_t=\mathrm{Writer}(M_{t-1},H_t;K).
$$

编码器使用当前输入内的局部 RoPE 位置。特征经过可训练投影后，添加固定正弦位置编码，表示各 token 在整份训练文本中的位置。$S_t$ 按已读源 tokens 累计，包含段落分隔符。

每次更新后，从当前段落问题和历史段落问题中分别均匀抽样。历史候选包含初始化文本及此前段落的问题，所有候选的证据均已读入。`new_count`、`history_count` 和 `max_visits` 分别控制两类问题的抽取数量及每题在本次轨迹中的访问上限。

每个问题独立读取当前 memory。Reader 输入由 BOS、投影后的 memory、问题提示及 teacher-forcing 答案组成。训练目标采用首个参考答案并附加 EOS，仅对目标 tokens 计算平均 NLL：

$$
\ell_{a,q}=-\frac{1}{T_{a,q}}\sum_{j=1}^{T_{a,q}}
\log p_\theta(y_{a,q,j}\mid M_{t_q},q,y_{a,q,<j}).
$$

一个训练样本内的各次 QA 等权，当前问题与历史问题使用相同权重。若样本 a 有 $R_a$ 次读取，每个 step 包含 B 个样本，则：

$$
\mathcal L_a=\frac{1}{R_a}\sum_{q=1}^{R_a}\ell_{a,q},
\qquad
\mathcal L_{\mathrm{step}}=\frac{1}{B}\sum_{a=1}^{B}\mathcal L_a.
$$

因此，每个样本对当前 step 的目标权重为 $1/B$。长文章通过生成更多训练样本获得更多训练机会。

## 7. 反向传播与参数更新

三个实验分别使用完整 BPTT、按源 token 数截断的 TBPTT，以及按 memory 更新次数截断的 TBPTT。

完整 BPTT 保留整份训练文本的更新计算图，处理完整份文本后对全部 QA 损失反向传播。TBPTT 按前向处理顺序划分分段，在完成当前段落更新和 QA 后检查累计源 token 数或更新次数；达到阈值时，对本段损失反向传播，再 detach memory，并重新开始计数。段落边界使实际跨度可以超过阈值。

首个 TBPTT 分段至少包含首次压缩和一次具有 QA 监督的后续动态更新，以便首次压缩获得梯度。此后按配置阈值划分分段，处理完整份文本时，对最后一个分段反向传播。

各分段损失统一按所在样本的总读取数和全局 batch size B 归一化。各卡累积本地样本梯度，同步求和后统一裁剪梯度并执行一次 optimizer step。记忆 detach 边界控制梯度传播跨度，batch 边界控制参数更新频率。

QA 激活重算可按显存需求启用，通过重新计算读取过程减少激活占用，并保持对应 BPTT 设置的梯度传播范围。

## 8. 定期评估并保存训练状态

训练开始、指定 step 间隔及训练结束时，在固定的 dev 评估文本上评估。评估固定问题、读取位置、随机种子和生成预算，分别汇总各 K 及实际长度、压缩率下的结果。

相同问题使用以下五个条件：

| 条件 | 回答输入 | Reader LoRA |
|---|---|---|
| `memory` | 当前评估文本在该位置的 memory + 问题 | 启用 |
| `no_memory` | 问题 | 启用 |
| `wrong_memory` | 来自另一篇原始文章的评估文本所生成的同容量 memory + 问题 | 启用 |
| `gold_paragraph` | 答案所在的完整原始段落 + 问题 | 启用 |
| `gold_paragraph_base` | 同一原始段落 + 问题 | 关闭 |

评估使用 greedy decoding，遇 EOS 停止，并设置统一的生成 token 上限。EM/F1 采用 SQuAD 1.1 归一化规则，多参考答案取最高分；NLL 按目标 tokens 加权，包含 EOS。同时记录生成触顶率。

结果区分当前问题与历史问题，逐题保存预测、参考答案、证据到达后的源 token 距离和更新次数。训练完成后，在固定的 test 评估文本上进行最终评估。

每个 micro epoch 记录候选文章数、有效训练文本数、实际使用的样本数、实际长度与压缩率、动态更新次数和 QA 数量；epoch 结束时汇总各 $(K,r)$ 的实际样本占比。Checkpoint 保存模型、optimizer、epoch、micro epoch、样本遍历位置、累计统计及随机状态。恢复训练使用原运行目录和相同的总 epoch 数。

SwanLab 使用 `latent-working-memory-v1` 项目，同一实验的训练与评估共享显式指定的 group。性能测试写入本地日志。

当前 micro epoch 完成后，继续下一个 micro epoch，继承模型和 optimizer 状态。完成本轮全部 $5m$ 个 micro epoch 后，更新目标压缩率占比策略，并重新安排下一轮的容量执行顺序。

## 9. 代码对应

| 文件 | 职责 |
|---|---|
| [data_preparation/squad.py](../../src/latent_working_memory/data_preparation/squad.py) | 原始数据校验、来源划分及 tokenizer 长度记录 |
| [v1/squad.py](../../src/latent_working_memory/v1/squad.py) | 连续段落读取、文本与问题答案对组织、问题抽样 |
| [v1/dynamic_data.py](../../src/latent_working_memory/v1/dynamic_data.py) | 动态配置、文本筛选与配额、micro epoch 调度及固定评估文本 |
| [v1/dynamic.py](../../src/latent_working_memory/v1/dynamic.py) | 记忆初始化、动态训练、梯度传播、多容量 QA 评估和 checkpoint |
| [v1/backbone.py](../../src/latent_working_memory/v1/backbone.py)、[v1/model.py](../../src/latent_working_memory/v1/model.py) | 文本特征提取、位置编码、记忆更新、读取与答案生成 |
| [test_squad_preparation.py](../../tests/v1/test_squad_preparation.py)、[test_dynamic.py](../../tests/v1/test_dynamic.py) | 数据构造、训练梯度、恢复与评估行为测试 |

[默认示例配置](../../configs/v1/dynamic_squad_example.json)采用 m=1、每个 micro epoch 选取 100 条文本、全局 batch B=2。`micro_epochs_per_capacity`、`samples_per_micro_epoch` 和 `batch_size` 分别设置这三个参数。`stage_ends` 与 `ratio_weights` 设置各训练阶段的终点及压缩率占比。

三种梯度传播配置为 [full](../../configs/v1/dynamic_squad/full.json)、[tokens1024](../../configs/v1/dynamic_squad/tokens1024.json) 和 [updates4](../../configs/v1/dynamic_squad/updates4.json)。训练入口为 `.venv/bin/python -m latent_working_memory.v1.dynamic train`，`--epochs` 设置总轮数，`--steps` 可设置计划内的停止步数。评估入口使用 `evaluate`，`--eval-texts-per-ratio` 设置每个容量下各目标压缩率的评估文本数。

输出包括训练日志、`data_plans/` 下的 micro epoch 文本记录与统计、固定 dev/test 评估结果及动态 checkpoint。性能测试入口 [dynamic_profile.py](../../src/latent_working_memory/v1/dynamic_profile.py)通过 `--capacity` 选择要测试的记忆容量。

## 10. 运行环境与入口

动态数据准备、训练、评估和性能测试使用仓库根目录 `.venv/`，依赖由根目录 `pyproject.toml` 与 `uv.lock` 管理。运行前通过 Git 同步仓库，命令在仓库根目录执行。训练 `run.json`、独立评估 `evaluation.json` 和性能测试日志记录 Python 路径、环境目录、源码路径、Git commit 及核心依赖版本。

双卡启动使用 `.venv/bin/python -m torch.distributed.run`。以下为 GPU 6、7 上的性能测试命令，`full` 可替换为 `tokens1024` 或 `updates4`：

```bash
CUDA_VISIBLE_DEVICES=6,7 LWM_ALLOWED_PHYSICAL_GPUS=6,7 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m latent_working_memory.v1.dynamic_profile \
  --checkpoint artifacts/v1/pretrain-data-comparison-2048_20260911/train/pretrain-mixed-157k-20260911/checkpoints/pretrain-step-020000.pt \
  --index data/v1/squad/llama-2-7b-chat_index.json \
  --recipe configs/v1/dynamic_squad/full.json \
  --capacity 1024 \
  --output artifacts/v1/dynamic-env-validation_20260912/plan/profile-full-k1024.jsonl
```

训练入口同样通过上述双卡启动前缀调用 `-m latent_working_memory.v1.dynamic train`，传入 `--checkpoint`、`--index`、`--recipe` 和 `--output-dir`。训练输出使用 `artifacts/v1/<series>/train/<run_name>/`；独立评估使用 `evaluate --split test`，输出到同系列 `eval/<run_name>/`。SwanLab 由 `--swanlab-mode` 控制，测试设为 `disabled`，正式运行设为 `online` 并显式指定 `--swanlab-group`。
