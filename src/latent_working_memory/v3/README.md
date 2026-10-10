# v3：token memory 容量分配对照

比较三种基线与两种动态扩容方法，检验“按信息损失程度扩容”能否在相近容量下改善旧信息保留。两种动态方法均支持四种写入方式，默认使用局部写入。

[运行指南](scripts/README.md) · [实验配置](../../../configs/v3/) · [实验方案](../../../notes/v3/20260926_step1_damage_guided_capacity_experiment.md)

## 一、方法与模型

记忆以向量块保存，一个 slot 对应一个记忆向量。$K$ 默认 512 slots，表示三个基线的总容量和动态方法的首次容量；动态追加容量 $\Delta K$ 默认 32 slots。

| 方法 | 记忆组织 | 默认训练流程 |
|---|---|---|
| `icae_single` | 完整历史一次压缩为 $K$ slots | AE／LM → QA |
| `icae_multi` | 完整历史均分为 n 块，独立压缩后拼接，总容量为 $K$ slots | 多段 AE／LM → 多段 QA |
| `autocompressors` | 新段与累计记忆共同压缩，n 次追加合计 $K$ slots | LM |
| `memory_change` | 覆盖末块或追加 $\Delta K$ slots，按记忆表示变化决定 | AE／LM → 动作 warmup → 策略训练 |
| `information_loss` | 同上的写入方式，按旧信息损失决定 | AE／LM → 动作 warmup → 策略训练 |

五组使用相同的模型设置，各自采用表中的记忆组织和训练目标：

| 共用部分 | 设置 |
|---|---|
| 基座 | Qwen3-4B-Instruct-2507 |
| 写入 | 在输入末尾添加可训练 gist embeddings，取对应位置的最终隐藏状态作为记忆 |
| 可训练参数 | 编码 LoRA（rank 128、alpha 32，作用于 attention／MLP projections）及 gist embeddings；后者使用零均值、标准差 0.02 的高斯初始化 |
| 读取 | 从相同基座复制独立冻结 Decoder |
| 执行 | verl `BaseEngine`、优化器与 DDP；默认使用 BF16、SDPA 和非重入逐层激活重计算 |

## 二、动态写入与扩容规则

### 2.1 四种写入方式

令 $X_t$ 为新文本、$A$ 为最后一个记忆块、$P$ 为更早的记忆。`objective.writer_mode` 指定写入方式，CLI 使用 `--writer-mode`：

| 值 | 追加写入 | 覆盖写入 | 动作区分 |
|---|---|---|---|
| `local`（默认） | 输入 $[X_t,S]$ | 输入 $[A,X_t,S]$ | 通过输入范围区分写入动作 |
| `tag` | 输入 $[P,A,T_{\rm app},X_t,S]$ | 输入 $[P,A,T_{\rm rw},X_t,S]$ | 每种动作使用 m 个独立可训练标记向量 |
| `mask` | 输入 $[P,A,X_t,S]$，$S$ 直接关注 $X_t$ | 输入 $[P,A,X_t,S]$，$S$ 直接关注 $A,X_t$ | 每层限制 gist 位置的 attention，历史仍可经文本表示间接传递 |
| `dual_lora` | 输入 $[P,A,X_t,S]$，使用追加 LoRA | 输入 $[P,A,X_t,S]$，使用覆盖 LoRA | 两套 LoRA 分别训练，共享 gist embeddings 和冻结读取端 |

- **容量与更新**：首段普通压缩为 K slots；随后追加输出 $\Delta K$ slots 并保留全部旧块，覆盖输出与 $A$ 等长的新块并替换 $A$。任务 QA 始终读取全部保存记忆。
- **首次写入**：不使用动作标记或特殊 mask；`dual_lora` 使用追加 LoRA，其余版本使用同一套 LoRA。
- **注意力限制**：`mask` 中 gist 还可关注自身及前序 gist，其余位置保持因果 attention；支持 `eager`／`sdpa`，不支持 `flash_attention_2`。
- **动作标记**：`tag_embeddings` 形状为 $2\times m\times d$，追加／覆盖各用一组 m 个标记向量，分别记为 $T_{\rm app}$、$T_{\rm rw}$（d 为隐藏维度）。m 由 `objective.tag_tokens` 指定，默认 3，可用 `--tag-tokens` 覆盖。这些向量直接传入 `inputs_embeds`，不扩充词表、不占记忆 slots；原词 embedding 与独立 Decoder 保持冻结。
- **参数与初始化**：`local`／`mask` 不因动作增加参数；`tag` 增加 $2md$ 个可训练参数，按均值 0、标准差 0.02 的高斯初始化；`dual_lora` 增加一套可训练 LoRA。每种版本内，追加／覆盖共享 gist embeddings；`local`／`tag`／`mask` 两种动作还共享一套 LoRA。

四种方式描述后训练中的追加／覆盖，所选版本贯穿 `warmup`、`policy` 和评估。共享预训练采用普通单次压缩，见 [3.5 节](#35-两种动态方法共享预训练与轨迹-qa)。

### 2.2 两种扩容规则

| 方法 | 决策过程 |
|---|---|
| `memory_change` | 先生成覆盖候选；对候选与旧末块逐 slot 做 RMS（均方根）归一化，计算相对 Frobenius 距离 $I$。$I >= threshold_i$ 时另生成追加块，否则保存覆盖候选。归一化仅用于评分。 |
| `information_loss` | 分别生成覆盖与追加候选，用固定的历史门控 QA 比较更新前、覆盖后、追加后的可读性。 |

两种门控均为固定规则，不含可训练参数；评分与动作选择不参与反向传播。

`information_loss` 方法所涉及的规则如下：

- 更新前记忆的平均答案 NLL：$L_0$
- 覆盖后记忆的平均答案 NLL：$L_{rw}$
- 追加后记忆的平均答案 NLL：$L_{app}$
- 覆盖造成的损失增加：$d = L_{rw} - L_0$
- 追加相对覆盖的收益：$g = L_{rw} - L_{app}$
- 追加时机：$g > threshold_g$ 或 ($d > threshold_d$ 且 $g > \eta$)
- 评估时也需要门控参考答案，结果标记为 **offline oracle（使用参考答案决策）**。
- 默认阈值尚未校准；质量—容量对照需分别训练不同阈值配置。

## 三、数据与训练

### 3.1 数据输入与通用设置

| 数据输入 | 目录格式 | 用途 |
|---|---|---|
| FineWeb Multisegment | `preparation.json`、`train/dev/test.jsonl`；每行是 `MultisegmentSample` | ICAE 和动态方法的 AE／LM，以及 AutoCompressors 的 LM |
| FineWeb FactQA | `preparation.json`、`train/dev/test.jsonl`；保留原文、字符段界、QA 和 `usage` | ICAE QA、动态方法的动作 warmup 和策略训练及最终评估 |

数据构造见 [FineWeb Multisegment](../data_preparation/fineweb_multisegment/README.md) 和 [FineWeb FactQA](../data_preparation/fineweb_factqa/README.md)。加载时校验各数据集的划分与来源，以及 FactQA 的题池隔离；QA 阶段启动时，将实际选用的 FactQA 来源与 checkpoint 记录的预训练来源对照，检查文档及去重簇是否重叠。最终评估再次检查来源隔离。

- **FineWeb Multisegment**：使用完整正文，训练时超出 `training.max_input_tokens` 的样本整条过滤。默认 12288，`null` 使用模型窗口。每次实际编码／解码另行检查窗口长度。
- **FineWeb FactQA**：使用完整正文，训练时超出 `training.max_qa_input_tokens` 的样本整条过滤。默认 12288，`null` 使用模型窗口。原始段界用于确定题目来源及 old／new 分类；各方法的压缩分段见下文。
- **QA 损失**：先平均每题答案 token 的 NLL，再按实际题数平均；任务题与门控题分离。
- **数据管线**：复用 verl 的 collator 和其 SFT 流程使用的 StatefulDataLoader；保持原随机顺序与完整 global batch，尾批按真实样本数处理，加载器状态负责恢复读取进度。

### 3.2 ICAE-single：完整正文单次压缩

| 阶段 | 数据使用与训练目标 |
|---|---|
| `pretrain` | FineWeb Multisegment 完整正文一次压缩为 K slots，读取记忆计算 AE 或 LM 损失 |
| `qa` | FineWeb FactQA 完整正文一次压缩为 K slots，读取记忆计算全部训练任务题 |

- 对于 FineWeb Multisegment，每条样本的训练目标（LM 或 AE）通过 `training.lm_ratio` 进行选择（默认概率 0.5）；LM 取 `training.lm_target_tokens` 个续文 tokens（默认 512），不足时改用 AE；选择由 seed 和样本 ID 决定，跨 epoch 保持不变。

### 3.3 ICAE-multi：分块压缩，总容量固定

| 阶段 | 数据使用与训练目标 |
|---|---|
| `pretrain` | FineWeb Multisegment 完整正文分段独立压缩，最后拼接为 K slots，读取该记忆计算 AE 或 LM 损失 |
| `qa` | FineWeb FactQA 完整正文分段独立压缩，最后拼接为 K slots，读取该记忆计算全部训练任务题 |

- 分段压缩逻辑：
    1. **选择段数**：由 `objective.icae_min_segments`／`icae_max_segments` 指定范围（默认 3–6），从中均匀采样 n。采样由 seed 和样本 ID 决定，同一样本跨 epoch 使用相同的 n。
    2. **切分文本**：将正文均分为 n 个连续段，不沿用数据原始段界。
    3. **分配容量**：将 K 个 slots 均分为 n 块。例如 K=512、n=3 时，各块为 **171／171／170 slots**。
- 各记忆块共用压缩器与 gist embeddings，按所需容量使用 embedding 表的前若干行；输出记忆分别生成。因此，embedding 表只需 `ceil(K / icae_min_segments)` 行，容纳单块的最大容量。
- Fineweb FactQA 评估时将完整正文均分为 n 段，不沿用原始段界。

### 3.4 AutoCompressors：顺序追加与分段 LM

| 阶段 | 数据使用与训练目标 |
|---|---|
| `pretrain` | FineWeb Multisegment 完整正文分段顺序压缩为 K slots，计算 LM 损失 |

- 分段压缩逻辑：
    1. **确定段数**：由 `objective.ac_num_segments` 指定正文压缩段数 n（默认 4）；续文不计入段数。
    2. **切分文本**：`objective.bptt_steps` 为正整数 B 时（默认 2），训练按 B 段一组：先将正文均分为 n 段以确定各组的 token 总量，再在组内随机重划段界；尾组取剩余段数。段长为组内平均段长的约 2/3–4/3，采样由 seed、样本 ID 和 epoch 决定。
    3. **分配容量与追加**：将 K 个 slots 均分为 n 块。每段文本结合累计记忆压缩为新块并追加，旧块保持不变；n 次追加后总容量为 K。例如 K=512、n=4 时，每次追加 **128 slots**。
- 各记忆块共用压缩器与 gist embeddings，按所需容量使用 embedding 表的前若干行；embedding 表只需 `ceil(K / n)` 行。
- **LM 监督**：逐段用此前的累计记忆与当前段已有文本计算 next-token LM，随后压缩当前段并追加记忆；第一段不读取历史记忆，也计算 LM 损失。正文处理完后，以完整 K 记忆预测独立续文；续文最多取 `training.lm_target_tokens` 个 tokens（默认 512），不参与压缩。两部分损失按实际目标 token 数合并平均。
- **梯度截断**：以 B 次写入为一个 BPTT 窗口，完成后继文本监督后再 detach（切断梯度传播）记忆；尾窗可短于 B，无整除要求。`bptt_steps=null` 表示完整 BPTT，此时全部 n 段组成一个随机切分组，记忆不截断梯度。
- Fineweb FactQA 评估时将完整正文均分为 n 段，不沿用原始段界。

### 3.5 两种动态方法：共享预训练与轨迹 QA

`memory_change` 与 `information_loss` 使用相同的数据与训练流程，仅 `policy` 阶段的扩容规则不同：

共享预训练的 `objective.method` 为 `dynamic`，仅运行 `pretrain`；后训练分别使用两种方法的名称。

共享预训练输入为 $[X,S_K]$，将完整正文一次压缩为 K slots，不输入历史记忆或追加／覆盖动作。AE／LM 损失只训练一套 LoRA 和 gist embeddings，使记忆能被冻结 Decoder 使用；它为四种更新方式提供共同基础，各自的写入行为在后训练中学习。

| 阶段 | 数据与记忆更新 | 两组关系 |
|---|---|---|
| `pretrain` | FineWeb Multisegment 完整正文一次压缩为 K slots，计算 AE 或 LM 损失 | 只训练一次，复用同一 checkpoint |
| `warmup` | FineWeb FactQA 原始段界逐段压缩；首次写入 K slots，后续按 `objective.append_probability`（默认 0.5）选择追加或覆盖，计算任务 QA 损失 | 分别训练 |
| `policy` | FineWeb FactQA 原始段界逐段压缩；首次写入 K slots，后续按各自固定规则追加或覆盖，沿选中路径计算任务 QA 损失，训练写入器的可训练参数 | 分别训练 |

- 每个更新点读取全部已保存记忆，使用该点配置的 old／new 任务题池，按实际题数合并损失；再平均更新点，最终按轨迹平均。信息损失方法额外用固定门控 QA 决策，这些题不参与计算任务损失。
- QA 阶段默认完整 BPTT，可用 `objective.bptt_steps` 设置截断窗口；例如设为 2 时每两轮截断，首次写入也计一轮。每个窗口反向并 detach 记忆，global batch 结束后更新一次参数。
- 四种写入版本均可复用同一共享预训练 checkpoint。`tag` 继承 LoRA／gist，按 m 新初始化两组标记向量，随后与它们一起进行 warmup／policy QA 训练；阶段衔接、恢复与评估沿用保存的 m。`dual_lora` 复制两套 LoRA，分别接收对应动作路径的梯度。

### 3.6 训练预算与批处理

预设的 `training.stage_max_train_samples` 按阶段指定样本上限，过滤后以 seed 可复现地选取；`null` 表示不限：

| 阶段 | 默认训练样本上限 |
|---|---:|
| `pretrain`（FineWeb Multisegment） | 12800 |
| ICAE `qa`、动态方法 `warmup` 和 `policy`（Fineweb FactQA） | 全部 |

各阶段可独立通过 CLI 覆盖；`smoke`／`pilot` 另受试跑档位上限约束，取较小值。

| 阶段 | 默认每卡 microbatch | 默认梯度累积 |
|---|---:|---:|
| ICAE 和 AutoCompressors 完整流程 | 8 | 1 |
| 动态方法完整流程| 4 | 2 |

全局 batch = GPU 数 × 每卡 microbatch × 梯度累积；双卡默认均为 **16**，尾批按实际样本数平均。

## 四、运行

### 4.1 启动与配置

按[运行指南](scripts/README.md#1-准备环境模型与数据)准备环境、模型、数据和 SwanLab key，并设置 `run_gpu.sh` 中的仓库路径。在仓库根目录执行，例如运行 ICAE-single：

```bash
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
    --method icae_single \
    --gpus 0,1
```

| 参数 | 用法 |
|---|---|
| `--method` | 默认 `all`，运行五种方法；或指定运行单方法 |
| `--mode` | 默认 `full`；试跑指定 `smoke` 或 `pilot` |
| `--config <文件>` | 替换所选单方法的默认配置；省略时读取 `configs/v3/` |
| `--writer-mode` | 动态后训练写入方式：`local`（默认）、`tag`、`mask`、`dual_lora` |
| `--tag-tokens` | 每种动作的控制 token 数 m，默认 3；仅用于动态 `tag` 写入 |

- **配置覆盖**：`objective.stages` 声明阶段顺序，显式 CLI 超参数覆盖配置值；更多参数见[运行指南](scripts/README.md#4-调整参数)。
- **阶段执行**：同一方法的连续阶段共用模型和训练进程，切换时重置优化器；不同方法分别启动。

### 4.2 复用预训练与中断恢复

- **共享预训练**：用 `--method dynamic_pretrain` 单独训练一次；随后分别运行 `memory_change` 和 `information_loss`，通过 `--init-checkpoint` 指向同一 checkpoint 目录。命令见[复用指南](scripts/README.md#3-复用共享预训练可选)。
- **来源记录**：后训练自动沿用预训练的实验系列编号 `run-id`。保留来源目录中的 `pretrain/run.json`，以及在线运行生成的 `swanlab.json`。
- **已有产物**：保留旧预训练的原路径与配置，可继续通过 `--init-checkpoint` 使用，不直接重命名旧目录。
- **中断恢复**：使用阶段目录保存的 `config.json`，调用训练模块的 `--resume` 并传入 checkpoint 目录。

## 五、评估与产物

### 5.1 评估方式与指标

- **评估对象**：默认使用最后一个 checkpoint，在同一批 FineWeb FactQA 轨迹的最终记忆上回答评估题；门控题不计入质量指标。
- **质量**：记录答案 NLL、EM（精确匹配）和 F1，生成使用 greedy decoding。按 all／old／new 及段距离统计；new 指末段事实，old 指更早事实，段距离按 FactQA 原始段界计算。
- **容量**：记录最终 slots 与各更新点保存状态的平均 slots，再按轨迹平均；ICAE-single 只有一次压缩状态。
- **成本**：记忆构建耗时包含候选写入及门控计算，不含最终 QA 读取与答案生成。

### 5.2 本地产物

```text
artifacts/v3/<run-id>/
├── plan/<method-dir>/           阶段配置、job.json、日志、result.json
├── train/<method-dir>/<stage>/  config.json、run.json、metrics.jsonl、checkpoints/
├── eval/<method-dir>/<stage>/   summary.json、trajectories.jsonl
└── compare/                     points.json、points.csv、compare.log、result.json
```

- **运行目录**：`run-id` 默认是上海时间 `YYYYMMDD-HHMMSS`，可用 `--run-id` 指定；父目录由 `--output-root` 控制。
- **方法目录**：三个 baseline 和动态预训练使用 `<method>-k<K>`；动态后训练统一使用 `<method>-k<K>+<ΔK>-<writer>`，`<writer>` 为 `local`、`tag`、`mask` 或 `dual-lora`，试跑最后添加 `_<mode>`。例如 `memory-change-k512+32-local`、`memory-change-k512+32-tag_smoke`。方法名中的 `_` 转为 `-`，共享预训练用 `dynamic-pretrain`；K、ΔK 分别来自 `model.memory_slots`、`objective.append_slots`。
- **Checkpoint**：目录为 `checkpoints/global_step_<阶段step>/`。`state.pt` 保存新增权重、优化器及运行状态，`data_<rank>.pt` 保存各卡加载器状态；每阶段默认保留最近两个，初始化、恢复和评估均传入该目录。外层结构及文件布局不变；原 adapter 字典中，`tag` 保存 $2\times m\times d$ 的 `tag_embeddings`，`dual_lora` 保存两套 LoRA。恢复与评估使用保存的写入方式和 m。
- **结果与比较**：`summary.json` 汇总指标，`trajectories.jsonl` 保存动作、分数、计时与逐题结果。本次调用有至少两个评估结果时生成 `compare/` 产物。

### 5.3 SwanLab 组织与展示

- **组织**：默认 project 为 `latent-working-memory-v3`，group 为 `run-id`。同一方法的训练、验证和最终评估共用一个 run；从头运行 `all` 共六个 run，包含独立的共享预训练。
- **名称**：使用 `<method-dir>_<run-id>`；两种动态方法与共享预训练保持相同后缀，关联共同来源，各自仍有独立的云端 ID。
- **横轴**：训练／验证使用累计 optimizer step；动态后训练的 step 不包含共享预训练。
- **曲线**：展示目标损失、旧／新 QA NLL、容量、梯度范数、耗时和训练步峰值显存。
- **追加比例**：动态 warmup／policy 将 `train/append_ratio` 与 `dev/append_ratio` 合并为一张原生折线图，横轴为累计 optimizer step。比例为追加次数／（追加次数＋覆盖次数），不计首次写入；仅记录有决策的训练步与实际验证点。
- **阶段与进度**：`train/stage` 中，1 = pretrain、2 = ICAE QA／动态方法 warmup、3 = 动态方法 policy；`train/epoch` 以小数记录当前阶段累计处理样本数／训练集样本数，阶段切换时重新计数。
- **最终柱状图**：NLL／EM／F1 各合并 all／old／new；另展示最终与平均 slots、每条轨迹的记忆构建耗时。
- **完整记录**：统计与样例保存在本地，不上传表格；`--tracking disabled` 仅记录本地结果。

## 六、代码入口与验证

| 职责 | 文件 |
|---|---|
| 配置与运行计划 | [config.py](config.py)、[job_plan.py](job_plan.py) |
| 启动与调度 | [gpu_job.py](gpu_job.py)、[train.py](train.py) |
| 数据加载 | [data.py](data.py)、[pretrain_data.py](pretrain_data.py) |
| 模型、写入与目标 | [model.py](model.py)、[objective.py](objective.py) |
| 训练、恢复与记录 | [engine.py](engine.py)、[runtime.py](runtime.py)、[tracking.py](tracking.py) |
| 评估与比较 | [evaluate.py](evaluate.py)、[compare.py](compare.py) |

```bash
uv run --frozen pytest -q tests/v3
uv run --frozen ruff check src/latent_working_memory/v3 tests/v3
```

测试覆盖数据契约、可微冻结读取、动作分支、门控隔离、截断 BPTT 和 CPU DDP；真实模型的质量、显存与吞吐由 GPU 实验测量。
