# v3：token memory 容量分配对照

比较三种基线与两种动态扩容方法，检验“按信息损失程度扩容”能否在相近容量下改善旧信息保留。两种动态方法均支持四种写入方式，默认使用局部写入。

[运行指南](scripts/README.md) · [实验配置](../../../configs/v3/) · [实验方案](../../../notes/v3/20260926_step1_damage_guided_capacity_experiment.md)

## 一、方法与模型

记忆以向量块保存，一个 slot 对应一个记忆向量。$K$ 默认 512 slots，表示三个基线的总容量和动态方法的首次容量；动态追加容量 $\Delta K$ 默认 32 slots。

| 方法 | 记忆组织 | 默认训练流程 |
|---|---|---|
| `icae_single` | 完整历史一次压缩为 $K$ slots | AE／LM → QA |
| `icae_multi` | 完整历史均分为 n 块，独立压缩后拼接，总容量为 $K$ slots | 多段 AE／LM → 多段 QA |
| `autocompressors` | 新段与累计记忆共同压缩，n 次追加合计 $K$ slots | 多段 LM |
| `memory_change` | 覆盖末块或追加 $\Delta K$ slots，按记忆表示变化决定 | 复用 AE／LM → warmup → policy |
| `information_loss` | 同上的写入方式，按旧信息损失决定 | 复用 AE／LM → warmup → policy |

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
| `information_loss` | 分别生成覆盖与追加候选，用同一组历史门控 QA 比较更新前、覆盖后、追加后的可读性。 |

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
| FineWeb Multisegment | `train/dev/test.jsonl`、`preparation.json` | 所有方法的预训练 |
| FineWeb FactQA | `train/dev/test.jsonl`、`preparation.json` | ICAE 和动态方法的后训练以及所有方法的评估 |

数据构造见 [FineWeb Multisegment](../data_preparation/fineweb_multisegment/README.md) 和 [FineWeb FactQA](../data_preparation/fineweb_factqa/README.md)。

通用设置：

- **数据校验**：加载时检查数据集的划分与来源，以及 FineWeb FactQA 的 task／gate 题池隔离。
- **长度限制**：两类数据均使用完整正文。Multisegment 和 FactQA 分别通过 `training.max_input_tokens` 与 `training.max_qa_input_tokens` 筛选 train 正文，默认均为 12288；正整数仅按给定值筛选，超限样本整条排除，`null` 跳过长度上限筛选。dev/test 不按该上限过滤。
- **QA 损失**：先平均每题答案 token 的 NLL，再按实际题数平均。

FineWeb FactQA 数据按如下流程使用：

1. **原始数据**：`qas` 保存全部问答及其所属段、类型（`task` 或 `gate`）等信息。
2. **运行前构建题池**：对每条数据的每段文本，按 QA 所属段和类型构建 task／gate 题池，并检查其是否达到后续抽样所需的数量，不足则排除整条轨迹。任一 split 保留不足原规模的 80% 时报错。通过后再进行正文长度上限过滤（仅对 train）以及样本数量上限过滤。实际编码／解码须满足模型窗口限制。
3. **运行中抽样与使用**：新 task QA 来自当前段，旧 task QA 和 gate QA 来自全部历史段。
   - **task QA**（用于 QA 损失计算）：
        - train 进行抽样，抽样总题数由 `task_qa_count`（默认 8）指定，新题比例由 `task_new_ratio`（默认 0.5）指定，乘积须为正整数。注意，首段仅使用新题。
        - dev/test 不进行抽样，而是使用全部可见题目：baselines 均在全文压缩后评估，动态方法还逐段评估可见题目，见 [5.1 节](#51-评估方式与指标)。
   - **gate QA**（用于信息损失门控计算）：train/dev/test 均进行抽样，抽样总题数由 `gate_qa_count`（默认 8）指定。注意，首段不执行门控。
   - **抽样规则**：
        - 动态方法在每次压缩新段时进行抽样，baselines 则在压缩完整全文后进行抽样。
        - 训练时 task QA 和 gate QA 按 `objective.seed`、epoch 可复现地重采样，允许跨 epoch 不同；dev gate 使用 `objective.seed`，test gate 使用评估 seed，均固定选题。

### 3.2 ICAE-single：完整正文单次压缩

| 阶段 | 数据使用与训练目标 |
|---|---|
| `pretrain` | FineWeb Multisegment 完整正文一次压缩为 K slots，读取记忆计算 AE 或 LM 损失 |
| `qa` | FineWeb FactQA 完整正文一次压缩为 K slots，以末段为新题来源，按训练题数和比例计算任务损失 |

- 对于 FineWeb Multisegment，每条样本的训练目标（LM 或 AE）通过 `training.lm_ratio` 进行选择（默认概率 0.5）；LM 取 `training.lm_target_tokens` 个续文 tokens（默认 512），不足时改用 AE；选择由 `training.seed` 和样本 ID 决定，跨 epoch 保持不变。

### 3.3 ICAE-multi：分块压缩，总容量固定

| 阶段 | 数据使用与训练目标 |
|---|---|
| `pretrain` | FineWeb Multisegment 完整正文分段独立压缩，最后拼接为 K slots，读取该记忆计算 AE 或 LM 损失 |
| `qa` | FineWeb FactQA 完整正文分段独立压缩，最后拼接为 K slots，以末段为新题来源，按训练题数和比例计算任务损失 |

- 分段压缩逻辑：
    1. **选择段数**：由 `objective.icae_min_segments`／`icae_max_segments` 指定范围（默认 3–6），从中均匀采样 n。采样由 `objective.seed` 和样本 ID 决定，同一样本跨 epoch 使用相同的 n；测试沿用 checkpoint 中的该 seed。
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
    2. **切分文本**：`objective.bptt_steps` 为正整数 B 时（默认 2），训练按 B 段一组：先将正文均分为 n 段以确定各组的 token 总量，再在组内随机重划段界；尾组取剩余段数。段长为组内平均段长的约 2/3–4/3，采样由 `objective.seed`、样本 ID 和 epoch 决定。
    3. **分配容量与追加**：将 K 个 slots 均分为 n 块。每段文本结合累计记忆压缩为新块并追加，旧块保持不变；n 次追加后总容量为 K。例如 K=512、n=4 时，每次追加 **128 slots**。
- 各记忆块共用压缩器与 gist embeddings，按所需容量使用 embedding 表的前若干行；embedding 表只需 `ceil(K / n)` 行。
- **LM 监督**：逐段用此前的累计记忆与当前段已有文本计算 next-token LM，随后压缩当前段并追加记忆；第一段不读取历史记忆，也计算 LM 损失。正文处理完后，以完整 K 记忆预测独立续文；续文最多取 `training.lm_target_tokens` 个 tokens（默认 512），不参与压缩。两部分损失按实际目标 token 数合并平均。
- **梯度截断**：以 B 次写入为一个 BPTT 窗口，完成后继文本监督后再 detach（切断梯度传播）记忆；尾窗可短于 B，无整除要求。`bptt_steps=null` 表示完整 BPTT，此时全部 n 段组成一个随机切分组，记忆不截断梯度。
- Fineweb FactQA 评估时将完整正文均分为 n 段，不沿用原始段界。

### 3.5 两种动态方法：共享预训练与轨迹 QA

`memory_change` 与 `information_loss` 使用相同的数据与训练流程，仅 `policy` 阶段的扩容规则不同：

| 阶段 | 数据与记忆更新 | 两组关系 |
|---|---|---|
| `pretrain` | FineWeb Multisegment 完整正文一次压缩为 K slots，计算 AE 或 LM 损失 | 默认单独训练一次并复用；同样支持和后训练一次性完整运行 |
| `warmup` | FineWeb FactQA 原始段界逐段压缩；首次写入 K slots，后续按 `objective.append_probability`（默认 0.5）选择追加或覆盖，计算任务 QA 损失 | 分别训练 |
| `policy` | FineWeb FactQA 原始段界逐段压缩；首次写入 K slots，后续按各自固定规则追加或覆盖，沿选中路径计算任务 QA 损失，训练写入器的可训练参数 | 分别训练 |

- 默认动态方法配置只运行后训练 `warmup → policy`，通过 `--checkpoint` 复用预训练，同样支持在自定义配置中显式声明 `pretrain → warmup → policy`，运行完整训练流程。
- 动态预训练输入为 $[X,S_K]$，将完整正文一次压缩为 K slots，不输入历史记忆或追加／覆盖动作。AE／LM 损失只训练一套 LoRA 和 gist embeddings，使记忆能被冻结 Decoder 使用。
- 计算损失函数时，对当前 batch 中的每个样本，在每个更新点读取全部已保存记忆，使用运行时选出的 old／new task QA 计算并合并损失，再按更新点平均，最终按样本平均。`information_loss` 额外使用 gate QA 进行记忆追加 / 覆盖决策，其不参与任务损失。
- QA 阶段默认完整 BPTT，可用 `objective.bptt_steps` 设置截断窗口；例如设为 2 时每两轮截断，首次写入也计一轮。
- 四种写入版本均可复用同一共享预训练 checkpoint。`tag` 继承 LoRA／gist，按 m 新初始化两组标记向量，随后与它们一起进行 warmup／policy QA 训练；阶段衔接、恢复与评估沿用保存的 m。`dual_lora` 复制两套 LoRA，分别接收对应动作路径的梯度。

### 3.6 训练预算与批处理

| 阶段 | 默认训练样本上限 |
|---|:---:|
| 所有方法的预训练（FineWeb Multisegment） | 12800 |
| 所有方法的后训练（Fineweb FactQA） | null（全部） |

- `training.stage_max_train_samples` 按阶段指定样本上限，过滤后以 `training.seed` 可复现地选取。
- 全局 batch = GPU 数（默认 2）× 每卡 microbatch（默认 4）× 梯度累积（默认 2）。双卡默认均为 **16**，尾批按实际样本数平均。

### 3.7 Checkpoint

- `--save_every` 支持正整数 optimizer step 间隔或 `"after_each_epoch"`。
- `--save_total_limit` 限制全部 checkpoint 的数量。

## 四、运行

按[运行指南](scripts/README.md#1-准备环境模型与数据)准备环境、模型和数据，并在项目 `.env` 中配置 SwanLab key。在仓库根目录执行，例如运行 ICAE-single：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
    --method icae_single --gpus 0,1
```

基本参数如下所示，显式 CLI 超参数会覆盖配置文件的值，更多参数见[运行指南](scripts/README.md#4-调整参数)。

| 参数 | 用法 |
|---|---|
| `--method` | 方法名，普通训练必须显式指定，断点恢复与单独评估可省略 |
| `--mode` | 运行模式，默认 `full`，支持试跑 `smoke` |
| `--config` | 配置文件地址，默认读取 `configs/v3/` 中的默认配置 |
| `--gpus` | GPU 序号 |

## 五、评估与产物

### 5.1 评估方式与指标

**评估场景**

| 场景 | 数据与评估对象 | 指标 |
|---|---|---|
| AE／LM 训练中验证 | FineWeb Multisegment dev，验证当前阶段的重建或续写文本 | AE／LM NLL |
| QA 训练中验证 | FineWeb FactQA dev，评估全部可见 task QA | NLL |
| 最终评估 | FineWeb FactQA test，评估全部可见 task QA | NLL、EM、F1 |

**QA 评估流程**

- **三个 baseline**：一次性或分段完成全文压缩后，统一在最终记忆上评估全部 task QA。
- **动态方法**：每次写入后计算全部已读 task QA 的 NLL，追踪同一道题随压缩过程的变化。最终评估时，额外在全文压缩后的记忆上计算 EM 和 F1。

图表展示见 [5.3 节](#53-swanlab-组织与展示)。

### 5.2 本地产物

```text
artifacts/v3/<run-id>/
├── plan/<method-dir>/           阶段配置、eval.json、job.json、日志、result.json
├── train/<method-dir>/<stage>/  config.json、run.json、metrics.jsonl、dev-qa.jsonl、checkpoints/
├── eval/<method-dir>/<stage>/   summary.json、trajectories.jsonl
└── compare/                     独立比较生成的 points.json、points.csv
```

- **运行目录**：`run-id` 可用 `--run-id` 指定，默认使用格式为 `YYYYMMDD-HHMMSS` 的当前时间，复用动态预训练且省略 `--run-id` 时继承 checkpoint 来源的编号。父目录由 `--output-root` 控制。
- **方法目录**：三个 baseline 和独立动态预训练使用 `<method>-k<K>{_<mode>}`，两种动态方法统一使用 `<method>-k<K>+<ΔK>-<writer>{_<mode>}`。例如，正式运行 `memory-change-k512+32-local`、试跑 `memory-change-k512+32-tag_smoke`。
- **Checkpoint**：目录为 `checkpoints/global_step_<阶段step>/`。`state.pt` 保存新增权重、优化器及运行状态，`data_<rank>.pt` 保存各卡加载器状态。
- **结果与比较**：`summary.json` 汇总指标，`trajectories.jsonl` 保存动作、分数、计时与逐题结果。比较单独调用 [compare.py](compare.py)，导出 `points.json` 与 `points.csv`；命令见[运行指南](scripts/README.md#5-查看结果)。比较要求 split、实际 task 题池、各段选中的 gate QA 签名、QA prompt 和生成 token 上限一致；seed 不同但实际选题相同仍可比较，各方法的训练数据与训练配置可不同。

### 5.3 SwanLab 组织与展示

**运行组织**

- 默认 project 为 `latent-working-memory-v3`，run 名称为 `<method-dir>_<run-id>`；相关 runs 通过 `--group` 关联，未指定时采用当前 `run-id`。
- 同一方法的连续训练、dev 验证和最终 test 评估共用一个 run，分别展示在 `train/*`、`dev/*`、`evaluation/*`。独立共享预训练使用自己的 run，来源通过 `pretraining` 配置追溯。

**训练与验证**

增量曲线以本次连续训练累计的 optimizer step 为横轴，只记录实际计算的指标；复用外部预训练时，不计来源运行的步数。

- **通用指标**：train/dev 展示目标损失和最终 slots；训练另记录梯度范数、每步耗时与峰值显存，后两项位于 `resources/*`。
- **QA dev**：ICAE 的 QA 阶段按问题首次引入的原始段编号及总体均值展示 NLL 柱状图，每次验证更新，按题平均；整体 dev loss 曲线仍按轨迹平均。动态方法展示旧／新 QA NLL 曲线。
- **追加比例**：动态 warmup／policy 将 train/dev 的追加比例合并为一张曲线图。比例为追加次数／（追加次数＋覆盖次数），不计首次写入，仅记录有决策的训练步和实际验证点。
- **阶段与进度**：`train/stage` 的 1、2、3 分别表示预训练、ICAE QA／动态 warmup、动态 policy；`train/epoch` 表示当前阶段已遍历的训练集轮数，支持小数，阶段切换时重新计数。

**最终评估**

最终 test 结果展示在对应方法的 run 中：

| 内容 | 展示方式 |
|---|---|
| NLL | 柱状图，横轴为 QA 首次引入的原始段编号及总体均值 |
| EM／F1 | 合并为双纵轴分组柱状图，横轴同上，两轴范围均为 `[0,1]` |
| 动态方法的过程保持 | “引入段 × 已读段” NLL 热图，以及按引入后经过的段数对齐的相对首次 NLL 变化曲线 |
| 容量与成本 | 最终／平均 slots 柱状图，以及每条轨迹平均记忆构建耗时柱状图 |

## 六、代码入口与验证

| 职责 | 文件 |
|---|---|
| 配置与运行计划 | [config.py](config.py)、[job_plan.py](job_plan.py) |
| 启动与调度 | [gpu_job.py](gpu_job.py)、[train.py](train.py) |
| 数据加载与选题 | [data.py](data.py)、[pretrain_data.py](pretrain_data.py)、[qa_selection.py](qa_selection.py) |
| 模型、写入与目标 | [model.py](model.py)、[objective.py](objective.py) |
| 训练、恢复与记录 | [engine.py](engine.py)、[runtime.py](runtime.py)、[tracking.py](tracking.py) |
| 评估与比较 | [evaluate.py](evaluate.py)、[compare.py](compare.py) |

```bash
uv run --frozen pytest -q tests/v3
uv run --frozen ruff check src/latent_working_memory/v3 tests/v3
```
