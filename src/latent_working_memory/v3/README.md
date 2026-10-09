# v3：token memory 容量分配对照

比较三种基线与两种动态扩容方法，检验“按信息损失程度扩容”能否在相近容量下改善旧信息保留。当前动态方法采用局部写入：只在覆盖时将最后一个记忆块传入编码器。

[运行指南](scripts/README.md) · [实验配置](../../../configs/v3/) · [实验方案](../../../notes/v3/20260926_step1_damage_guided_capacity_experiment.md)

## 一、方法与模型

记忆以向量块保存，一个 slot 对应一个记忆向量。$K$ 默认 512 slots，表示三个基线的总容量和动态方法的首次容量；动态追加容量 $\Delta K$ 默认 32 slots。

| 方法 | 记忆组织 | 默认训练流程 |
|---|---|---|
| `icae_single` | 完整历史一次压缩为 $K$ slots | AE／LM → QA |
| `icae_multi` | 完整历史均分为 n 块，独立压缩后拼接，总容量为 $K$ slots | 多段 AE／LM → 多段 QA |
| `autocompressors` | 新段与累计记忆共同压缩，n 次追加合计 $K$ slots | 多段 LM |
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

## 二、两种扩容规则

令 $X_t$ 为新文本、$A$ 为最后一个记忆块：

- **追加**：输入 $X_t$，输出 $\Delta K$ slots 的新块；所有旧块保持不变。
- **覆盖**：输入 $[A, X_t]$，输出与 $A$ 等长的新块，替换 $A$。
- 更早的块不参与写入；任务 QA 读取全部已保存的块。两种动作共享 LoRA 和 gist embeddings。

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
| `lm` | FineWeb Multisegment 完整正文分段顺序压缩为 K slots，逐段计算正文的 LM 损失，最后以完整记忆预测独立续文 |

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

| 阶段 | 数据与记忆更新 | 两组关系 |
|---|---|---|
| `pretrain` | FineWeb Multisegment 完整正文一次压缩为 K slots，计算 AE 或 LM 损失 | 只训练一次，复用同一 checkpoint |
| `warmup` | FineWeb FactQA 原始段界逐段压缩；首次写入 K slots，后续按 `objective.append_probability`（默认 0.5）选择追加或覆盖，计算任务 QA 损失 | 分别训练 |
| `policy` | FineWeb FactQA 原始段界逐段压缩；首次写入 K slots，后续按各自固定规则追加或覆盖，沿选中路径计算任务 QA 损失，训练写入器的编码 LoRA 和 gist embeddings | 分别训练 |

- 每个更新点读取全部已保存记忆，使用该点配置的 old／new 任务题池，按实际题数合并损失；再平均更新点，最终按轨迹平均。信息损失方法额外用固定门控 QA 决策，这些题不参与计算任务损失。
- QA 阶段默认完整 BPTT，可用 `objective.bptt_steps` 设置截断窗口；例如设为 2 时每两轮截断，首次写入也计一轮。每个窗口反向并 detach 记忆，global batch 结束后更新一次参数。

### 3.6 训练预算与批处理

预设的 `training.stage_max_train_samples` 按阶段指定样本上限，过滤后以 seed 可复现地选取；`null` 表示不限：

| 阶段 | 默认训练样本上限 |
|---|---:|
| ICAE／动态方法 `pretrain`、AutoCompressors `lm`（FineWeb Multisegment） | 12800 |
| ICAE `qa`、动态方法 `warmup` 和 `policy`（Fineweb FactQA） | 全部 |

各阶段可独立通过 CLI 覆盖；`smoke`／`pilot` 另受试跑档位上限约束，取较小值。

| 阶段 | 默认每卡 microbatch | 默认梯度累积 |
|---|---:|---:|
| ICAE 和 AutoCompressors 完整流程、动态方法 `pretrain` | 8 | 1 |
| 动态方法 `warmup` 和 `policy`| 4 | 2 |

全局 batch = GPU 数 × 每卡 microbatch × 梯度累积；双卡默认均为 **16**，尾批按实际样本数平均。

## 四、运行

先按[运行指南](scripts/README.md#1-准备环境模型与数据)准备环境、模型、数据和 SwanLab key，并修改 `run_gpu.sh` 中的仓库路径。以下命令在仓库根目录执行：

```bash
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
    --mode full \
    --gpus 0,1
```

- `--method` 默认 `all`；也可选择表中的单个方法。`smoke`、`pilot`、`full` 控制运行预算。
- 默认配置位于 `configs/v3/`，`objective.stages` 声明阶段顺序；`--method <方法> --config <文件>` 可替换单方法配置，显式 CLI 超参数优先。参数及作用范围见[运行指南](scripts/README.md#4-调整参数)。
- 同一方法的连续阶段复用模型和训练进程，切换阶段时重置优化器；共享预训练与不同方法分别启动。
- 单独准备共享预训练使用 `--method dynamic_pretrain`。后续用 `--method memory_change` 或 `information_loss` 搭配同一个 `--init-checkpoint`，默认连续执行 warmup → policy → 评估，并继承预训练的 `run-id`。保留来源运行目录及身份文件。
- 中断恢复使用阶段目录保存的 `config.json` 和 checkpoint，调用训练模块的 `--resume`。

## 五、评估与产物

默认评估最后一个 checkpoint，在同一批 FineWeb FactQA 轨迹的最终记忆上回答评估题，门控题不计入质量指标。记录答案 NLL、EM（精确匹配）和 F1，并按末段新事实、更早旧事实及段距离分层；生成使用 greedy decoding。

```text
artifacts/v3/<run-id>/
├── plan/<method-dir>/
├       阶段配置、job.json、日志、result.json
├── train/<method-dir>/<stage>/
├       config.json、run.json、metrics.jsonl、checkpoints/
├── eval/<method-dir>/<stage>/
├       summary.json、trajectories.jsonl
└── compare/
        points.json、points.csv、compare.log、result.json
```

`<method-dir>` 正式运行使用 `<method>-k<K>`，试跑添加 `_<mode>`；方法名中的 `_` 转为 `-`，共享预训练使用 `dynamic-pretrain`，K 对应 `model.memory_slots`，追加容量 ΔK 单独记录在配置中。`run-id` 默认是上海时间 `YYYYMMDD-HHMMSS`，可用 `--run-id` 指定；父目录可用 `--output-root` 修改。

- **本地结果**：`summary.json` 汇总质量、容量与读写成本，`trajectories.jsonl` 保存动作、分数、计时及逐题结果。本次调用有至少两个评估结果时生成比较，不扫描既有运行。
- **SwanLab 组织**：默认 project 为 `latent-working-memory-v3`，group 为 `run-id`。每个方法的训练、验证与最终评估共用一个 run；共享预训练独立，因此默认 `all` 共六个 run。
- **运行名称**：`<method-dir>_<run-id>`。两种动态方法与共享预训练保持同一后缀，关联共同来源；连续阶段使用累计 optimizer step，动态后训练步数不包含共享预训练。

| SwanLab 展示 | 内容 |
|---|---|
| 训练／验证曲线 | 目标损失、旧／新 QA NLL、容量、梯度范数、阶段、整步耗时和峰值显存 |
| 阶段 `train/stage` | 1 = 预训练／LM，2 = ICAE QA／动态动作预热，3 = 动态策略训练 |
| 最终评估柱状图 | NLL／EM／F1（各合并 all／old／new）、最终与平均 slots、每条轨迹构建记忆的耗时 |

构建耗时包含所有候选写入及门控计算，不含最终 QA 读取与答案生成。平均容量取各更新点保存的 slots，ICAE-single 只有一次压缩状态；汇总容量按轨迹平均。完整统计与样例保存在本地，不上传表格。`--tracking disabled` 仅记录本地结果。

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
