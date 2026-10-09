# v3：token memory 容量分配对照

比较三种基线与两种动态扩容方法，检验“按信息损失程度扩容”能否在相近容量下改善旧信息保留。当前动态方法采用局部写入：只在覆盖时将最后一个记忆块传入编码器。

[运行指南](scripts/README.md) · [实验配置](../../../configs/v3/) · [实验方案](../../../notes/v3/20260926_step1_damage_guided_capacity_experiment.md)

## 方法与模型

记忆以向量块保存，一个 slot 对应一个记忆向量。容量使用以下符号：

| 符号 | 含义与配置字段 | 默认值 |
|---|---|---:|
| `K` | ICAE 总容量、AutoCompressors 每块容量、动态首次写入容量；`model.memory_slots` | 512 slots |
| `ΔK` | 动态方法每次追加的新块容量；`objective.append_slots` | 32 slots |
| `n` | ICAE-multi 的压缩块数；`objective.icae_min_segments` 至 `icae_max_segments` | 3–6 块 |

AE 表示重建输入，LM 表示预测后续文本，QA 表示问答训练。

| 方法 | 记忆组织 | 默认训练流程 |
|---|---|---|
| `icae_single` | 完整历史一次压缩为 K slots | AE／LM → QA |
| `icae_multi` | 完整历史均分为 n 块，独立压缩后拼接，总容量为 K slots | 多段 AE／LM → QA |
| `autocompressors` | 新段与累计记忆共同压缩，每次追加 K slots | 随机分段 next-token LM |
| `memory_change` | 覆盖末块或追加 ΔK slots，按记忆表示变化决定 | AE／LM → 动作预热 → 策略训练 |
| `information_loss` | 同样的写入方式，按旧信息损失决定 | AE／LM → 动作预热 → 策略训练 |

五组使用相同的模型设置，各自采用表中的记忆组织和训练目标：

| 共用部分 | 设置 |
|---|---|
| 基座 | Qwen3-4B-Instruct-2507 |
| 写入 | 在输入末尾添加可训练 gist embeddings，取对应位置的最终隐藏状态作为记忆 |
| 可训练参数 | 编码 LoRA（rank 128、alpha 32，作用于 attention／MLP projections）及 gist embeddings；后者使用零均值、标准差 0.02 的高斯初始化 |
| 读取 | 从相同基座复制独立冻结 Decoder，不安装 LoRA；读取损失仍向记忆及写入器反传 |
| 执行 | verl `BaseEngine`、优化器与 DDP；默认使用 BF16、SDPA 和非重入逐层激活重计算 |

ICAE-multi 的分块规则用于 AE／LM、QA 与评估：

- 按 seed 和样本 ID 在范围内均匀采样 n，同一样本跨 epoch 保持不变。
- 正文按 token 顺序均分，块长最多相差 1；容量按商与余数分配，总和严格为 K。例如 K=512、n=3 时为 **171／171／170 slots**。
- 各块复用同一组 gist embeddings，仅分配 `ceil(K / icae_min_segments)` 行。FactQA 原始段界用于题目来源和 old／new 划分。

## 两种扩容规则

令 `X_t` 为新文本、`A` 为最后一个记忆块：

- **追加**：输入 `X_t`，输出 ΔK slots 的新块；所有旧块保持不变。
- **覆盖**：输入 `[A, X_t]`，输出与 `A` 等长的新块，替换 `A`。
- 更早的块不参与写入；任务 QA 读取全部已保存的块。两种动作共享 LoRA 和 gist embeddings。

“首次写入 → 追加 → 覆盖 → 追加”的总容量为 `K → K+ΔK → K+ΔK → K+2ΔK`。覆盖保持末块大小：第一次追加前为 K，之后为 ΔK。以 K=512、ΔK=32 为例，总容量为 **512 → 544 → 544 → 576**。

| 方法 | 决策过程 |
|---|---|
| `memory_change` | 先生成覆盖候选；对候选与旧末块逐 slot 做 RMS（均方根）归一化，计算相对 Frobenius 距离 `I`。`I >= threshold_i` 时另生成追加块，否则保存覆盖候选。归一化仅用于评分。 |
| `information_loss` | 分别生成覆盖与追加候选，用固定的历史门控 QA 比较更新前、覆盖后、追加后的可读性。 |

信息损失规则如下，NLL 表示负对数似然，越低越好：

```text
L0   = 更新前记忆的平均答案 NLL
Lrw  = 覆盖后记忆的平均答案 NLL
Lapp = 追加后记忆的平均答案 NLL
d = Lrw - L0        # 覆盖造成的损失增加
g = Lrw - Lapp      # 追加相对覆盖的收益
追加 ⇔ g > threshold_g 或 (d > threshold_d 且 g > eta)
```

- 门控题固定使用、不筛题；每题先平均答案 token 的 NLL，再平均题目。决策无梯度，任务 QA 与门控题分离，只有选中的记忆路径参与任务损失。
- 信息损失法在评估时也需要门控参考答案，结果标记为 **offline oracle（使用参考答案决策）**。
- 默认阈值尚未校准；质量—容量对照需分别训练不同阈值配置。

## 数据与训练

| 数据输入 | 目录格式 | 用途 |
|---|---|---|
| FineWeb 多段文本 | `preparation.json`、`train/dev/test.jsonl`；每行是 `MultisegmentSample` | ICAE 和动态共享预训练的 AE／LM，以及 AutoCompressors 的 LM |
| FactQA | `preparation.json`、`train/dev/test.jsonl`；保留原文、字符段界、QA 和 `usage` | ICAE QA、动态动作预热／策略训练及最终评估 |
| 已有文本成品（可选） | `TextSample` 格式的 `train/dev/test.jsonl` | 通过 `pretrain_data_view=text_samples` 选择，保留完整输入与目标 |

数据构造见 [FineWeb 多段文本](../data_preparation/fineweb_multisegment/README.md) 和 [FactQA](../data_preparation/fineweb_factqa/README.md)。预训练与 QA 的来源文档及去重簇不得重叠；加载时校验划分、来源和题池隔离。

**预训练输入与目标**

- 默认 `multisegment_full_text`：使用完整正文，超出 `training.max_input_tokens` 的样本整条过滤。该参数仅计算正文 tokens，不计记忆、提示和目标；`null` 使用模型窗口，预设为 12288（12k）。模型仍检查每次实际编码／解码的窗口长度。
- ICAE 与动态共享预训练为每条来源选择一个目标：`training.lm_ratio` 控制 LM 概率（默认 0.5）；AE 重建全文，LM 预测保存的紧邻续文。LM 目标最多取 `training.lm_target_tokens`（默认 512）；续文不足时转为 AE。目标选择由 seed 和样本 ID 决定，跨 epoch 复用。
- ICAE-single 与动态共享预训练一次将完整正文压缩为 K slots；ICAE-multi 按上述规则独立压缩后联合读取。
- AutoCompressors 始终使用 LM：拼接完整正文与可用续文，按 `objective.ac_min_segment_tokens` 至 `objective.ac_max_segment_tokens` 随机分段（默认 768–1024 tokens），尾段允许更短。截断周期由 `objective.ac_bptt_steps` 控制（默认每两段），按目标 token 数平均损失。

QA 保留完整轨迹与对应题池。训练数据上限由预设的 `training.stage_max_train_samples` 按阶段分别指定，过滤后以 seed 可复现地选取；`null` 表示不限：

| 阶段 | 默认训练样本上限 |
|---|---:|
| ICAE／共享预训练 `pretrain`、AutoCompressors `lm` | 12800 |
| ICAE `qa`、动态 `warmup`／`policy` | 全部 |

各阶段可独立通过 CLI 覆盖；`smoke`／`pilot` 另受试跑档位上限约束，取较小值。预训练正文长度过滤不应用于 QA。

**动态方法的三个阶段**

1. **共享预训练 `pretrain`**：AE／LM 学习一次写入 K slots，两组复用同一 checkpoint。
2. **动作预热 `warmup`**：按 `objective.append_probability` 指定的概率追加（默认 0.5），否则覆盖；两组使用相同的随机动作日程，分别训练。
3. **策略训练 `policy`**：启用各自扩容规则，沿选中的动作路径计算 QA 损失，两组继续分别训练。

QA 损失先平均每题答案 token，再按实际题数合并旧题与新题；动态方法继续平均更新点，最终按轨迹平均。ICAE 在最终记忆上读取全部任务题。

动态阶段默认完整 BPTT，可用 `objective.bptt_steps` 设置截断窗口；例如设为 2 时按两轮截断，首次写入也计一轮。每个窗口反向并 detach 记忆，global batch 结束后更新一次参数；此设置与 AutoCompressors 的截断参数独立。

| 默认批处理 | 每卡 microbatch | 梯度累积 |
|---|---:|---:|
| ICAE 的预训练／QA、动态共享预训练 | 8 | 1 |
| AutoCompressors、动态动作预热／策略训练 | 4 | 2 |

全局 batch = GPU 数 × 每卡 microbatch × 梯度累积；双卡默认均为 **16**，尾批按实际样本数平均。

## 运行

先按[运行指南](scripts/README.md#1-准备环境模型与数据)准备环境、模型、数据和 SwanLab key，并修改 `run_gpu.sh` 中的仓库路径。以下命令在仓库根目录执行：

```bash
# 预览五方法的运行计划
bash src/latent_working_memory/v3/scripts/run_gpu.sh --gpus 0,1 --dry-run

# 完整流程试跑；正式运行将 smoke 改为 full
bash src/latent_working_memory/v3/scripts/run_gpu.sh --mode smoke --gpus 0,1
```

- `--method` 默认 `all`；也可选择表中的单个方法。`smoke`、`pilot`、`full` 控制运行预算。
- 默认配置位于 `configs/v3/`，`objective.stages` 声明阶段顺序；`--method <方法> --config <文件>` 可替换单方法配置，显式 CLI 超参数优先。参数及作用范围见[运行指南](scripts/README.md#4-调整参数)。
- 同一方法的连续阶段复用模型和训练进程，切换阶段时重置优化器；共享预训练与不同方法分别启动。
- 单独准备共享预训练使用 `--method dynamic_pretrain`。后续用 `--method memory_change` 或 `information_loss` 搭配同一个 `--init-checkpoint`，默认连续执行 warmup → policy → 评估，并继承预训练的 `run-id`。保留来源运行目录及身份文件。
- 中断恢复使用阶段目录保存的 `config.json` 和 checkpoint，调用训练模块的 `--resume`。

## 评估与产物

默认评估最后一个 checkpoint，在同一批 FactQA 轨迹的最终记忆上回答评估题，门控题不计入质量指标。记录答案 NLL、EM（精确匹配）和 F1，并按末段新事实、更早旧事实及段距离分层；生成使用 greedy decoding。

```text
artifacts/v3/<run-id>/
├── plan/<method-dir>/            阶段配置、job.json、日志、result.json
├── train/<method-dir>/<stage>/   config.json、run.json、metrics.jsonl、checkpoints/
├── eval/<method-dir>/<stage>/    summary.json、trajectories.jsonl
└── compare/                     points.json、points.csv、compare.log、result.json
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

## 代码入口与验证

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
