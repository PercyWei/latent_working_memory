# v3：token memory 容量分配对照

实现依据：`notes/v3/20260926_step1_damage_guided_capacity_experiment.md`。首版实现五个比较对象及动态方法的**局部写入版本**，用于检验“按信息损失程度扩容”能否在相近容量下改善旧信息保留。

## 模型与方法

统一使用 **Qwen/Qwen3-4B-Instruct-2507**。基线每块 512 slots；动态方法首次写入 512 slots，后续每次追加 32 slots。

| 共用部分 | 实现 |
|---|---|
| 写入 | 新文本末尾添加可训练 gist embeddings，取其最终隐藏状态作为记忆 |
| 可训练参数 | 编码 LoRA（rank 128、alpha 32，attention/MLP projections）与 gist embeddings |
| 初始化 | gist embeddings 使用零均值高斯分布，标准差 0.02 |
| 读取 | 从相同基座复制独立冻结 Decoder，不安装 LoRA；训练时仍保留读取损失对记忆的梯度 |
| 数值与位置 | 两端基座 bfloat16、LoRA/embeddings float32；CUDA 训练、验证和最终评估使用 BF16 autocast；普通因果位置编码，dropout 为 0 |
| 激活重计算 | 编码器和解码器默认启用原生非重入逐层 checkpoint；`model.gradient_checkpointing` 控制开关 |
| 训练框架 | 复用 verl `BaseEngine`、优化器和 replicated DDP；按 microbatch 并行处理轨迹，可将动态轨迹分为 BPTT 窗口 |

| 方法标识 | 写入范围与容量 | 训练流程 |
|---|---|---|
| `icae_single` | 完整原文一次压缩为一个块 | AE＋LM → 全历史 QA |
| `icae_multi` | 每段独立压缩，拼接所有块 | 多段 AE＋LM → 全历史 QA |
| `autocompressors` | 新段与累计记忆共同压缩，始终追加 | 随机分段的 next-token LM |
| `memory_change` | 首次 512 slots，覆盖末块或追加 32 slots；按归一化表示变化决定 | AE／LM 预训练 → 动作预热 QA → 策略 QA |
| `information_loss` | 同样的局部写入；按历史可读性损失决定 | 与上一组相同，后两阶段各自训练 |

动态追加输入为 `X_t`，覆盖输入为 `[A, X_t]`，其中 `A` 是当前最后一个块；更早的块不传入写入器。两种动作共享 LoRA 与 gist embeddings。完整历史动作标记、限制 attention、双 LoRA 三种写入版本留待后续架构对照。

`model.memory_slots=512` 决定首次写入大小，`objective.append_slots=32` 决定动态追加大小。覆盖保持末块的 slots 数：第一次追加之前改写 512 slots，之后改写最后新增的 32 slots，更早的块不变。例如“首次写入 → 追加 → 覆盖 → 追加”的总容量为 **512 → 544 → 544 → 576**。动态共享预训练一次压缩完整选中前缀，输出 512 slots；较小的追加与覆盖输出在动作预热、策略训练中学习。

共享预训练与后续阶段须使用相同的模型及 slots 配置；此前 64 slots 的 checkpoint 不适用于当前 512 slots 设置。动态 warmup／policy checkpoint 明确记录 `append_slots`。

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
| 基础预训练 | `MultisegmentSample` 文本，根目录 `train/dev/test.jsonl` 与 `preparation.json` | 按字符段独立分词，在长度上限内随机选择连续前缀段；每条来源只构造一个 AE 或 LM 样本 |
| 已有文本成品 | 基础构造器的 `TextSample`，目录内 `train/dev/test.jsonl` | 由 `pretrain_data_view=text_samples` 显式选择；按输入长度筛选，不截断文本或目标 |
| QA | FactQA 发布目录的 `preparation.json` 与 `train/dev/test.jsonl` | 校验已有来源、事实与题池隔离；保留原文、分段、更新点及 `usage` |

- QA 加载按保存的字符区间独立分词，保留正文、段界、QA 和 `usage`；验证字符布局、事实与题池隔离、使用安排及跨划分来源一致性。构造参数用于追溯，训练不重放采样或来源划分。当前默认数据为 `data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008/`，由现有成品迁移，保留 6–10 段及全部 1007 / 118 / 120 条轨迹。
- 训练入口读取并验证数据后记录实际 token 长度、选中段数、实际任务比例与题数。
- 阶段衔接保存并核对 AE＋LM 来源文档与去重簇，QA 不能与其重叠。
- 基础训练统一使用 `multisegment_random_prefix`，`max_input_tokens=8192`。计算从第一段开始、不超过上限的最大完整段数，再均匀抽取 1 至该段数作为连续输入前缀；只有首段超过上限时才裁剪首段，保留较短输入。
- 两个 ICAE 与动态共享预训练按 `training.lm_ratio` 为每条来源选择 AE 或 LM，默认 LM 概率为 0.5；AE 重建选中前缀，LM 从其实际终点取紧邻的 Q 个 tokens，Q 由训练配置 `training.lm_target_tokens` 指定，默认 512，与数据构造时估算的续文候选长度分别配置。续文依次来自未选中的正文和保存的 `continuation`；不足 Q 时切换为 AE。
- 段数和任务由 `training.seed` 与 `trajectory_id` 确定，各来源独立采样，加载后所有 epoch 复用同一结果。每条来源只生成一条训练样本，运行记录保存续文不足导致的任务切换和实际 AE／LM 数量；AutoCompressors 只使用 LM 目标。
- ICAE-single 与动态共享预训练将完整选中前缀一次写入 512 slots；ICAE-multi 按 `segment_tokens=1024` 独立写入，各块 512 slots，拼接后联合读取。动态 QA 仍按数据保存的段界更新，两种动态方法共用预训练产物。
- 数据构造无放回分批读取来源，直到各划分达到配额；每篇合格文档只生成一条轨迹。默认抽取 3–5 段，每段先采样名义长度 l∈[K,3K]，分别保存 `ceil(4 × l × α)` 个字符；尾部 `continuation` 独立保存 `ceil(4 × Q × α)` 个字符，α 为 `content_reserve_ratio`。默认 K=512、α=1.5，每段 3072–9216 字符，正文 9216–46080 字符，对应名义总长 1536–7680 tokens；Q=512 的尾部为 3072 字符。K64 的正文段为 384–1152 字符，正文共 1152–5760 字符。保存的 `estimated_tokens = len(text) / 4` 包含余量，实际 token 数仍由 tokenizer 决定；换 tokenizer 不改变字符分段。
- 默认数据目录为 `data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/`，**尚未构造，训练前须先生成**。每行保存 `trajectory_id`、正文 `text`、字符区间 `segments` 与尾部候选 `continuation`；正文布局与 FactQA 共用，`source` 仅供追溯。元数据中的 `capacity` 控制构造段长，与 `model.memory_slots` 分别配置；当前两者均为 512。加载只在内存组织前缀和任务，不另存派生文本。

| 目标 | 实现细节 |
|---|---|
| AE / LM | 分别重建输入、预测未写入的续文；每条来源按 `lm_ratio` 选择一种目标，末尾添加 EOS |
| ICAE QA | 完整记忆上读取全部任务题，按实际题数平均 |
| 动态 QA | 每个更新点读取 `task_new_qa_ids + task_old_qa_ids`，按实际题数平均，再平均更新点 |
| 动作预热 | 每次更新以 0.5 概率追加；由 seed、epoch、trajectory_id 确定，两组动作日程相同 |
| 策略训练 | 采用各自门控；默认完整 BPTT，选中路径保留跨步梯度；可按 `objective.bptt_steps` 截断 |
| AutoCompressors LM | 随机分段，累计记忆参与预测及写入；默认每两段为一个 BPTT 子块，之后 detach 累计记忆，参数在整条样本结束后更新 |

AutoCompressors 对每条多段文本轨迹只生成一个 continuation 样本，拼接选中前缀与最多 Q-token 的可用续文，再按自身规则随机分段进行 next-token LM 训练；续文不足 Q 时仍保留 LM 目标。对已有 `TextSample`，AE 样本使用输入全文，continuation 样本使用输入和续写的 token 流。子块内保留跨分段的下一 token 预测，子块首 token 不计损失；按目标 token 数平均。冻结读取端的损失通过当前子块内的记忆写入回传，不使用 QA 微调。当前随机分段范围为 768–1024 tokens，尾段可以更短；短流的首段也可缩短，为后续记忆读取保留至少一个可训练的 next-token 目标，不因随机前缀较短而丢弃来源。

动态 warmup／policy 的 `objective.bptt_steps` 默认 `null`，使用完整轨迹 BPTT；当前先保留此设置。设置为 2 时，首段也计一轮：`[首段, 更新1]`、`[更新2, 更新3]`。每个窗口立即反向传播并累积梯度，随后 detach 全部记忆块；仅在 global batch 结束时更新参数。窗口损失保留原有题目、更新点、轨迹平均权重。截断不改变记忆数值、写入次数或门控题数，后续 QA 无法向此前窗口的写入反传。AutoCompressors 的 `ac_bptt_steps` 独立控制其 LM 截断，不受此参数影响。

每个 global batch 按实际样本数平均，包括不足一批的尾批。`micro_batch_size_per_gpu` 控制每卡一次并行处理的样本／轨迹数，`gradient_accumulation_steps` 控制累积次数；全局 batch 为两者与 GPU 数的乘积。共享预训练默认每卡 **8**、累积 **1** 次；三个 baseline 和动态 warmup／policy 默认每卡 **4**、累积 **2** 次，双卡全局 batch 均为 **16**。`qa_batch_size` 默认为 8。

| 配置 | 作用范围 |
|---|---|
| `configs/v3/dynamic_pretrain.json` | 动态共享预训练，参数仅用于此阶段 |
| `configs/v3/memory_change.json`、`information_loss.json` | 对应动态方法的 warmup＋policy |
| `configs/v3/icae_single.json`、`icae_multi.json` | 对应基线的 AE／LM＋QA |
| `configs/v3/autocompressors.json` | AutoCompressors LM |

批量读写采用独立行的右侧 padding，仅使用有效前缀输出；因果 attention 保证有效位置不会读取右侧 padding，因此不传 padding mask，保留 SDPA 的纯 causal 路径。预训练样本在既定 global batch 的每卡分片内按目标长度、输入长度分组，减少补齐计算；采样、各卡样本归属和损失权重保持不变。动态分支按轨迹独立决策。`qa_batch_size` 控制每条轨迹一次读取的题数，批量调用合并各活跃轨迹的题目，但保留原有每题、更新点和轨迹的损失权重。多个样本共享调用的计时按参与样本分摊，调用/题目数仍按每条样本的逻辑工作量记录。

两端职责固定后，Transformer 逐层重算无需切换 adapter 状态；训练时保持 train 模式并关闭 dropout，生成时临时切换 Decoder 为 eval，使用 KV cache 后恢复原模式。独立 Decoder 每卡增加约 8 GB 的 BF16 权重，以换取逐层重算节省的激活；长目标的词表投影与 CE 继续按 256 个有效位置独立分块重算。当前保留 PyTorch 2.6 环境，不接入 v2 依赖其他接口的 padding-free 后端。实际 4B 长轨迹的显存与吞吐需在 GPU 上测量。

`model.dtype` 指基座权重存储精度；CUDA 的计算使用 BF16 autocast，CPU 保留原精度。实际 `autocast_dtype` 记录于运行元数据和 SwanLab 阶段配置，损失与门控统计使用 FP32。

checkpoint 继续只保存 gist embeddings 和编码 LoRA 等可训练状态，不保存冻结 Decoder。已有共享预训练产物可以通过 `--init-checkpoint` 初始化后续阶段；更改工程设置后的继续训练以新阶段记录为准，不保证与旧计算路径逐步数值一致。

## 启动

终端和公司 GPU 网站使用 [run_gpu.sh](scripts/run_gpu.sh)。脚本开头的 `LWM_REPO_DIR` 固定为当前服务器路径 `/data/zhangdw12/percyw/latent_working_memory`，启动时切换到该目录并设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。换机器时修改此常量；GPU、方法和 microbatch 等仍通过命令行参数指定。默认运行五方法的 `smoke` 流程，模型读取 `~/models/Qwen3-4B-Instruct-2507`，SwanLab project 为 `latent-working-memory-v3`，新运行标识默认按上海当前时间生成，格式为 `YYYYMMDD-HHMMSS`。以下正式命令使用自定义标识 `capacity-comparison_20261007-01`，表示当前五方法的记忆容量分配对照实验；它不是默认值。

```bash
bash /data/zhangdw12/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh --mode smoke --gpus 4,5
bash /data/zhangdw12/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode full --gpus 4,5 --run-id capacity-comparison_20261007-01
```

公司 GPU 网站填写脚本的实际绝对路径即可。`--gpus 4,5` 启动两个训练进程，最终评估使用 GPU 4；卡号默认仍为 `0,1`。只运行一个方法时增加 `--method`。通用命令行参数覆盖各阶段预设；`--max-input-tokens`、`--lm-ratio`、`--lm-target-tokens` 仅影响预训练，`--append-slots` 和 `--bptt-steps` 仅影响动态 warmup／policy。参数与数据准备见 [GPU 任务说明](scripts/README.md)。

公开入口按完整方法运行：ICAE 自动执行 pretrain → QA，动态方法执行共享预训练 → warmup → policy，AutoCompressors 执行 LM；`smoke`、`pilot` 仅缩短各阶段预算，仍走完整流程。需要先单独准备两种动态方法的共同起点时，使用 `--method shared_pretrain`。

```bash
bash /data/zhangdw12/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode full --method shared_pretrain --gpus 4,5 --run-id capacity-comparison_20261007-01
```

已有共享预训练 checkpoint 时，动态方法通过 `--init-checkpoint` 接着执行 warmup → policy → 最终评估，不读取预训练数据：

```bash
bash /data/zhangdw12/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode full --method memory_change --gpus 4,5 \
  --init-checkpoint artifacts/v3/capacity-comparison_20261007-01/train/shared-pretrain-k512_capacity-comparison_20261007-01/pretrain/checkpoints/step-NNNNNN.pt
```

外部 checkpoint 的同阶段 `run.json` 提供 `run-id`，省略时自动继承，显式指定不同值会报错。将 `--method` 改为 `information_loss` 并保持同一 checkpoint，即可在同一系列中分别启动两个动态方法；它们与共享预训练使用相同名称后缀。`--method dynamic` 顺序运行这两种方法，`--method all` 运行全部五种方法。

SwanLab 以一个完整方法为一个 run，阶段间累计 optimizer step。ICAE 的 pretrain 与 QA 共用 run；动态方法的 warmup 与 policy 共用 run，其步数从自身 warmup 开始，不包含共享预训练；AutoCompressors 使用一个 LM run。共享预训练单独记为 `shared-pretrain-k512_<run-id>`，因此一次 `all` 完整流程共六个 run。正式方法名为 `<method>-k512_<run-id>`，试跑方法名为 `<method>-k512_<mode>_<run-id>`，其中 `<mode>` 为 `smoke` 或 `pilot`，共享预训练遵循同一命名规则。动态方法名称中的 `k512` 表示首次容量，追加大小单独记录在 config 中。

同一方法的训练阶段由同一组进程连续执行，复用模型、DDP 和 SwanLab 会话。切换时保留 LoRA 与 gist embeddings，重置 optimizer 和阶段内步数，按新阶段配置切换数据、目标及 seed；累计 optimizer step 连续。共享预训练与不同方法分别启动，最终评估独立执行。

checkpoint 仍分阶段保存在 `train/<run-name>/<stage>/checkpoints/`，来源路径、step、SwanLab ID 和 URL 记录在 config 中。中断恢复使用内部训练模块的 `--resume` 与该阶段保存的 `config.json`，沿用原目录、卡数及批处理设置。

## 评估与产物

默认评估最后一个 checkpoint，不改写训练阈值。最终评估追加到对应方法的 SwanLab run，使用该方法的累计 optimizer step。所有方法在同一批轨迹的最终记忆上回答评估题；按末段新事实、更早旧事实以及段距离分别统计。

所有档位的产物统一保存在 `artifacts/v3/<run-id>/`。相同 `run-id` 的不同档位共用 `plan/<method>/`；试跑和正式训练作为独立运行时，使用不同 `run-id` 或省略该参数自动生成，避免已有方法目录冲突。

```bash
CUDA_VISIBLE_DEVICES=4 uv run --frozen python -m latent_working_memory.v3.evaluate \
  --checkpoint artifacts/v3/capacity-comparison_20261007-01/train/memory-change-k512_capacity-comparison_20261007-01/policy/checkpoints/step-NNNNNN.pt \
  --dataset-dir /absolute/path/to/fineweb-factqa \
  --output-dir artifacts/v3/capacity-comparison_20261007-01/eval/memory-change-k512_capacity-comparison_20261007-01/policy \
  --split test --device cuda --max-new-tokens 64 --log-to-swanlab

uv run --frozen python -m latent_working_memory.v3.compare \
  /absolute/path/to/eval-one/summary.json /absolute/path/to/eval-two/summary.json \
  --output-dir artifacts/v3/capacity-comparison_20261007-01/compare/dynamic
```

| 产物 | 内容 |
|---|---|
| `experiment.json` / `swanlab.json` | 方法的阶段配置与预训练来源，以及唯一的 SwanLab run 身份 |
| `<stage>/config.json` / `<stage>/run.json` | 该阶段配置、模型 revision、来源范围、数据统计与跨阶段运行身份 |
| `<stage>/metrics.jsonl` | 该阶段 optimizer step 对应的训练、开发集损失、容量、整步耗时及 CUDA 峰值显存 |
| `<stage>/checkpoints/step-*.pt` | LoRA、gist embeddings、optimizer、训练游标与 RNG；不重复保存冻结基座 |
| `trajectories.jsonl` | 动作轨迹、门控分数、容量、计时、逐题答案与 NLL/EM/F1 |
| `summary.json` | 问答质量、按段距离分层、平均/最终容量与读写成本 |
| `points.json` / `points.csv` | 同一题池的质量—容量比较点，保留阈值与 oracle 标记 |

写入计时包含实际生成的所有候选；门控题实例数与任务读取分开。CUDA 计时显式同步。ICAE-single 的平均 slots 指唯一一次完整压缩状态；其他方法平均所有更新点的保存容量。生成采用 greedy decoding；提示及答案格式由配置固定。

启动器默认在 `latent-working-memory-v3` 中记录 `train`、`dev` 和最终 `evaluation`，group 默认直接使用 `run-id`，可通过 `--group` 覆盖；`--tracking disabled` 仅保存本地。单独调用评估模块时，`--log-to-swanlab` 将结果追加到 checkpoint 对应的方法 run。未执行的评估指标不会补零。

SwanLab 保留损失、容量、梯度范数、整步耗时和峰值显存曲线，以及最终 NLL／EM／F1 柱状图；新增以下 5 张图，不再上传表格：

| 图表 | 含义 |
|---|---|
| `train/stage` | 随累计 optimizer step 记录阶段：1 = AE＋LM 或 AutoCompressors LM，2 = ICAE QA 或动态动作预热，3 = 动态策略训练 |
| `dev/qa_old_nll` | 验证时读取旧信息的 QA NLL，越低越好 |
| `dev/qa_new_nll` | 验证时读取新信息的 QA NLL，越低越好 |
| `evaluation/capacity` | 合并最终 slots 与更新过程平均 slots，均按轨迹取均值 |
| `evaluation/build_seconds_per_trajectory` | 每条轨迹平均构建记忆的秒数，包含候选写入和门控计算，不含最终 QA 读取及答案生成 |

阶段曲线只记录当前 run 实际执行的步骤，动态方法从阶段 2 开始。两张 QA 曲线沿用任务的统计口径：动态方法读取各更新点，ICAE 读取最终记忆；未验证或没有对应题目时不补零。最终质量图仍将 all／old／new 合并展示。动作次数、门控分数、细分统计及生成样例保存在本地 JSON／JSONL；已有云端 run 的旧表格不自动删除。

## 验证与边界

```bash
uv run --frozen pytest -q tests/v3
uv run --frozen ruff check src/latent_working_memory/v3 tests/v3
```

测试使用本地随机初始化 tiny Llama/Qwen3，覆盖可微冻结读取、分支梯度、历史输入范围、门控隔离、数据契约、BPTT 截断及 CPU DDP。它们验证实现行为，不代表真实 QA 质量或 4B GPU 性能；正式运行前需在 GPU 上测量完整轨迹显存。

参考实现：[ICAE](https://github.com/getao/icae)、[AutoCompressors](https://github.com/princeton-nlp/AutoCompressors)、[verl Model Engine](https://verl.readthedocs.io/en/latest/workers/model_engine.html)。
