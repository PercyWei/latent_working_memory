# 20260914_配置组织规则

创建时间：20260914 11:48:20 UTC+08:00
最后修订时间：20260917 11:24:23 UTC+08:00

本规则用于本项目创建和整理配置。配置区分基础数据准备、具体实验与实验组；正式实验按下述目录组织。

## 目录与职责

```text
configs/
  data_preparation/                 # 构造基础数据
  v1/
    pretrain/
      <具体实验名>/
        model.json                 # 基座、记忆模块与训练参数
        selection.json             # 本实验的数据选取规则
        experiment.json            # 单项实验的执行入口配置
    dynamic/
      <具体实验名>/
        ...                        # 按实际职责保存需要的文件
    capacity/
      <具体实验名>/
        ...
  archive/                         # 旧试跑、冒烟验证及已退役配置
```

- `data_preparation/` 只保存原始来源、文本构造、去重、基础划分及参考 tokenizer 等基础数据准备参数。当前基础数据包括 FineWeb-4096、FineWeb-128、FineWeb 重构数据、SQuAD 和 PersonaMem-v2 FactQA。
- 每个具体实验占一个目录，目录名能区分模型、任务或实验条件。名称遵守 [命名规范](../docs/naming.md)，日期只在需要区分实际实验时添加。
- `model.json` 保存本实验使用的模型、训练目标、优化器与课程参数；不加入其他阶段不使用的字段。
- `selection.json` 保存已有数据路径、epoch 来源／任务／长度分布、固定评估选样规则和选择 seed。它属于训练／评估协议，不放入 `data_preparation/`；只描述本实验的选择，不枚举整个实验组的训练条件。
- `experiment.json` 引用本实验需要的配置，并明确训练预算（预训练为 epoch 数）、保存安排、设备、最终评估和运行记录参数。动态阶段从预训练 checkpoint 取得的模型结构无需在本地再复制一份；文件按实际职责设置，不为凑齐固定文件数创建空配置。

## 实验与实验组

### v2 固定容量重构配置

`v2/pretrain/qwen3-4b_pooling_*` 提供 AE／AE＋LM × warm-up／直接多次压缩四个主实验，以及 `ae-static`、`ae-lm-static` 两个全程单次压缩 baseline。每个目录包含 `model.json`（codec）、`selection.json`（已构造数据目录）和 `experiment.json`（配置引用与执行参数）。入口为 `latent_working_memory.v2.pretrain.train --experiment <配置> --output-dir <产物目录>`，支持单设备与 verl replicated/DDP。

先通过 `v2.pretrain.prepare_data` 和 `data_preparation/fineweb-reconstruction-k512-doc100k.json` 独立构造数据：沿用 v1 质量过滤、去重与来源划分，按字符数÷4 估算长度，只保存 `single/`、`multi/` 两套索引，记录原始 Parquet 的相对路径、row group、组内行号、候选字符范围和目标 `write_token_ends`。`content_reserve_ratio=1.5` 为主内容留余量，`continuation_reserve_tokens=768` 为真实 `continuation_tokens=512` 的续文预留字符预算；候选字符长度为 `ceil(4 × L × 1.5) + 4 × 768`。构造不加载 tokenizer；训练分词后按目标 token 计划截取连续正文及 512-token 续文，额外缓冲不进入训练。六份 `selection.json` 只包含 `dataset_dir`。训练启动时按文件和 row group 批量读取原文、分词并按真实长度筛选一次，各 epoch 完整复用；AE／AE＋LM 的过滤一致，记录候选数、保留数和原因。预算由 `warmup_epochs`、`multiround_epochs` 与各阶段筛选后的样本数计算，尾批保留；`global_batch_size` 不随 world size 改变。静态 baseline 设置 `warmup_epochs=3`、`multiround_epochs=0`；六组均训练 3 epochs，使用相同的多次压缩 dev/test 与评估指标。`compression` 可选 `mean`、`weighted`、`spectral`。六份默认配置均使用完整 causal Encoder（`encoder_layers: null`）与基础 pooling。正式入口默认 SwanLab online，project 为 `latent-working-memory-v2`，要求显式指定同组 runs 共用的 `--swanlab-group`；具体参数进入 config，运行名称取输出目录名。已完成真实模型资源短测，完整训练仍保持停止；运行、恢复与产物说明见 [v2 开发记录](../src/latent_working_memory/v2/README.md)。

### v2 初始静态训练配置

`v2/pretrain/gmsa-qwen3-4b/model.json` 保存跨 AE／QA 阶段的 GMSA 模型结构；
`v2/pretrain/gmsa-qwen3-4b/experiment.json` 和 `v2/finetune/gmsa-qwen3-4b/experiment.json`
分别保存阶段、token 预算和 HF Trainer 参数。当前静态入口通过 `--model-config`、`--training-config`
显式传入两份配置，数据直接通过 `--train-file`／`--eval-file` 引用共享 JSONL，不创建空的 selection 配置。
这些是 4K 输入的开发起点，尚未完成真实模型显存和论文效果验证；不是已执行实验记录。
具体命令及跨阶段初始化、同 run 恢复规则见 [v2 开发记录](../src/latent_working_memory/v2/README.md)。

### 通用关联规则

每个实验能单独选择和执行。实验组是若干实验及运行结果的关联，不在配置目录中增加组目录，不把多个实验的关键参数只保存在组配置中。

运行时选择多个具体实验配置，指定共同的 SwanLab `group` 即可组成实验组。组名属于本次运行记录，不固定成为实验定义的一部分；运行关联与指标记录遵守 [SwanLab 使用规范](../docs/swanlab.md)。

同一配置可参与不同实验组；同一配置重复执行会产生不同的运行实例。启动命令、实际 group、配置快照和结果清单保存在 `artifacts/`，不作为新的源配置放回 `configs/`。产物组织遵守 [实验产物组织](../docs/artifacts.md)；当前 v1 入口的具体布局见下文。

当前包含数据类型对比的 semantic/random/mixed、目标对比的 AE-only/joint/AE-warmup、动态 BPTT 对比的 full/tokens1024/updates4，以及 Qwen mixed-2048 的每轮 32k／64k 两种预算。比较关系不作为配置目录层级。

## 独立性、数据与参数

- 同阶段对照实验从约定初始化各自训练。AE warm-up 自己完成 AE 阶段并切换联合目标，不继承 AE-only 的 checkpoint 或 optimizer；相同初始化 seed 不等于共享一次训练过程。
- 跨阶段初始化是显式依赖：dynamic 可以使用 mixed 预训练 checkpoint。记录具体 checkpoint；同一实验的 `resume` 与新实验的阶段初始化分别处理。
- 共享数据直接引用 `data/<数据集名>/`。已有原文时直接保存来源位置和字符索引，跨实验复用原始数据；各实验、更换 tokenizer 或 epoch 不另存正文或 token 数据副本。重构候选索引采用字符范围，真实长度在训练启动时筛选。
- 更换模型或 tokenizer 时，为新实验确定适用的选择规则和配额，验证长度／窗口约束与选样可行性。不要直接把 Llama 的样本配额作为 Qwen 的要求；需要严格配对比较时显式定义共同面板。
- 每个参数只保留一个权威来源，避免在模型、选择、执行配置及命令行中重复维护。同类配置采用一种规范格式，不添加未经需要的继承、合并、别名、fallback 或版本兼容层。
- 创建后核对配置引用、入口可用性与实际选样；涉及训练行为的修改验证相应行为。配置快照记录实际运行值，原始结果及历史执行配置不按新协议改写。

## 历史配置

旧 pilot、冒烟验证、学习率探索和已被正式实验替代的配置统一放入 [archive/](archive/README.md)，保留原参数和用途说明，修复现行文档及测试引用。新实验不默认引用归档配置；归档不表示历史产物也需要迁移。

## 当前配置与运行入口

`data_preparation/` 保留 FineWeb-4096、FineWeb-128、FineWeb 重构数据、SQuAD 和 PersonaMem FactQA 的准备配置。FineWeb-128 的 `data` 字段独立定义构造所用参考 tokenizer 与来源协议，不再借用训练配置。

| 阶段 | 具体实验目录 |
|---|---|
| pretrain | `llama-semantic-2048`、`llama-random-2048`、`llama-mixed-2048` |
| pretrain | `llama-ae-only_mixed-128`、`llama-joint_mixed-128`、`llama-ae-warmup_mixed-128` |
| pretrain | `qwen2.5-3b-instruct_mixed-2048`、`qwen2.5-3b-instruct_mixed-2048_epoch64k` |
| dynamic | `bptt-full_squad`、`bptt-tokens1024_squad`、`bptt-updates4_squad` |

每个目录包含 `model.json`、`selection.json`、`experiment.json`。动态实验的 `model.json` 保存动态训练与评估课程，语言模型结构由 `experiment.json` 指定的预训练 checkpoint 提供。外部预训练配置不含动态／容量阶段及基础数据构造字段；内部 checkpoint 配置契约保持原结构。

所有命令从项目根目录运行。`experiment.json` 中 `model`、`selection` 引用相对于该文件；共享数据路径及初始化 checkpoint 路径相对于项目根目录。入口先检查全部配置，再将三份配置快照、运行分组、命令、状态及报告路径写入 `artifacts/v1/<系列>/plan/`。

```bash
.venv/bin/python -m latent_working_memory.v1.pretrain.experiment \
  --experiments configs/v1/pretrain/<具体实验名>/experiment.json \
  --output-dir artifacts/v1/<新系列> \
  --swanlab-group <本次分组> --swanlab-mode online
```

`--experiments` 可接多个配置，按传入顺序独立训练并测试；多实验必须显式给出 `--swanlab-group`。单实验的 group 可省略。`--swanlab-tag study:<研究性质>` 由调用方指定。动态实验改用 `latent_working_memory.v1.dynamic.experiment` 与对应阶段配置，入口依次完成选样和评估记录准备、训练、最终 test；相同选择参数生成相同评估记录。

添加 `--plan-only` 仅加载配置、检查引用和生成命令，不加载真实数据／权重、不启动训练、不连接 SwanLab；状态为 `planned`。正式执行使用新的产物系列目录。GPU 来自具体实验配置并受项目允许范围约束；入口不自行设置 `LWM_ALLOWED_PHYSICAL_GPUS` 扩大权限。

正常入口默认 `--swanlab-mode disabled`；`online` 将最终 test 追加到对应训练 run，`offline` 保留本地记录。入口不自动生成跨实验 compare run。需要比较时，预训练调用 `v1.pretrain.publish_reports --reports <系列>/plan/reports.json --output-dir <系列>/compare/<名称>`，并显式传入 `--swanlab-project`、`--swanlab-group` 与发布模式；动态比较使用 `v1.dynamic.reporting --reports`。`plan-only` 中的报告清单是预期路径，不代表结果已生成。

同一运行中断后，直接使用计划中记录的训练命令和配置快照。预训练增加 `--resume <该运行 checkpoint>`；动态训练将 `--checkpoint` 改为自身动态 checkpoint 并增加 `--resume`。新建单项实验与中断续训不共用初始化语义；最终 test 可独立执行计划中的评估命令。

## 预训练选样格式与配额

`selection.json` 只包含 `sources`、`seed`、`training`、`evaluation`：

| 字段 | 内容 |
|---|---|
| `sources` | 共享数据来源及路径，同时定义可用训练来源与评估来源 |
| `training.input_tokens` | 可选的输入 X 长度闭区间 `{min, max}`；省略时使用 1 到模型的 `max_input_tokens`，显式上限不能超过模型上限 |
| `training.source_schedule` | 来源权重节点，或 `null`（全程不约束来源比例） |
| `training.task_schedule` | 允许任务集合及权重节点，或 `null`（全程允许两种任务且不约束比例） |
| `training.length_schedule` | 长度档权重节点，或 `null`（不约束长度比例，仍按长度范围筛选） |
| `evaluation` | `balance_task_lengths` 与 `samples_per_source: {dev, test}`，固定评估面板 |

非空 schedule 从 `epoch: 1` 开始，节点格式为 `{"epoch": 1, "weights": {...}}`，也可用 `weights: null` 在该节点取消比例约束。任务节点额外接受 `tasks: ["ae"]` 或 `["ae", "continuation"]` 等非空任务集合；省略时允许两种任务，显式权重必须覆盖允许集合。来源和任务按节点分段固定，长度只在两个非空权重节点间对归一化分布作线性插值；与 `null` 之间按节点切换。权重非负且至少一项为正，等概率写全 1。

训练保留完整有效 train 池，epoch 采样器先应用训练长度范围及允许任务，再仅对受比例约束的维度建立组合池、计算最大无放回整数配额。三个 schedule 都为 `null` 时，直接对整个有效池等概率抽样；AE warm-up 可先允许 AE，再允许两种任务，不要求二者等量。完整配置示例见 [预训练与评估 3.1](../notes/v1/20260910_pretraining_and_evaluation.md#31-三种独立分布)。

`experiment.json` 的 `max_samples_per_epoch` 设置每轮样本上限，`null` 表示不限；实际数量向下取满足现有比例约束和全局 batch 整除的最大值。无比例约束时只受有效池规模、样本上限和 batch 整除约束。不要求 batch 内比例；无法组成完整 batch 时明确报错。训练长度范围不截断原文、不改变 LM 目标长度和读写窗口检查，也不改变 dev/test 选样。

评估数量是每个来源各自的配额：整数必须严格满足，`null` 尽量使用有效样本。开启均衡时按 AE/LM × 长度档等量选择；关闭时直接按总数选样。评估不随训练 epoch 改变；Qwen 的 dev/test 为 `null`，Llama 保留已有评估配额。

`model.json` 使用 `compression_mode: "sample"`（默认）或 `"mean"`，容量权重课程由 `ratio_curriculum_epochs` 控制。旧的 batch 均衡、任务损失权重、step 长度课程和预训练 `lr_decay_steps` 不再放入正式模型配置。`experiment.json` 使用 `epochs`，总步数由启动时的各轮计划确定。当前默认训练 3 轮，Qwen 默认每轮上限为 32,000 条，`epoch64k` 配置为 64,000 条并使用物理 GPU 6、7；其他配置暂不设上限；文献依据、损失与精确配额公式见 [预训练与评估](../notes/v1/20260910_pretraining_and_evaluation.md#3-按-epoch-选样与容量分配)。

训练目录保存 `epoch-plan.json` 与 `training-result.json`。最终评估通过 `--training-result` 读取实际最终 checkpoint，仅接受完整训练；显式 `--checkpoint` 仍可用于独立诊断。`--stop-after-steps` 只截短本次执行，不改变计划预算；恢复保持原 epoch 数与样本上限。直接训练入口使用 `--max-samples-per-epoch <每轮上限>`。

历史数据、checkpoint 与执行产物不迁移，也不按新协议改写历史结果。新选样与 epoch 进度契约不能替代旧协议做精确续训；不添加自动格式转换。

## v1 训练执行框架

v1 的预训练与动态训练使用 verl 0.8.0 的自定义 engine。`EngineRegistry` 分别注册 `lwm_pretrain` 与 `lwm_dynamic`，共用 `MemoryEngine` 的复制参数后端；optimizer 由 verl 配置构造，zero-grad → forward/backward → optimizer-step 的外层生命周期使用 `BaseEngine.train_batch`。两阶段直接以 SPMD 方式调用 engine，未接入 Ray TrainingWorker；不叠加 Accelerate 或其他 trainer。

现有命令通过 `torch.distributed.run` 启动，CUDA 进程组使用 verl 的初始化入口；CPU 单卡和 Gloo 多进程用于验证。当前 backend 为 `replicated`，多卡使用 PyTorch DDP，CUDA 运算为 BF16、可训练参数保持 FP32。当前仅提供复制参数后端。

预训练保持全局 epoch 计划、长度排序、rank 分配和 sample/mean 容量权重；非末 microbatch 使用 no_sync，末批同步。动态训练将一个 episode 划分为反传区间：完整 BPTT 为一个区间，TBPTT 按 token 数或写入次数分段。每个区间内部连续传递 memory，边界处 backward 和 detach；最后一个有 loss 的本地区间执行唯一一次同步，尾部无 QA 的写入仍然执行。每个全局 batch 只裁剪和更新一次 optimizer。

损失保留全局样本平均及 episode 内读取平均。backward 保持原损失尺度，DDP 平均后再恢复参数梯度尺度；动态阶段保留逐次写入的 autocast 边界，避免改变 BF16 权重转换缓存和梯度累加。非有限梯度直接中止，不采用跳过更新的默认 SFT 策略。

checkpoint 继续使用项目的单文件契约，保存 Writer、投影、Reader LoRA、optimizer、epoch/order/cursor 与各 rank RNG，冻结基座继续从配置引用加载。独立评估和动态初始化使用同一导出。固定 dev/test、同步评估及 SwanLab global step 的语义保持原样。

依赖固定为 `verl==0.8.0`，因为 0.9.0 要求 Transformers 5.x，而本项目保留 4.x。该版本要求 NumPy <2，锁文件使用 1.26.4；PyTorch 和 Transformers 的锁定版本保持不变。开发与验证应使用当前 Git worktree 根目录 `.venv/`，由本分支的 `uv.lock` 创建，不修改其他 worktree 的环境。

## 预训练 CPU 分词

`v1.pretrain.experiment` 与 `v1.pretrain.train` 接收以下运行参数，完整实验入口将实际值写入计划命令：

| 参数 | 命令行默认值 | 用途 |
|---|---:|---|
| `--tokenizer-workers` | 4 | 每个训练进程的 CPU worker 数；0 表示同步执行 |
| `--tokenization-batch-size` | 256 | 启动筛选时每次分词的文本样本数 |
| `--prefetch-batches` | 2 | 预取的后续全局 batch 数；0 表示按需读取 |

`v1.pretrain.data_selection` 和 `v1.pretrain.evaluate` 的 `--data-selection` 路径使用前两项参数。worker 采用 `spawn`，进程内关闭 tokenizer 的线程并行；两个训练进程使用 4 workers 时共占用 8 个 CPU workers。参数描述 CPU 执行方式，独立于 GPU microbatch 和实验采样比例。

选样重新分词计算长度，丢弃 token IDs，仅在内存保留文件位置及选样元数据；训练 batch 到达时重新分词，用完释放。并行结果按原始顺序合并，预取不推进选样或容量随机状态。checkpoint 只保存已取用 batch 的游标，恢复时重建未消费的预取任务。运行参数另存训练 `provenance.json`，无需改变 checkpoint 格式。

## 动态训练与评估来源

动态 `selection.json` 将训练来源与 dev/test 来源独立配置，来源名称用于结果目录和图表；名称不从实验名或 tags 推断。例如：

```json
{
  "sources": {
    "squad": {"dataset": "squad", "dataset_dir": "data/squad"},
    "personamem-factqa": {
      "dataset": "personamem",
      "dataset_dir": "data/personamem-v2-factqa-32k-doc100_20260913"
    }
  },
  "training": "squad",
  "evaluation": {
    "dev": ["squad", "personamem-factqa"],
    "test": ["squad", "personamem-factqa"]
  }
}
```

`dataset` 选择适配器；同一适配器可用于多个独立命名的来源。dev/test 列表可以不同，只读取相应划分，不混合计分；空 dev 列表关闭训练中验证。评估来源只检查评估预算，不要求满足训练课程配额。

准备入口接收 `--selection`，生成 `evaluation-sets.json` 和各来源的固定评估计划。训练与独立评估统一接收 `--evaluation-sets`，训练报告位于 `dev/<来源>/`，独立评估位于 `<输出目录>/<来源>/`；dev 按来源分图，test 按指标合并展示。来源定义和固定请求纳入续训一致性校验。
