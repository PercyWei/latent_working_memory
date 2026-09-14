# 20260914_配置组织规则

创建时间：20260914 11:48:20 UTC+08:00
最后修订时间：20260914 21:03:14 UTC+08:00

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

- `data_preparation/` 只保存原始来源、文本构造、去重、基础划分及参考 tokenizer 等基础数据准备参数。当前四项基础数据为 FineWeb-4096、FineWeb-128、SQuAD、PersonaMem-v2 FactQA。
- 每个具体实验占一个目录，目录名能区分模型、任务或实验条件。名称遵守 [命名规范](../docs/naming.md)，日期只在需要区分实际实验时添加。
- `model.json` 保存本实验使用的模型、训练目标、优化器与课程参数；不加入其他阶段不使用的字段。
- `selection.json` 保存已有数据路径、epoch 来源／任务／长度分布、固定评估选样规则和选择 seed。它属于训练／评估协议，不放入 `data_preparation/`；只描述本实验的选择，不枚举整个实验组的训练条件。
- `experiment.json` 引用本实验需要的配置，并明确训练预算（预训练为 epoch 数）、保存安排、设备、最终评估和运行记录参数。动态阶段从预训练 checkpoint 取得的模型结构无需在本地再复制一份；文件按实际职责设置，不为凑齐固定文件数创建空配置。

## 实验与实验组

每个实验能单独选择和执行。实验组是若干实验及运行结果的关联，不在配置目录中增加组目录，不把多个实验的关键参数只保存在组配置中。

运行时选择多个具体实验配置，指定共同的 SwanLab `group` 即可组成实验组。组名属于本次运行记录，不固定成为实验定义的一部分；运行关联与指标记录遵守 [SwanLab 使用规范](../docs/swanlab.md)。

同一配置可参与不同实验组；同一配置重复执行会产生不同的运行实例。启动命令、实际 group、配置快照和结果清单保存在 `artifacts/`，不作为新的源配置放回 `configs/`。产物组织遵守 [实验产物组织](../docs/artifacts.md)；当前 v1 入口的具体布局见下文。

当前主实验整理范围为 10 项：数据类型对比的 semantic/random/mixed，目标对比的 AE-only/joint/AE-warmup，动态 BPTT 对比的 full/tokens1024/updates4，以及单独的 Qwen mixed-2048。前三组是比较关系，不是目录层级。

## 独立性、数据与参数

- 同阶段对照实验从约定初始化各自训练。AE warm-up 自己完成 AE 阶段并切换联合目标，不继承 AE-only 的 checkpoint 或 optimizer；相同初始化 seed 不等于共享一次训练过程。
- 跨阶段初始化是显式依赖：dynamic 可以使用 mixed 预训练 checkpoint。记录具体 checkpoint；同一实验的 `resume` 与新实验的阶段初始化分别处理。
- 共享数据直接引用 `data/<数据集名>/`。仅筛选、抽样、混合或更换 tokenizer 时不创建派生数据副本；实际重构文本时才增加基础数据及构造配置。
- 更换模型或 tokenizer 时，为新实验确定适用的选择规则和配额，验证长度／窗口约束与选样可行性。不要直接把 Llama 的样本配额作为 Qwen 的要求；需要严格配对比较时显式定义共同面板。
- 每个参数只保留一个权威来源，避免在模型、选择、执行配置及命令行中重复维护。同类配置采用一种规范格式，不添加未经需要的继承、合并、别名、fallback 或版本兼容层。
- 创建后核对配置引用、入口可用性与实际选样；涉及训练行为的修改验证相应行为。配置快照记录实际运行值，原始结果及历史执行配置不按新协议改写。

## 历史配置

旧 pilot、冒烟验证、学习率探索和已被正式实验替代的配置统一放入 [archive/](archive/README.md)，保留原参数和用途说明，修复现行文档及测试引用。新实验不默认引用归档配置；归档不表示历史产物也需要迁移。

## 当前配置与运行入口

`data_preparation/` 保留 FineWeb-4096、FineWeb-128、SQuAD、PersonaMem FactQA 四份准备配置。FineWeb-128 的 `data` 字段独立定义构造所用参考 tokenizer 与来源协议，不再借用训练配置。

| 阶段 | 具体实验目录 |
|---|---|
| pretrain | `llama-semantic-2048`、`llama-random-2048`、`llama-mixed-2048` |
| pretrain | `llama-ae-only_mixed-128`、`llama-joint_mixed-128`、`llama-ae-warmup_mixed-128` |
| pretrain | `qwen2.5-3b-instruct_mixed-2048` |
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
| `training.source_schedule` | 来源权重的 epoch 节点；未启用来源显式设为 0 |
| `training.task_schedule` | `ae`、`continuation` 权重的 epoch 节点 |
| `training.length_schedule` | 按 `model.json` 长度档上限命名的权重节点 |
| `evaluation` | `balance_task_lengths` 与 `samples_per_source: {dev, test}`，固定评估面板 |

三种 schedule 均从 `epoch: 1` 开始，节点格式为 `{"epoch": 1, "weights": {...}}`；epoch 递增。来源和任务按节点分段固定，长度在相邻节点的归一化分布间线性插值。权重使用非负数，不要求和为 1；等概率写全 1，避免用循环小数表达精确比例。

训练保留完整有效 train 池，每轮在来源 × 任务 × 长度的乘积分布下求最大无放回整数配额，同时满足全局 batch 整除。`experiment.json` 的 `max_samples_per_epoch` 设置每轮样本上限，`null` 表示不限；实际配额在上限内向下取满足比例及 batch 整除的最大值。不约束 batch 内比例；空池或无法满足约束时明确报错。Qwen 与 Llama 均由当前 tokenizer 和本轮分布确定训练规模。

评估数量是每个来源各自的配额：整数必须严格满足，`null` 尽量使用有效样本。开启均衡时按 AE/LM × 长度档等量选择；关闭时直接按总数选样。评估不随训练 epoch 改变；Qwen 的 dev/test 为 `null`，Llama 保留已有评估配额。

`model.json` 使用 `compression_mode: "sample"`（默认）或 `"mean"`，容量权重课程由 `ratio_curriculum_epochs` 控制。旧的 batch 均衡、任务损失权重、step 长度课程和预训练 `lr_decay_steps` 不再放入正式模型配置。`experiment.json` 使用 `epochs`，总步数由启动时的各轮计划确定。当前默认训练 3 轮，Qwen 每轮上限为 32,000 条，其他配置暂不设上限；文献依据、损失与精确配额公式见 [预训练与评估](../notes/v1/20260910_pretraining_and_evaluation.md#3-按-epoch-选样与容量分配)。

训练目录保存 `epoch-plan.json` 与 `training-result.json`。最终评估通过 `--training-result` 读取实际最终 checkpoint，仅接受完整训练；显式 `--checkpoint` 仍可用于独立诊断。`--stop-after-steps` 只截短本次执行，不改变计划预算；恢复保持原 epoch 数与样本上限。直接训练入口使用 `--max-samples-per-epoch <每轮上限>`。

历史数据、checkpoint 与执行产物不迁移，也不按新协议改写历史结果。新选样与 epoch 进度契约不能替代旧协议做精确续训；不添加自动格式转换。

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
