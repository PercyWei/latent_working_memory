# 运行 v3 实验

使用 [run.sh](run.sh) 完成单方法的训练与评估。默认 Qwen3-4B-Instruct-2507、K=512、ΔK=32；K 为 baseline 总容量或动态方法首次容量，ΔK 为动态追加容量。方法与训练细节见 [v3 说明](../README.md)，默认配置位于 [configs/v3](../../../../configs/v3/)。

## 1. 准备环境、模型与数据

项目通过 uv 管理环境，使用 Python 3.11、PyTorch 2.6.0。先执行 `uv sync`，再参考 [.env.example](../../../../.env.example) 完善各服务器的 `.env`。

| 内容 | 默认位置 | 覆盖参数 |
|---|---|---|
| Qwen3-4B-Instruct-2507 | `~/models/Qwen3-4B-Instruct-2507` | `--model-path` |
| AE／LM 数据 | `data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/` | `--pretrain-data` |
| QA 数据 | `data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008/` | `--qa-data` |

数据目录须保留 `preparation.json` 与 `train/dev/test.jsonl`。数据集详细格式见 [FineWeb Multisegment](../../data_preparation/fineweb_multisegment/README.md) 和 [FineWeb FactQA](../../data_preparation/fineweb_factqa/README.md)，可从 [ModelScope](https://modelscope.cn/datasets/percyWeeei/latent-working-memory/files) 下载。

预训练读取 `--pretrain-data` 下的 train／dev，后训练阶段读取 `--qa-data` 下的 train／dev；最终评估统一读取 QA 目录的 `test.jsonl`。

默认记录到 SwanLab 项目 `latent-working-memory-v3`，使用项目 `.env` 中的 key。加 `--tracking disabled` 仅保存本地产物，无需 SwanLab key。

## 2. 运行

基础指令为：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --mode full --method icae_single --gpus 0,1 \
  --run-id capacity-comparison_20261010-01
```

支持试跑（`smoke`）和完整运行（`full`，默认）：

| 运行模式 | 每阶段训练样本上限 | 每阶段 steps 上限 | dev / test 数据上限 |
|---|---|---|---|
| `smoke` | 16 | 2 | 4 / 2 |
| `full` | 不同阶段默认配置不同 | 默认 1 epoch | 默认全部 / 全部 |

### 2.1 完整运行

完整训练默认仅评估最后一个保存的 checkpoint，多 checkpoint 评估见[第 2.2 节](#22-单独评估)。

#### 1）Baseline

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --mode full --method icae_single --gpus 0,1 \
  --run-id capacity-comparison_20261010-01 \
  --micro-batch-size-per-gpu 4 \
  --gradient-accumulation-steps 2
```

`--method` 可以设为 `icae_single`、`icae_multi`、`autocompressors`。

#### 2）动态方法

默认配置只声明 `warmup → policy`，须先准备预训练 checkpoint。单独运行一次动态预训练：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --mode full --method dynamic_pretrain --gpus 0,1 \
  --run-id capacity-comparison_20261010-01 \
  --micro-batch-size-per-gpu 4 \
  --gradient-accumulation-steps 2
```

完成后，从 `artifacts/v3/capacity-comparison_20261010-01/train/dynamic-pretrain-k512/pretrain/summary.json` 的 `checkpoint` 字段取得实际目录，替换下例的 `global_step_N`：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --mode full --method memory_change --gpus 0,1 \
  --run-id capacity-comparison_20261010-01 \
  --micro-batch-size-per-gpu 4 \
  --gradient-accumulation-steps 2 \
  --writer-mode local \
  --checkpoint artifacts/v3/capacity-comparison_20261010-01/train/dynamic-pretrain-k512/pretrain/checkpoints/global_step_N
```

- `--method` 可以设为 `memory_change`、`information_loss`。
- `--writer-mode` 仅用于动态后训练，可选 `local`、`tag`、`mask`、`dual_lora`；使用 `tag` 时可加 `--tag-tokens 3`。
- 同系列后训练可省略 `--run-id`，自动继承 checkpoint 来源预训练运行的实验编号。
- 若要执行完整训练流程，在配置中声明 `"stages": ["pretrain", "warmup", "policy"]`，将 `training.max_pretrain_train_samples` 设为 `12800`，然后通过 `--config` 启动。

### 2.2 单独评估

评估一个方法 run 当前保留的全部 checkpoint：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --evaluate-only --gpus 0,1 \
  --eval-run artifacts/v3/capacity-comparison_20261010-01/train/information-loss-k512+32-local
```

也可显式选择一个或多个 checkpoint：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --evaluate-only --gpus 0,1 \
  --checkpoint \
    artifacts/v3/capacity-comparison_20261010-01/train/information-loss-k512+32-local/policy/checkpoints/global_step_25 \
    artifacts/v3/capacity-comparison_20261010-01/train/information-loss-k512+32-local/policy/checkpoints/global_step_50
```

- `--evaluate-only` 须搭配 `--eval-run` 或 `--checkpoint`，两者互斥。
- `--eval-run` 仅用于单独评估，指向 `artifacts/v3/<run-id>/train/<method-dir>/` 方法级目录，会扫描其各阶段当前保留且包含 `state.pt` 的 checkpoint。
- `--checkpoint` 仅接受属于同一 run 下的单/多个 checkpoint 列表。
- 多 checkpoint 评估时支持设置多 GPU 并行计算，全部计算结束后，由父进程汇总结果，串行上传完整报告，checkpoint 按真实累计 step 排列。
- 默认沿用评估配置 `plan/<method-dir>/test.json`。缺少该文件时，必须传入 `--test-data` 以创建评估配置，`--max-test-samples`、`--max-new-tokens` 可使用默认值。模型、方法和写入配置等从 checkpoint 读取。
- 每个 checkpoint 的结果独立保存。相同 checkpoint、相同配置复用已有结果和日志；在线发布将本次选中结果与此前已发布结果合并，按真实累计 step 排序重建完整报告；其他离线结果须明确选择后才发布。内容相同时仅检查、修复云端布局；需要更新时从缓存重建，不重新计算 QA，也无需保留此前 checkpoint 的权重。计算日志见各 checkpoint 的 `eval.log`，发布状态及错误见汇总 `status.json`。
- 默认追加到原训练 SwanLab run，须保留原训练目录与身份文件且云端 run 状态为 `FINISHED`；加 `--tracking disabled` 仅保存本地结果。

### 2.3 中断恢复

从中断阶段的 checkpoint 恢复当前方法，随后完成未启动的后续阶段和最终评估；动态预训练仅完成预训练：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --resume --gpus 0,1 \
  --checkpoint artifacts/v3/capacity-comparison_20261010-01/train/information-loss-k512+32-local/warmup/checkpoints/global_step_N
```

- 训练配置等信息均沿用原运行，拒绝 CLI 参数覆盖。
- 可重新选择物理 GPU，卡数须与原任务一致。

## 3. 复用动态预训练

- **标识**：同系列后训练省略 `--run-id` 时继承 checkpoint 来源预训练的编号，跨系列复用时显式指定新的 `--run-id`。`--group` 独立指定，默认采用当前 `run-id`。
- **来源**：SwanLab 的 `pretraining` 配置与本地 `run.json` 记录来源编号、名称、URL、checkpoint 和 step；保留原预训练路径、`pretrain/run.json` 及在线运行的 `swanlab.json`。
- **使用范围**：动态预训练本身不执行追加／覆盖，因此同一预训练 checkpoint 可用于不同 **追加容量**、**写入方式** 或 **扩容方式** 的后训练。更换 **基座模型** 或 **首次容量 K** 时则需要重新进行预训练。

## 4. 调整参数

以下默认值以正式运行（`full`）和仓库预设为准，`smoke` 的额外限制见[第 2 节](#2-运行)。

### 4.1 训练输入与执行预算

| 参数 | 用途／默认值 |
|---|---|
| **运行方式** |
| `--mode` | 运行模式，支持 `full`（默认）和 `smoke` |
| `--train-only` | 只进行训练及 dev 验证，跳过 test 评估 |
| `--resume` | 从 checkpoint 恢复并沿用原配置继续训练及评估，须指定 `--checkpoint` |
| `--evaluate-only` | 仅评估已有 checkpoint，须指定 `--checkpoint` 或 `--eval-run` |
| **配置与模型** |
| `--method` | 训练的方法，必须指定；方法列表见[第 2.1 节](#21-完整运行) |
| `--config` | 所选方法的 JSON 配置文件，默认 `configs/v3/` 下的预设 |
| `--model-path` | 基座模型位置，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--seed` | 随机种子，控制初始化、样本选择、数据打乱等随机性，默认 20261004 |
| **目录与设备** |
| `--run-id` | 实验运行编号，默认 `YYYYMMDD-HHMMSS` |
| `--output-root` | 运行产物的父目录，默认 `artifacts/v3` |
| `--gpus` | 物理 GPU 编号，默认 `0,1` |
| **数据集** |
| `--pretrain-data` | 预训练数据集目录，默认位置见[第 1 节](#1-准备环境模型与数据) |
| `--qa-data` | QA 数据集目录，默认位置见[第 1 节](#1-准备环境模型与数据) |
| `--max-pretrain-train-samples` | 预训练阶段训练样本数上限，默认 12800 |
| `--max-qa-train-samples` | QA 训练样本数上限，默认 `null`（全部） |
| `--max-pretrain-data-tokens` | 预训练正文上限，默认 12288 |
| `--max-qa-data-tokens` | QA 训练正文上限，默认 12288 |
| **训练时长** |
| `--epochs` | 每阶段训练轮数，默认 1 |
| `--max-steps` | 每阶段优化器更新步数上限，`full` 默认 `null`（完整 epoch），`smoke` 默认 2 |
| **批处理** |
| `--micro-batch-size-per-gpu` | 每张 GPU 一次并行处理的样本数，默认 4 |
| `--gradient-accumulation-steps` | 每次优化器更新前累积的 microbatch 数，默认 2 |
| `--qa-batch-size` | 每条轨迹每次并行读取的题数上限，默认 8 |

- `--train-only`、`--resume`、`--evaluate-only` 互斥。
- 配置中的 `objective.method` 须与所选方法对应，`objective.stages` 指定训练阶段顺序。独立动态预训练（`--method dynamic_pretrain`）的 method／stages 分别为 `dynamic`／`["pretrain"]`。
- 长度筛选仅用于 train 正文，不含系统提示、gist token、目标文本等。`--max-<pretrain/qa>-data-tokens` 接受正整数或 `null`：超限样本整条排除，`null` 跳过该长度筛选。实际每次编码／解码仍须满足模型窗口限制。
- 全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数，默认按全局 batch 16 进行设置。`--qa-batch-size` 控制读取并行量，不改变选题数量。

### 4.2 训练目标与压缩策略

| 参数 | 用途／默认值 |
|---|---|
| **配置** |
| `--bptt-steps` | 每个反向传播窗口包含的写入轮数，AutoCompressors 默认 2，动态后训练默认 `null`（完整 BPTT） |
| **AE／LM 目标** |
| `--lm-ratio` | 使用 AE／LM 数据进行训练时选择 LM 目标的概率，默认 0.5，AutoCompressors 始终使用 LM |
| `--lm-target-tokens` | LM 续文目标的 token 数，默认 512 |
| **QA 抽样** |
| `--task-qa-count` | 使用 QA 数据训练时每次选取的 task QA 总数，默认 8 |
| `--task-new-ratio` | 使用 QA 数据训练时选取的 task QA 中新题比例，默认 0.5 |
| `--gate-qa-count` | 每次门控时选取的 gate QA 总数，默认 8 |
| **Baseline 分段** |
| `--icae-min-segments` | ICAE-multi 正文随机分段数下限，默认 3 |
| `--icae-max-segments` | ICAE-multi 正文随机分段数上限，默认 6 |
| `--ac-num-segments` | AutoCompressors 正文固定分段数，默认 4 |
| **动态方法写入与门控** |
| `--append-slots` | 动态后训练每次追加的 slots 数，默认 32；首次写入使用 K slots，覆盖保持末块大小 |
| `--writer-mode` | 动态后训练的写入方式，可选 `local`、`tag`、`mask`、`dual_lora`，默认 `local` |
| `--tag-tokens` | 每种写入动作的控制 token 数，默认 3；仅用于 `tag` |
| `--threshold-i` | `memory_change` 的记忆变化阈值，默认 0.1；变化分数 I 达到阈值时追加 |
| `--threshold-d` | `information_loss` 的旧信息损伤阈值，默认 0.1 |
| `--threshold-g` | `information_loss` 的追加收益阈值，默认 0.1 |
| `--eta` | `information_loss` 在旧信息损伤超限时要求的最低追加收益，默认 0；须小于 `--threshold-g` |

- 选题、跨 epoch 随机性与题池数量检查见[数据说明](../README.md#31-数据输入与通用设置)。
- BPTT 表示跨写入轮次的反向传播；AutoCompressors 还用该窗口组织随机分块。分段与训练目标的具体规则见[方法说明](../README.md#三数据与训练)。

### 4.3 验证、测试与结果记录

| 参数 | 用途／默认值 |
|---|---|
| **Checkpoint** |
| `--save-every` | checkpoint 保存的正整数 optimizer step 间隔或 `after_each_epoch`，默认 25，`smoke` 默认 1 |
| `--save-total-limit` | 保留最近成功保存的 checkpoint 数，默认 2 |
| `--checkpoint` | 用于动态方法后训练初始化、训练恢复或独立评估，其中独立评估支持传入同一 run 的多个 checkpoint 目录 |
| `--eval-run` | 仅用于独立评估，指定 `train/<method-dir>/`，评估其全部保留的 checkpoint |
| **Dev 验证** |
| `--max-dev-samples` | dev 验证样本数上限，默认 `null`（全部），`smoke` 默认 4 |
| `--dev-every` | dev 验证的 optimizer step 间隔，`full` 默认 25，`smoke` 默认 1 |
| **Test 评估** |
| `--test-data` | 用于完整运行或独立评估时覆盖 test 数据目录，默认 `null` |
| `--max-test-samples` | test 评估样本数上限，默认 `null`（全部）；仅新训练 `smoke` 的自动评估默认 2 |
| `--max-new-tokens` | test 评估时 token 生成上限，默认 64 |
| **SwanLab** |
| `--tracking` | SwanLab 记录方式，支持 `online`（默认）和 `disabled`（仅保存本地产物） |
| `--swanlab-project` | SwanLab 项目，默认 `latent-working-memory-v3` |
| `--group` | SwanLab 实验分组，默认当前 `run-id` |

- 除了 `--save-every` 和 `--dev-every` 所指定的 step，epoch 末会自动保存 checkpoint 及进行 dev 验证。

## 5. 查看结果

产物位于 `artifacts/v3/<run-id>/`：

```text
artifacts/v3/<run-id>/
├── plan/<method-dir>/
│   ├── config.json             通用配置
│   ├── train.json              训练设置
│   └── test.json               评估配置
├── train/<method-dir>/
│   ├── status.json             训练流程状态、阶段登记与产物路径
│   ├── swanlab.json            SwanLab 记录
│   ├── train.log               训练日志
│   └── <stage>/
│       ├── run.json            阶段训练实际配置、数据与运行信息
│       ├── summary.json        阶段训练完成情况
│       ├── metrics.jsonl       阶段训练与验证指标
│       ├── details.jsonl       逐样本详细验证记录
│       └── checkpoints/global_step_N/
└── eval/<method-dir>/test/
    ├── status.json             评估流程状态记录
    ├── swanlab.json            SwanLab 记录
    └── <stage>/
        └── global_step_N/
            ├── eval.json       评估配置快照
            ├── eval.log        评估日志
            ├── summary.json    评估指标与数据统计
            └── details.jsonl   逐样本详细评估记录
```

- **目录命名**：`<method-dir>` 为 `<method>-k<K>`（baseline 和动态预训练）或 `<method>-k<K>+<ΔK>-<writer-mode>`（动态后训练）。方法名将 `_` 转为 `-`；试跑在目录名末尾添加 `_<mode>`。
- **实验配置**：`--config` 读取 `configs/v3/` 中的方法预设。`plan/<method-dir>/` 下，`config.json` 保存模型设置、seed 等通用配置，`train.json` 保存训练流程配置，`test.json` 保存评估流程配置。
- **训练与验证**：`run.json` 保存阶段训练实际配置、数据与运行信息；`status.json` 的 `experiment` 保存方法身份与已登记的有序阶段索引，用于关联和校验各阶段 `run.json`；`metrics.jsonl` 保存聚合指标；`details.jsonl` 按需保存 dev QA 逐题观测。
- **评估**：各 checkpoint 的 `summary.json` 汇总指标，`details.jsonl` 按样本保存动作、计时与各道 QA 结果。
- **累计发布**：`eval/<method-dir>/test/swanlab.json` 保存最新成功报告的 run 身份、成员 `summary.json` 路径、报告签名、发布版本和面板信息；上传并核验成功后原子更新，失败时保留此前成功记录。结果与配置由各自文件保存。
- **Checkpoint 与恢复**：图中 `N` 为阶段内 step；checkpoint 保存于 `<stage>/checkpoints/global_step_N/`，阶段 `summary.json` 记录训练完成情况、进度统计与最终 checkpoint 路径，`train/<method-dir>/status.json` 的 `checkpoints` 仅索引仍保留的 checkpoint。初始化、恢复和评估通过 `--checkpoint` 传目录路径。中断恢复见[第 2.3 节](#23-中断恢复)。
- **SwanLab 展示**：同一方法的训练、验证和评估共用 `<method-dir>_<run-id>` run，分别位于 `train/*`、`dev/*` 和 `evaluation/*`。评估按 QA 首次引入段及总体展示 NLL、EM、F1 合图，checkpoint 按真实累计 step 排序；补评估时合并已发布与本次选中结果，按真实累计 step 重建完整报告；滑块表示独立递增的报告发布版本。详细布局见 [SwanLab 组织与展示](../README.md#53-swanlab-组织与展示)。
- **配置与执行记录**：`plan/` 仅保存流程设置和配置；训练、评估的日志与执行结果分别保存于 `train/`、`eval/`。评估任务级 `status.json` 保存任务信息、本轮选择 `selected_checkpoints`、整体状态与进度 `progress`，以及 `checkpoints` 中以 `<stage>/global_step_N` 为键的最近执行状态，包括命令、GPU、起止时间、返回码及计算／发布失败详情。补评估保留未选中的旧条目，整体状态与进度仅统计本轮选择。各 checkpoint 的日志与结果按阶段独立存放。
- **失败重跑**：计算或发布失败时保留本地结果，重新执行原命令可复用已完成结果并补传。在线追加要求云端原训练 run 为 `FINISHED`。
