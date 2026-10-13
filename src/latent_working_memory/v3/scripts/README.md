# 运行 v3 实验

使用 [run.sh](run.sh) 完成单方法的训练与评估，方法比较单独调用 [compare.py](../compare.py)。默认 Qwen3-4B、K=512、ΔK=32；K 为 baseline 总容量或动态方法首次容量，ΔK 为动态追加容量。方法与训练细节见 [v3 说明](../README.md)，默认配置位于 [configs/v3](../../../../configs/v3/)。

## 1. 准备环境、模型与数据

项目通过 uv 管理环境，使用 Python 3.11、PyTorch 2.6.0。先执行 `uv sync`，再参考 [.env.example](../../../../.env.example) 完善各服务器的 `.env`。

| 内容 | 默认位置 | 覆盖参数 |
|---|---|---|
| Qwen3-4B-Instruct-2507 | `~/models/Qwen3-4B-Instruct-2507` | `--model-path` |
| AE／LM 训练数据 | `data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/` | `--pretrain-data` |
| QA 训练数据 | `data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008/` | `--qa-data` |
| QA 评估数据 | 默认同上 | `--eval-data` |

数据目录须保留 `preparation.json` 与 `train/dev/test.jsonl`。数据集详细格式见 [Fineweb Multisegment](../../data_preparation/fineweb_multisegment/README.md) 和 [Fineweb FactQA](../../data_preparation/fineweb_factqa/README.md)，可从 [ModelScope](https://modelscope.cn/datasets/percyWeeei/latent-working-memory/files) 下载。当前 FactQA 的完整题池格式已同步至云端，不保存逐条轨迹的 `usage`；使用此前下载的数据时须先删除该字段，其他字段及 API 用量统计保持原样。

默认记录到 SwanLab 项目 `latent-working-memory-v3`，使用项目 `.env` 中的 key。加 `--tracking disabled` 仅保存本地产物，无需 SwanLab key。

## 2. 运行

基础指令为：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --mode full --method icae_single --gpus 0,1 \
  --run-id capacity-comparison_20261010-01
```

- `--mode` 支持冒烟运行（`smoke`）和正式运行（`full`）。

  | 档位 | 每阶段训练／开发集样本上限 | 每阶段 steps 上限 | 初次最终评估默认值 |
  |---|---|---|---|
  | `smoke` | 16／4 | 2 | dev，最多 2 条轨迹 |
  | `full`（默认） | 按阶段配置／全部开发集 | 默认 1 epoch | 完整 test |

- 新训练加 `--train-only` 时，保留训练中的 dev NLL 验证，跳过训练结束后的生成评估。该参数不能与 `--resume` 或 `--evaluate-only` 同用。
- `--run-id` 默认使用格式为 `YYYYMMDD-HHMMSS` 的当前时间。动态后训练复用预训练 checkpoint 时，省略该参数则继承来源编号。

### 2.1 完整运行

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

完成后，从 `artifacts/v3/capacity-comparison_20261010-01/train/dynamic-pretrain-k512/pretrain/training-result.json` 的 `checkpoint` 字段取得实际目录，替换下例的 `global_step_N`：

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
- `--writer-mode` 仅用于动态后训练，可选 `local`、`tag`、`mask`、`dual_lora`；使用 `tag` 时可加 `--tag-tokens 3`。动态预训练不使用此参数。
- 同系列后训练可省略 `--run-id`，自动继承 checkpoint 来源预训练的编号。
- 若要执行完整训练流程，应该在配置中声明 `"stages": ["pretrain", "warmup", "policy"]`，并为 `training.stage_max_train_samples` 增加 `"pretrain": 12800`，然后通过 `--config` 启动。

### 2.2 单独评估

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --evaluate-only --gpus 0 \
  --checkpoint artifacts/v3/capacity-comparison_20261010-01/train/information-loss-k512+32-local/policy/checkpoints/global_step_N \
  --eval-data data/fineweb-factqa-evaluation \
  --eval-split test \
  --eval-seed 20261004 \
  --eval-trajectories 0 \
  --max-new-tokens 64
```

- `--evaluate-only` 和 `--checkpoint` 是基础必须参数，默认复用原配置文件 `plan/<method-dir>/eval.json`，模型、方法和写入配置等则从 checkpoint 读取。
- 显式传入 `--eval-data`、`--eval-split`、`--eval-seed`、`--eval-gate-qa-count`、`--eval-trajectories` 或 `--max-new-tokens` 则会更新旧评估配置并覆盖已有评估结果 `eval/<method-dir>/<stage>/` 和 `plan/<method-dir>/eval.log`（如果有的话）。省略 seed 时复用 `eval.json`；该 seed 同时控制轨迹和 gate 选题。
- 当旧产物缺少配置文件时，需显式传入 `--eval-data <目录>` 以建立，其余参数默认为使用 test split、全部轨迹、seed 20261004、8 道 gate QA 以及最多生成 64 tokens。
- 默认追加到原训练 SwanLab run，须保留原训练目录与身份文件且云端 run 状态为 `FINISHED`；加 `--tracking disabled` 仅保存本地结果。

### 2.3 中断恢复

从中断阶段的 checkpoint 恢复当前方法，随后完成未启动的后续阶段和最终评估；动态预训练仅完成预训练：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --resume --gpus 0,1 \
  --checkpoint artifacts/v3/capacity-comparison_20261010-01/train/information-loss-k512+32-local/warmup/checkpoints/global_step_N
```

- 保留原 `plan/<method-dir>/job.json` 和阶段 `config.json`，恢复沿用原 `run-id`、训练配置、预算与 SwanLab 身份，不应用 CLI 训练超参数覆盖。
- 最终评估沿用 `eval.json`，可通过上述评估参数独立调整。
- 可重新选择物理 GPU，卡数须与原任务一致。

## 3. 复用动态预训练

- **标识**：同系列后训练省略 `--run-id` 时继承 checkpoint 来源预训练的编号，跨系列复用时显式指定新的 `--run-id`。`--group` 独立指定，默认采用当前 `run-id`。
- **来源**：SwanLab 的 `pretraining` 配置与本地 `run.json` 记录来源编号、名称、URL、checkpoint 和 step；保留原预训练路径、`pretrain/run.json` 及在线运行的 `swanlab.json`。
- **使用范围**：动态预训练本身不执行追加／覆盖，因此同一预训练 checkpoint 可用于不同 **追加容量**、**写入方式** 或 **扩容方式** 的后训练。更换 **基座模型** 或 **首次容量 K** 时则需要重新进行预训练。

## 4. 调整参数

### 4.1 配置与覆盖

| 参数 | 用途／默认值 |
|---|---|
| `--config` | 所选方法的 JSON 配置文件，默认读取 `configs/v3/` 下的默认配置 |
| `--model-path` | 基座模型位置，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--pretrain-data` | AE／LM 训练数据目录，默认位置见[第 1 节](#1-准备环境模型与数据) |
| `--qa-data` | QA 训练数据目录，默认位置见[第 1 节](#1-准备环境模型与数据) |

- 配置文件中的 `objective.method` 对应所选方法，`objective.stages` 指定训练阶段顺序，`training.stage_max_train_samples` 指定各阶段样本上限。需要注意的是，动态预训练（`--method dynamic_pretrain`）配置中的 method／stages 应分别为 `dynamic`／`["pretrain"]`。
- 完整参数见 `bash src/latent_working_memory/v3/scripts/run.sh --help`。

### 4.2 批处理、目标与动态写入

| 参数 | 用途／默认值 |
|---|---|
| `--micro-batch-size-per-gpu` | 每张 GPU 一次并行处理的样本／轨迹数，默认 4 |
| `--gradient-accumulation-steps` | 每次优化器更新前累积的 microbatch 数，默认 2 |
| `--qa-batch-size` | 每条轨迹每次并行读取的题数上限，默认 8；超出的题分批读取 |
| `--task-qa-count` | 每段训练任务题总数，默认 8；首段只计算其中的新题 |
| `--task-new-ratio` | 训练任务题中新题比例，默认 0.5；总题数乘比例须为正整数 |
| `--gate-qa-count` | 训练与 dev 每次门控从全部历史 gate 池抽取的题数，默认 8 |
| `--epochs` | 每阶段训练轮数，默认 1 |
| `--lm-ratio` | AE／LM 预训练选择 LM 目标的概率，默认 0.5；AutoCompressors 始终使用 LM |
| `--lm-target-tokens` | 预训练 LM 续文目标的 token 数，默认 512 |
| `--icae-min-segments` | ICAE-multi 随机分块数下限，默认 3 |
| `--icae-max-segments` | ICAE-multi 随机分块数上限，默认 6 |
| `--ac-num-segments` | AutoCompressors 的正文分块数，默认 4 |
| `--bptt-steps` | 每个反向传播窗口包含的写入轮数；AutoCompressors 默认 2，动态后训练默认完整 BPTT，`0` 表示完整 BPTT |
| `--append-slots` | 动态后训练每次追加的 slots 数，默认 32；首次写入使用 K slots，覆盖保持末块大小 |
| `--writer-mode` | 动态后训练的写入方式，可选 `local`、`tag`、`mask`、`dual_lora`，默认 `local` |
| `--tag-tokens` | 每种写入动作的控制 token 数，默认 3；仅用于 `tag` |
| `--threshold-i` | `memory_change` 的记忆变化阈值，默认 0.1；变化分数 I 达到阈值时追加 |
| `--threshold-d` | `information_loss` 的旧信息损伤阈值，默认 0.1 |
| `--threshold-g` | `information_loss` 的追加收益阈值，默认 0.1 |
| `--eta` | `information_loss` 在旧信息损伤超限时要求的最低追加收益，默认 0；须小于 `--threshold-g` |

全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数，双卡默认 16。BPTT（跨写入轮次的反向传播）、分块及动态策略的具体规则见 [v3 说明](../README.md)。

训练 task 与 gate 选题随 epoch 确定性变化，dev/test 固定 gate 并使用全部已读 task。加载时仅对 train 检查每段的新题池和首段旧题池；各 split 均检查首段 gate 池。不足则排除整条轨迹，每个 split 保留不足原规模的 80% 时报错。该检查先于长度过滤和样本上限选择，不减少题数或重复补采。

### 4.3 长度、运行预算与最终评估

下表中的训练默认值以正式运行（`full`）及仓库预设为准；`smoke` 的额外限制见[第 2 节](#2-运行)。已有 `eval.json` 时，最终评估沿用存档值，显式评估参数会更新该配置。

| 参数 | 用途／默认值 |
|---|---|
| `--max-input-tokens` | AE／LM 训练正文上限，默认 12288 |
| `--max-qa-input-tokens` | QA 训练正文上限，默认 12288 |
| `--pretrain-train-samples` | 预训练样本上限，默认 12800 |
| `--qa-train-samples`、`--warmup-train-samples`、`--policy-train-samples` | 对应后训练阶段的独立样本上限，默认全部 |
| `--dev-samples` | 每阶段 dev 样本上限；`full` 默认全部，`smoke` 默认 4 |
| `--max-steps` | 每阶段优化器更新步数上限；`full` 默认不限，`smoke` 默认 2 |
| `--eval-every` | dev 验证的 optimizer step 间隔，`full` 默认 25，`smoke` 默认 1；epoch 末也验证 |
| `--save-every` | 正整数 optimizer step 保存间隔或 `after_each_epoch`；`full` 沿用配置（预设 25），`smoke` 默认 1；epoch 末也保存 |
| `--save-total-limit` | 每阶段 checkpoint 保留数，默认 2 |
| `--group` | SwanLab 实验分组，默认当前 `run-id` |
| `--output-root` | 产物父目录，默认 `artifacts/v3` |
| `--eval-data` | 最终 QA 评估数据目录，首次训练时默认采用 `--qa-data` |
| `--eval-split` | 最终评估划分；首次训练时 `full` 默认 `test`，`smoke` 默认 `dev` |
| `--eval-trajectories` | 最终评估轨迹上限；首次训练时 `full` 默认全部，`smoke` 默认 2 |
| `--eval-seed` | 最终评估轨迹及 gate 选题 seed，初次默认 20261004；已有 `eval.json` 时复用存档值 |
| `--eval-gate-qa-count` | 最终评估每次门控题数；新训练初次沿用训练配置，后续沿用 `eval.json` |
| `--max-new-tokens` | 最终评估每题生成的 token 上限，初次默认 64 |

- **长度限制**：仅过滤 train，超限样本整条排除；长度只计正文，不含记忆、提示和目标。设为 `0`（配置中为 `null`）时使用模型窗口上限，dev／test 不按训练长度上限过滤。
- **样本与步数限制**：各训练阶段按训练 seed 独立选择样本。阶段样本参数设为 `0` 时解除该阶段预算，但 `smoke` 仍保留档位上限；`--dev-samples`、`--max-steps` 设为 `0` 时解除各自的档位上限。`--eval-trajectories 0` 表示评估完整 split。
- **按 epoch 保存**：使用 `--save-every after_each_epoch`，或在 JSON 的 `training` 中设置 `"save_every": "after_each_epoch"`。训练数据完成过滤、抽样和 batch 划分后，按真实 dataloader 长度确定保存间隔，与 verl 的 `save_freq: after_each_epoch` 语义相同。若训练 3 epochs 并保留每轮末的 checkpoint，使用 `--epochs 3 --save-every after_each_epoch --save-total-limit 3`。`--max-steps` 主动停止时也会保存，其产物同样计入保留数量。

## 5. 查看结果

产物位于 `artifacts/v3/<run-id>/`：

```text
artifacts/v3/<run-id>/
├── plan/<method-dir>/             阶段配置、eval.json、job.json、日志、result.json
├── train/<method-dir>/<stage>/    config.json、metrics.jsonl、dev-qa.jsonl、checkpoints/
├── eval/<method-dir>/<最终阶段>/   summary.json、trajectories.jsonl
└── compare/                       独立比较生成的 points.json、points.csv
```

| 方法 | `<method-dir>` |
|---|---|
| 三个 baseline 和独立共享预训练 | `<method>-k<K>` |
| 两种动态方法（含显式完整流程） | `<method>-k<K>+<ΔK>-<writer>` |

方法名将 `_` 转为 `-`；独立共享预训练用 `dynamic-pretrain`，完整动态流程的各阶段保存在同一方法目录。writer 为 `local`、`tag`、`mask` 或 `dual-lora`。试跑在末尾添加 `_<mode>`。SwanLab 名称为 `<method-dir>_<run-id>`；跨系列复用使用当前系列编号，预训练来源通过 `pretraining` 配置追溯。

- **进度**：终端每 step 一行摘要，完整指标见 `train/` 下的 `metrics.jsonl`。SwanLab 同一方法的训练、验证、评估共用一个 run；阶段与进度看 `train/stage`、`train/epoch`，最终结果看柱状图。
- **评估与比较**：逐条结果见 `eval/`；不同方法的结果通过独立比较命令汇总。比较要求 split、实际 task 题池、各段选中的 gate QA 签名、QA prompt 和生成 token 上限一致；各方法的训练数据与训练配置可不同。指标含义见 [v3 说明](../README.md#五评估与产物)。
- **错误**：先看对应方法 `plan/` 下的 `result.json`，再看 `train.log`／`eval.log`；独立比较错误直接在终端报告。
- **Checkpoint 与恢复**：每阶段默认保留最近两个 `checkpoints/global_step_<阶段step>/` 目录；初始化、恢复和评估均通过 `--checkpoint` 传目录路径。中断恢复见[第 2.3 节](#23-中断恢复)。

例如，两种动态方法评估完成后，独立导出比较结果：

```bash
uv run --frozen python -m latent_working_memory.v3.compare \
  artifacts/v3/capacity-comparison_20261010-01/eval/memory-change-k512+32-local/policy/summary.json \
  artifacts/v3/capacity-comparison_20261010-01/eval/information-loss-k512+32-local/policy/summary.json \
  --output-dir artifacts/v3/capacity-comparison_20261010-01/compare
```

新试跑与正式训练使用不同 `run-id`；复用预训练时可继承来源编号，或显式指定新系列编号。已有运行目录不覆盖。
