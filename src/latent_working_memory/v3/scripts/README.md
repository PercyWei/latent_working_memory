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

数据目录须保留 `preparation.json` 与 `train/dev/test.jsonl`。数据集详细格式见 [FineWeb Multisegment](../../data_preparation/fineweb_multisegment/README.md) 和 [FineWeb FactQA](../../data_preparation/fineweb_factqa/README.md)，可从 [ModelScope](https://modelscope.cn/datasets/percyWeeei/latent-working-memory/files) 下载。

默认记录到 SwanLab 项目 `latent-working-memory-v3`，使用项目 `.env` 中的 key。加 `--tracking disabled` 仅保存本地产物，无需 SwanLab key。

## 2. 运行

基础指令为：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --mode full --method icae_single --gpus 0,1 \
  --run-id capacity-comparison_20261010-01
```

- `--mode` 支持试跑（`smoke`）和正式运行（`full`）。

  | 档位 | 每阶段训练样本上限 | 每阶段 steps 上限 | dev / test 数据上限 |
  |---|---|---|---|
  | `smoke` | 16 | 2 | 4 / 2 |
  | `full`（默认） | 不同阶段默认配置不同 | 默认 1 epoch | 默认全部 / 全部 |

- `--max-dev-samples` 和 `--max-test-samples` 分别控制 dev 和 test 评估，均接受正整数或 `null`（默认，表示全部）。
- 新训练加 `--train-only` 时会跳过训练结束后的 QA 评估，其不能与 `--resume` 或 `--evaluate-only` 同用。
- `--run-id` 默认使用格式为 `YYYYMMDD-HHMMSS` 的当前时间。动态后训练复用预训练 checkpoint 时，省略该参数则继承来源编号。

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
- `--gpus 0,1` 为每张物理卡分配一个固定计算槽位，各 checkpoint 并行计算；全部计算结束后，由父进程汇总结果，串行上传完整报告，checkpoint 按真实累计 step 排列。计算子进程退出后释放 GPU，缓存结果和上传不占用 GPU。训练时该参数仍指定参与训练的 GPU。
- 同一方法 run 同时只运行一个评估调度任务。所有 checkpoint 共用原评估配置与 seed，GPU 分配不改变选题。
- 默认原样复用 `plan/<method-dir>/eval.json`，模型、方法和写入配置从 checkpoint 读取。首次评估前可用 `--eval-data`、`--eval-seed`、`--max-test-samples`、`--max-new-tokens` 设置评估条件；已有评估历史后，同一 run 须保持相同 test 配置与 QA 协议。
- 每个 checkpoint 的结果独立保存。相同 checkpoint、相同配置复用已有结果和日志；在线发布将本次选中的结果与此前已发布的本地结果合并为完整报告，其他离线结果须明确选择后才发布。内容相同时仅检查、修复云端布局；需要更新时从缓存重建，补发日志追加到原 `eval.log`，不重新计算 QA，也无需保留此前 checkpoint 的权重。
- 单个 checkpoint 计算失败不影响其他结果保存，报告上传失败也保留全部本地结果；最终汇总失败并以非零状态退出，重新执行原命令复用已完成的本地结果。
- 保持参数不变重新执行上述命令，可从本地结果更新展示；需要转换布局时上传一次完整报告。删除云端 `evaluation` 分组后也可用同一命令恢复图表和统计表；仅修复布局时复用已上传的媒体版本，不重复追加版本。
- 默认追加到原训练 SwanLab run，须保留原训练目录与身份文件且云端 run 状态为 `FINISHED`；加 `--tracking disabled` 仅保存本地结果。

### 2.3 中断恢复

从中断阶段的 checkpoint 恢复当前方法，随后完成未启动的后续阶段和最终评估；动态预训练仅完成预训练：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --resume --gpus 0,1 \
  --checkpoint artifacts/v3/capacity-comparison_20261010-01/train/information-loss-k512+32-local/warmup/checkpoints/global_step_N
```

- 保留原 `plan/<method-dir>/job.json` 和阶段 `config.json`，恢复沿用原 `run-id`、训练配置、预算与 SwanLab 身份，不应用 CLI 训练超参数覆盖。
- 最终评估沿用 `eval.json`；已有评估历史时须保持同一 test 配置与 QA 协议，配置冲突会在恢复训练前报错。
- 可重新选择物理 GPU，卡数须与原任务一致。

## 3. 复用动态预训练

- **标识**：同系列后训练省略 `--run-id` 时继承 checkpoint 来源预训练的编号，跨系列复用时显式指定新的 `--run-id`。`--group` 独立指定，默认采用当前 `run-id`。
- **来源**：SwanLab 的 `pretraining` 配置与本地 `run.json` 记录来源编号、名称、URL、checkpoint 和 step；保留原预训练路径、`pretrain/run.json` 及在线运行的 `swanlab.json`。
- **使用范围**：动态预训练本身不执行追加／覆盖，因此同一预训练 checkpoint 可用于不同 **追加容量**、**写入方式** 或 **扩容方式** 的后训练。更换 **基座模型** 或 **首次容量 K** 时则需要重新进行预训练。

## 4. 调整参数

以下默认值以正式运行（`full`）和仓库预设为准，`smoke` 的额外限制见[第 2 节](#2-运行)。新训练的显式 CLI 超参数覆盖方法配置；中断恢复沿用保存的训练配置，见[第 2.3 节](#23-中断恢复)。完整参数可通过 `bash src/latent_working_memory/v3/scripts/run.sh --help` 查看。

`--train-seed`、`--objective-seed` 和 `--eval-seed` 相互独立，均接受非负整数（0 有效），不接受 `null`。

### 4.1 训练输入与执行预算

**配置与模型**

| 参数 | 用途／默认值 |
|---|---|
| `--config` | 所选方法的 JSON 配置文件，默认读取 `configs/v3/` 下的预设 |
| `--model-path` | 基座模型位置，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--train-seed` | `training.seed`：控制初始化、train/dev 样本选择、数据打乱及 AE／LM 目标选择；省略时沿用配置，预设为 20261004 |

配置中的 `objective.method` 须与所选方法对应，`objective.stages` 指定训练阶段顺序。独立动态预训练（`--method dynamic_pretrain`）的 method／stages 分别为 `dynamic`／`["pretrain"]`。

**训练数据与长度**

| 参数 | 用途／默认值 |
|---|---|
| `--pretrain-data` | AE／LM 训练数据目录，默认位置见[第 1 节](#1-准备环境模型与数据) |
| `--max-input-tokens` | AE／LM 训练正文上限，默认 12288 |
| `--qa-data` | QA 训练数据目录，默认位置见[第 1 节](#1-准备环境模型与数据) |
| `--max-qa-input-tokens` | QA 训练正文上限，默认 12288 |

长度筛选仅用于 train 正文，不含记忆、提示和目标。两个参数均接受正整数或 `null`：正整数仅按给定值筛选，超限样本整条排除；`null` 跳过长度上限筛选。dev/test 不按该上限过滤，实际每次编码／解码仍须满足模型窗口限制。

**批处理**

| 参数 | 用途／默认值 |
|---|---|
| `--micro-batch-size-per-gpu` | 每张 GPU 一次并行处理的样本／轨迹数，默认 4 |
| `--gradient-accumulation-steps` | 每次优化器更新前累积的 microbatch 数，默认 2 |
| `--qa-batch-size` | 每条轨迹每次并行读取的题数上限，默认 8 |

全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数，双卡默认 16。`--qa-batch-size` 控制读取并行量，不改变选题数量。

**训练规模与时长**

| 参数 | 用途／默认值 |
|---|---|
| `--pretrain-train-samples` | 预训练样本上限，默认 12800 |
| `--qa-train-samples`、`--warmup-train-samples`、`--policy-train-samples` | 对应不同后训练阶段的独立样本上限，默认 `null`（全部） |
| `--epochs` | 每阶段训练轮数，默认 1 |
| `--max-steps` | 每阶段优化器更新步数上限，`full` 默认 `null`（完整 epoch），`smoke` 默认 2 |

各阶段样本预算由配置中的 `training.stage_max_train_samples` 指定，按训练 seed 选取。上述样本与步数上限接受正整数或 `null`。

### 4.2 训练目标与压缩策略

| 参数 | 用途／默认值 |
|---|---|
| `--objective-seed` | `objective.seed`：控制随机分块、warmup 动作、训练 task/gate 选题及 dev gate 选题；省略时沿用配置，预设为 20261004 |

**AE／LM 目标**

| 参数 | 用途／默认值 |
|---|---|
| `--lm-ratio` | AE／LM 预训练选择 LM 目标的概率，默认 0.5；AutoCompressors 始终使用 LM |
| `--lm-target-tokens` | 预训练 LM 续文目标的 token 数，默认 512 |

**QA 选题**

| 参数 | 用途／默认值 |
|---|---|
| `--task-qa-count` | 训练时每次选取的 task QA 总数，默认 8；首段仅使用其中的新题 |
| `--task-new-ratio` | 训练任务题中新题比例，默认 0.5；总题数乘比例须为正整数 |
| `--gate-qa-count` | 每次门控时选取的 gate QA 总数，默认 8；首段不进行门控 |

选题、跨 epoch 随机性与题池数量检查见 [v3 数据说明](../README.md#31-数据输入与通用设置)。

**分块与反向传播**

| 参数 | 用途／默认值 |
|---|---|
| `--icae-min-segments` | ICAE-multi 随机分块数下限，默认 3 |
| `--icae-max-segments` | ICAE-multi 随机分块数上限，默认 6 |
| `--ac-num-segments` | AutoCompressors 的正文分块数，默认 4 |
| `--bptt-steps` | 每个反向传播窗口包含的写入轮数，AutoCompressors 默认 2，动态后训练默认 `null`（完整 BPTT） |

BPTT 表示跨写入轮次的反向传播；AutoCompressors 还用该窗口组织随机分块。分块与训练目标的具体规则见 [v3 方法说明](../README.md#三数据与训练)。

**动态写入与门控**

以下写入配置用于动态后训练，门控阈值按方法区分。

| 参数 | 用途／默认值 |
|---|---|
| `--append-slots` | 动态后训练每次追加的 slots 数，默认 32；首次写入使用 K slots，覆盖保持末块大小 |
| `--writer-mode` | 动态后训练的写入方式，可选 `local`、`tag`、`mask`、`dual_lora`，默认 `local` |
| `--tag-tokens` | 每种写入动作的控制 token 数，默认 3；仅用于 `tag` |
| `--threshold-i` | `memory_change` 的记忆变化阈值，默认 0.1；变化分数 I 达到阈值时追加 |
| `--threshold-d` | `information_loss` 的旧信息损伤阈值，默认 0.1 |
| `--threshold-g` | `information_loss` 的追加收益阈值，默认 0.1 |
| `--eta` | `information_loss` 在旧信息损伤超限时要求的最低追加收益，默认 0；须小于 `--threshold-g` |

### 4.3 验证、测试与结果记录

**训练中的 dev 验证**

| 参数 | 用途／默认值 |
|---|---|
| `--max-dev-samples` | 训练中 dev 验证的样本上限，默认 `null`（全部），`smoke` 默认 4 |
| `--dev-every` | dev 验证的 optimizer step 间隔，`full` 默认 25，`smoke` 默认 1；除此之外，epoch 末会自动验证 |

**Checkpoint 保存**

| 参数 | 用途／默认值 |
|---|---|
| `--save-every` | checkpoint 保存的正整数 optimizer step 间隔或 `after_each_epoch`；`full` 默认 25，`smoke` 默认 1；除此之外，epoch 末会自动保存 |
| `--save-total-limit` | 保留最近成功保存的 checkpoint 数，默认 2 |

例如，warmup 与 policy 各训练 3 epochs，设置 `--save-every after_each_epoch --save-total-limit 6`，最终保留所有六个 epoch 末 checkpoint。

**最终 test 评估**

| 参数 | 用途／默认值 |
|---|---|
| `--eval-data` | 最终 QA 评估数据目录，新训练首次默认采用 `--qa-data` |
| `--eval-seed` | 控制 test 子集及 gate 选题；省略时沿用 `eval.json`，初次默认 20261004 |
| `--max-test-samples` | 最终 test 评估的样本上限，默认 `null`（全部），`smoke` 默认 2 |
| `--max-new-tokens` | 最终评估每题生成的 token 上限，初次默认 64 |

以上四项保存于 `plan/<method-dir>/eval.json`，每个 checkpoint 的结果目录另存本次快照。正常训练、续训与单独评估默认复用该配置；已有评估历史后，显式参数须与原设置一致。方法配置从 checkpoint 读取。指标与计算场景见 [v3 评估说明](../README.md#51-评估方式与指标)。

**产物与分组**

| 参数 | 用途／默认值 |
|---|---|
| `--output-root` | 产物父目录，默认 `artifacts/v3` |
| `--group` | SwanLab 实验分组，默认当前 `run-id` |

## 5. 查看结果

产物位于 `artifacts/v3/<run-id>/`：

```text
artifacts/v3/<run-id>/
├── plan/<method-dir>/             阶段配置、eval.json、job.json、训练日志、result.json
├── train/<method-dir>/<stage>/    config.json、metrics.jsonl、dev-qa.jsonl、checkpoints/
├── eval/<method-dir>/
│   ├── evaluations.json           checkpoint 评估历史与累计报告发布记录
│   └── <stage>/global_step_<阶段step>/
│       ├── eval.json              本次四项评估配置快照
│       ├── summary.json、trajectories.jsonl、eval.log
│       └── swanlab-publication.json
└── compare/                       独立比较生成的 points.json、points.csv
```

| 方法 | `<method-dir>` |
|---|---|
| 三个 baseline 和独立共享预训练 | `<method>-k<K>` |
| 两种动态方法（含显式完整流程） | `<method>-k<K>+<ΔK>-<writer>` |

方法名将 `_` 转为 `-`；独立共享预训练用 `dynamic-pretrain`，完整动态流程的各阶段保存在同一方法目录。writer 为 `local`、`tag`、`mask` 或 `dual-lora`。试跑在末尾添加 `_<mode>`。SwanLab 名称为 `<method-dir>_<run-id>`；跨系列复用使用当前系列编号，预训练来源通过 `pretraining` 配置追溯。

- **进度与图表**：终端每 step 一行摘要，完整指标见 `train/` 下的 `metrics.jsonl`。SwanLab 同一方法的训练、验证、评估共用一个 run；阶段与进度看 `train/stage`、`train/epoch`。`evaluation` 展示 NLL、EM、F1 三张合图，横轴为 QA 首次引入段及“总体”，每个 checkpoint 一条带点折线；容量与耗时汇入一张统计表，动态方法的各 checkpoint 热图排在最后。图例、表格和热图按真实累计 step 排序。最新发布版本包含全部已发布 checkpoint，底部滑块表示报告版本。详细布局见 [v3 说明](../README.md#53-swanlab-组织与展示)。
- **评估与比较**：逐条结果见 `eval/`；不同方法的结果通过独立比较命令汇总。比较要求 split、实际 task 题池、各段选中的 gate QA 签名、QA prompt 和生成 token 上限一致；各方法的训练数据与训练配置可不同。指标含义见 [v3 说明](../README.md#五评估与产物)。
- **错误**：训练查看方法 `plan/` 下的 `result.json`、`train.log`；评估查看对应 checkpoint 结果目录的 `eval.log`。发布失败保留本地结果，重新执行原命令即可补传；追加要求云端原训练 run 为 `FINISHED`。独立比较错误直接在终端报告。
- **Checkpoint 与恢复**：checkpoint 保存目录为 `<stage>/checkpoints/global_step_<阶段step>/`，`result.json` 仅记录仍保留的 checkpoint。初始化、恢复和评估均通过 `--checkpoint` 传目录路径。中断恢复见[第 2.3 节](#23-中断恢复)。

例如，两种动态方法评估完成后，独立导出比较结果：

```bash
uv run --frozen python -m latent_working_memory.v3.compare \
  artifacts/v3/capacity-comparison_20261010-01/eval/memory-change-k512+32-local/policy/global_step_N/summary.json \
  artifacts/v3/capacity-comparison_20261010-01/eval/information-loss-k512+32-local/policy/global_step_N/summary.json \
  --output-dir artifacts/v3/capacity-comparison_20261010-01/compare
```

新试跑与正式训练使用不同 `run-id`；复用预训练时可继承来源编号，或显式指定新系列编号。已有运行目录不覆盖。
