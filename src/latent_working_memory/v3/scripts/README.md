# 运行 v3 实验

使用 [run.sh](run.sh) 完成训练、评估与方法比较。默认 Qwen3-4B、K=512、ΔK=32；K 为 baseline 总容量或动态方法首次容量，ΔK 为动态追加容量。方法与训练细节见 [v3 说明](../README.md)，默认配置位于 [configs/v3](../../../../configs/v3/)。

## 1. 准备环境、模型与数据

项目通过 uv 管理环境，使用 Python 3.11、PyTorch 2.6.0。先执行 `uv sync`，再参考 [.env.example](../../../../.env.example) 完善各服务器的 `.env`。

| 内容 | 默认位置 | 覆盖参数 |
|---|---|---|
| Qwen3-4B-Instruct-2507 | `~/models/Qwen3-4B-Instruct-2507` | `--model-path` |
| AE／LM 数据 | `data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/` | `--pretrain-data` |
| QA 数据 | `data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008/` | `--qa-data` |

数据目录须保留 `preparation.json` 与 `train/dev/test.jsonl`。数据集详细格式见 [Fineweb Multisegment](../../data_preparation/fineweb_multisegment/README.md) 和 [Fineweb FactQA](../../data_preparation/fineweb_factqa/README.md)，均可从 [ModelScope](https://modelscope.cn/datasets/percyWeeei/latent-working-memory/files) 下载；

默认记录到 SwanLab 项目 `latent-working-memory-v3`，使用项目 `.env` 中的 key。加 `--tracking disabled` 仅保存本地产物，无需 SwanLab key。

## 2. 试跑与正式运行

GPU 编号按实际分配修改；最终评估使用第一张卡。一次完成五种方法：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --mode full --method all --gpus 0,1 \
  --run-id capacity-comparison_20261010-01
```

- `--method all` 默认运行三组 baseline、一次动态预训练和两组动态后训练，支持方法单独运行。
- `--run-id` 默认使用格式为 `YYYYMMDD-HHMMSS` 的当前时间，动态后训练在省略该参数时使用 checkpoint 来源（动态预训练）的编号。

| 档位 | 每阶段训练／开发集样本上限 | 每阶段 steps 上限 | 最终评估 |
|---|---|---|---|
| `smoke` | 16／4 | 2 | dev，最多 2 条轨迹 |
| `full`（默认） | 按阶段配置／全部开发集 | 默认 1 epoch | 完整 test |

先用 `--mode smoke` 检查显存、日志和 SwanLab，再用 `full` 正式训练。

### 2.1 常用指令

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

先运行一次动态预训练：

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
  --init-checkpoint artifacts/v3/capacity-comparison_20261010-01/train/dynamic-pretrain-k512/pretrain/checkpoints/global_step_N
```

- `--method` 可以设为 `memory_change`、`information_loss`。
- `--writer-mode` 仅用于动态后训练，可选 `local`、`tag`、`mask`、`dual_lora`；使用 `tag` 时可加 `--tag-tokens 3`。动态预训练不使用此参数。
- 同系列后训练可省略 `--run-id`，自动继承 checkpoint 来源预训练的编号。

### 2.2 单独补跑评估

从最终训练阶段的 `training-result.json` 中取得 `checkpoint` 目录：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --evaluate-only --gpus 0 \
  --checkpoint artifacts/v3/capacity-comparison_20261010-01/train/information-loss-k512+32-local/policy/checkpoints/global_step_N
```

- 模型、方法和写入配置从 checkpoint 读取；只需 QA 数据，位置不同时传 `--qa-data <目录>`。
- 默认评估完整 `test`，不受 `--mode` 影响；可用 `--eval-split dev`、`--eval-trajectories <数量>`、`--max-new-tokens <数量>` 调整。
- 使用原规范目录下的 checkpoint，更新原系列的 `eval/<method-dir>/<stage>/` 结果与 `plan/<method-dir>/eval.log`。
- 默认追加到原训练 SwanLab run，须保留原训练目录与身份文件且云端 run 状态为 `FINISHED`；加 `--tracking disabled` 仅保存本地结果。

## 3. 复用动态预训练

- **标识**：同系列后训练省略 `--run-id` 时继承 checkpoint 来源预训练的编号，跨系列复用时显式指定新的 `--run-id`。`--group` 独立指定，默认采用当前 `run-id`。
- **来源**：SwanLab 的 `pretraining` 配置与本地 `run.json` 记录来源编号、名称、URL、checkpoint 和 step；保留原预训练路径、`pretrain/run.json` 及在线运行的 `swanlab.json`。
- **使用范围**：动态预训练本身不执行追加／覆盖，因此同一预训练 checkpoint 可用于不同 **追加容量**、**写入方式** 或 **扩容方式** 的后训练。更换 **基座模型** 或 **首次容量 K** 时则需要重新进行预训练。

## 4. 调整参数

### 4.1 配置与覆盖

- 未指定 `--config` 时读取 `configs/v3/`；`--config <文件>` 仅用于单方法，不能与 `--method all` 组合。
- 文件中的 `objective.method` 须与方法一致，`objective.stages` 声明阶段顺序。`dynamic_pretrain` 使用 `"method": "dynamic"` 和 `["pretrain"]`。
- 显式 CLI 参数覆盖文件值；动态自定义配置只替换后训练，自动预训练仍读取独立配置。
- `training.stage_max_train_samples` 分阶段保存预算；启动器展开后在 `plan/` 保存各阶段配置。训练与方法细节见 [v3 说明](../README.md)。

完整参数通过 `bash src/latent_working_memory/v3/scripts/run.sh --help` 查看。

### 4.2 批处理、目标与动态写入

| 参数 | 用途／默认值 |
|---|---|
| `--micro-batch-size-per-gpu`、`--gradient-accumulation-steps` | 每卡并行样本数、梯度累积次数；默认均为 4／2 |
| `--qa-batch-size 8` | 每条轨迹一次读取的题数 |
| `--epochs` | 各阶段训练轮数，默认 1 |
| `--lm-ratio 0.5`、`--lm-target-tokens 512` | AE／LM 预训练选择 LM 的概率、各预训练方法的续文目标长度；AutoCompressors 使用 LM |
| `--icae-min-segments 3`、`--icae-max-segments 6` | ICAE-multi 的随机分块数范围 |
| `--ac-num-segments 4` | AutoCompressors 的正文分块数 |
| `--bptt-steps` | 写入梯度窗口；AutoCompressors 默认 2，动态后训练默认完整，`0` 表示完整 BPTT |
| `--append-slots 32` | 动态后训练每次追加容量；首次 K，覆盖保持末块大小 |
| `--writer-mode local` | 动态后训练使用 `local`、`tag`、`mask` 或 `dual_lora` |
| `--tag-tokens 3` | 仅用于 `tag`，每种动作的控制 token 数 |
| `--threshold-i`、`--threshold-d`、`--threshold-g`、`--eta` | 动态策略阈值 |

全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数。写入方式、分块和 BPTT 细节见 [v3 说明](../README.md)。

### 4.3 长度与运行预算

| 参数 | 用途／默认值 |
|---|---|
| `--max-input-tokens 12288` | AE／LM 训练正文上限 |
| `--max-qa-input-tokens 12288` | FactQA 训练正文上限 |
| `--pretrain-train-samples` | 预训练样本上限，默认 12800 |
| `--qa-train-samples`、`--warmup-train-samples`、`--policy-train-samples` | 对应阶段的独立样本上限，默认全部 |
| `--dev-samples`、`--max-steps`、`--eval-trajectories` | 开发集、每阶段步数和最终评估轨迹上限 |
| `--save-total-limit 2` | 每阶段 checkpoint 保留数 |
| `--group`、`--output-root` | SwanLab group 可独立指定，默认当前 `run-id`；产物父目录默认 `artifacts/v3` |

- **长度**：只过滤训练集，超限整条排除；上限仅计正文，不含记忆、提示和目标。`0`（配置 `null`）使用模型窗口；dev／test 不按该上限过滤。
- **数量**：各阶段独立按 seed 选择样本。阶段样本参数的 `0` 解除阶段预算，试跑仍受档位上限限制；`--dev-samples`／`--max-steps`／`--eval-trajectories` 的 `0` 解除对应档位上限。

## 5. 查看结果

产物位于 `artifacts/v3/<run-id>/`：

```text
artifacts/v3/<run-id>/
├── plan/<method-dir>/             阶段配置、job.json、日志、result.json
├── train/<method-dir>/<stage>/    config.json、metrics.jsonl、checkpoints/
├── eval/<method-dir>/<最终阶段>/   summary.json、trajectories.jsonl
└── compare/                      points.json、points.csv、compare.log、result.json
```

| 方法 | `<method-dir>` |
|---|---|
| 三个 baseline 和共享预训练 | `<method>-k<K>` |
| 动态后训练 | `<method>-k<K>+<ΔK>-<writer>` |

方法名将 `_` 转为 `-`；共享预训练用 `dynamic-pretrain`，writer 为 `local`、`tag`、`mask` 或 `dual-lora`。试跑在末尾添加 `_<mode>`。SwanLab 名称为 `<method-dir>_<run-id>`；跨系列复用使用当前系列编号，预训练来源通过 `pretraining` 配置追溯。

- **进度**：终端每 step 一行摘要，完整指标见 `train/` 下的 `metrics.jsonl`。SwanLab 同一方法的训练、验证、评估共用一个 run；阶段与进度看 `train/stage`、`train/epoch`，最终结果看柱状图。
- **评估与比较**：逐条结果见 `eval/`；同一次调用至少产生两个评估结果时导出 `compare/`，六次独立调用不自动汇总。指标含义见 [v3 说明](../README.md#五评估与产物)。
- **错误**：先看对应方法 `plan/` 下的 `result.json`，再看 `train.log`／`eval.log`；比较错误查看 `compare/result.json` 和 `compare.log`。
- **Checkpoint 与恢复**：每阶段默认保留最近两个 `checkpoints/global_step_<阶段step>/` 目录；初始化、恢复和评估均传目录路径。中断恢复调用 `latent_working_memory.v3.train`，传入保存的 `--config` 和 `--resume` 目录。

新试跑与正式训练使用不同 `run-id`；复用预训练时可继承来源编号，或显式指定新系列编号。已有运行目录不覆盖。
