# 运行 v3 实验

使用 [run.sh](run.sh) 完成训练、评估与方法比较。默认 Qwen3-4B、K=512、ΔK=32；K 为 baseline 总容量或动态方法首次容量，ΔK 为动态追加容量。方法与训练细节见 [v3 说明](../README.md)，默认配置位于 [configs/v3](../../../../configs/v3/)。

## 1. 准备环境、模型与数据

项目通过 uv 管理环境，使用 Python 3.11、PyTorch 2.6.0。先执行 `uv sync`，再参考 [.env.example](../../../../.env.example) 完善各服务器的 `.env`：

```dotenv
SWANLAB_API_KEY=你的_API_Key
MODELSCOPE_API_TOKEN=你的_ModelScope_Token
```

ModelScope token 按需填写。

| 内容 | 默认位置 | 覆盖参数 |
|---|---|---|
| Qwen3-4B-Instruct-2507 | `~/models/Qwen3-4B-Instruct-2507` | `--model-path` |
| AE／LM 数据 | `data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/` | `--pretrain-data` |
| QA 数据 | `data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008/` | `--qa-data` |

数据目录须保留 `preparation.json` 与 `train/dev/test.jsonl`。可从 [ModelScope](https://modelscope.cn/datasets/percyWeeei/latent-working-memory/files) 下载；格式见 [Fineweb Multisegment](../../data_preparation/fineweb_multisegment/README.md) 和 [Fineweb FactQA](../../data_preparation/fineweb_factqa/README.md)。

默认记录到 SwanLab 项目 `latent-working-memory-v3`，使用项目 `.env` 中的 key，不读取账号保存的登录凭据。加 `--tracking disabled` 仅保存本地产物，无需 SwanLab key。

## 2. 试跑与正式运行

GPU 编号按实际分配修改；最终评估使用第一张卡。一次完成五种方法：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --mode full --method all --gpus 0,1 \
  --run-id capacity-comparison_20261011-01
```

- `--method all` 默认运行三组 baseline、一次共享预训练和两组动态后训练；分别启动见下节。
- 加 `--dry-run` 只预览计划，不加载模型、数据或连接 SwanLab。
- `--run-id` 可自定义；复用共享预训练且省略时继承来源编号，其余情况按上海当前时间生成 `YYYYMMDD-HHMMSS`。

| 档位 | 每阶段训练／开发集样本上限 | 每阶段 steps 上限 | 最终评估 |
|---|---|---|---|
| `smoke` | 16／4 | 2 | dev，最多 2 条轨迹 |
| `full`（默认） | 按阶段配置／全部开发集 | 默认 1 epoch | 完整 test |

先用 `--mode smoke` 检查显存、日志和 SwanLab，再用 `full` 正式训练。各档均按配置完成该方法的阶段流程；同一方法的连续阶段共用模型，切换时重置优化器。

### 2.1 六次运行的指令

分别调度时，用同一个 `run-id` 启动三组 baseline、一次共享预训练和两组动态后训练。microbatch 与梯度累积可按显存调整。

#### 1）Baseline

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --mode full --method icae_single --gpus 0,1 \
  --run-id capacity-comparison_20261011-01 \
  --micro-batch-size-per-gpu 4 --gradient-accumulation-steps 2
```

将 `--method` 分别设为 `icae_single`、`icae_multi`、`autocompressors`，执行三次；每次完成该方法的训练流程和 test 评估。

#### 2）动态方法

先运行一次共享预训练：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --mode full --method dynamic_pretrain --gpus 0,1 \
  --run-id capacity-comparison_20261011-01 \
  --micro-batch-size-per-gpu 4 --gradient-accumulation-steps 2
```

完成后，从 `artifacts/v3/capacity-comparison_20261011-01/train/dynamic-pretrain-k512/pretrain/training-result.json` 的 `checkpoint` 字段取得实际目录，替换下例的 `global_step_N`：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --mode full --method memory_change --gpus 0,1 \
  --run-id capacity-comparison_20261011-01 \
  --micro-batch-size-per-gpu 4 --gradient-accumulation-steps 2 \
  --writer-mode local \
  --init-checkpoint artifacts/v3/capacity-comparison_20261011-01/train/dynamic-pretrain-k512/pretrain/checkpoints/global_step_N
```

将后训练的 `--method` 分别设为 `memory_change`、`information_loss`，执行两次。两组复用同一 checkpoint，独立训练，各自连续完成 warmup → policy → test 评估。

- 同系列后训练可省略 `--run-id`，自动继承来源编号；跨系列复用见下节。
- `--writer-mode` 仅用于后训练，可选 `local`、`tag`、`mask`、`dual_lora`；使用 `tag` 时可加 `--tag-tokens 3`。共享预训练不传此参数。
- 上述双卡设置的 global batch 均为 16；更多可调参数见[第四节](#4-调整参数)。

## 3. 复用共享预训练（可选）

四种动态写入方式均可复用同一基础预训练；命令见[第 2.1 节](#21-六次运行的指令)。复用后仅需 QA 数据。

- **标识**：同系列后训练省略 `--run-id` 时继承来源编号；跨系列复用时显式指定新的 `--run-id`。`--group` 独立指定，默认采用当前 `run-id`。
- **模型**：模型配置须与预训练一致。更换基座或首次容量 K 时，先准备匹配的 checkpoint。
- **来源**：SwanLab 的 `pretraining` 配置与本地 `run.json` 记录来源编号、名称、URL、checkpoint 和 step；保留原预训练路径、`pretrain/run.json` 及在线运行的 `swanlab.json`。
- **使用范围**：同一预训练可用于不同追加容量和写入方式的独立后训练；共享预训练本身不执行追加／覆盖。

例如，在新的写入方式对比系列中复用容量对比系列的 checkpoint，仍按第 2.1 节取得实际 `global_step_N`：

```bash
bash src/latent_working_memory/v3/scripts/run.sh \
  --method information_loss --gpus 0,1 --writer-mode tag \
  --run-id writer-mode-comparison_20261012 \
  --init-checkpoint artifacts/v3/capacity-comparison_20261011-01/train/dynamic-pretrain-k512/pretrain/checkpoints/global_step_N
```

## 4. 调整参数

### 4.1 配置与覆盖

- 未指定 `--config` 时读取 `configs/v3/`；`--config <文件>` 仅用于单方法，不能与 `--method all` 组合。
- 文件中的 `objective.method` 须与方法一致，`objective.stages` 声明阶段顺序。`dynamic_pretrain` 使用 `"method": "dynamic"` 和 `["pretrain"]`。
- 显式 CLI 参数覆盖文件值；动态自定义配置只替换后训练，自动预训练仍读取独立配置。数据路径参数见第一节。
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
