# 运行 v3 实验

使用 [run_gpu.sh](run_gpu.sh) 完成训练、评估与方法比较。默认使用 Qwen3-4B，容量 K=512、追加容量 ΔK=32；K 表示三个基线的总容量和动态方法的首次容量。方法细节见 [v3 说明](../README.md)，预设位于 [configs/v3](../../../../configs/v3/)。

## 1. 准备环境、模型与数据

安装 `uv` 后，在仓库根目录执行：

```bash
cd /path/to/latent_working_memory
uv sync
```

项目使用 Python 3.11，默认安装 PyTorch 2.6.0。修改 `run_gpu.sh` 开头的仓库路径：

```bash
LWM_REPO_DIR="/path/to/latent_working_memory"
```

准备本地模型和数据：

| 内容 | 默认位置 | 参数 |
|---|---|---|
| Qwen3-4B-Instruct-2507 | `~/models/Qwen3-4B-Instruct-2507` | `--model-path` |
| AE／LM 多段文本数据 | `data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/` | `--pretrain-data` |
| QA 数据 | `data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008/` | `--qa-data` |

两个数据目录均保留 `preparation.json` 与 `train/dev/test.jsonl`，无需原始 FineWeb Parquet。数据可从 [ModelScope 数据仓库](https://modelscope.cn/datasets/percyWeeei/latent-working-memory/files) 下载；构造格式见 [多段文本](../../data_preparation/fineweb_multisegment/README.md) 和 [FactQA](../../data_preparation/fineweb_factqa/README.md)。

- AE／LM 使用完整正文，超过正文 token 上限的样本整条过滤；预设 12k，上限不包含 memory、提示或目标。
- AE 重建全文，LM 使用保存的紧邻续文，最多取 `lm_target_tokens`；QA 使用完整轨迹与对应题池。
- ICAE-multi 按 seed 和样本 ID 在 3–6 中采样块数，将全文均分后独立压缩，总容量固定为 K。预训练、QA 与评估沿用同一规则；详见 [记忆组织](../README.md#方法与模型)。

默认将实验记录到 SwanLab 项目 `latent-working-memory-v3`。在仓库根目录的 `.env` 中填写：

```dotenv
SWANLAB_API_KEY=你的_API_Key
```

存在 `.env` 时只读取其中的 key；没有该文件时读取终端环境变量，不使用账号保存的登录凭据。仅保存本地结果可加 `--tracking disabled`，无需 key。

## 2. 试跑与正式运行

以下命令均在仓库根目录执行，GPU 编号按实际分配修改：

```bash
# 预览计划，不加载模型、数据或连接 SwanLab
bash src/latent_working_memory/v3/scripts/run_gpu.sh --gpus 0,1 --dry-run

# 五方法试跑
bash src/latent_working_memory/v3/scripts/run_gpu.sh --mode smoke --gpus 0,1

# 五方法正式训练
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
  --gpus 0,1 --run-id capacity-comparison_20261007-01
```

GPU 任务平台使用同一命令，将脚本位置替换为绝对路径即可。`--run-id` 可自定义；省略时按上海当前时间生成 `YYYYMMDD-HHMMSS`。

| 档位 | 每阶段训练／开发集样本上限 | 每阶段 optimizer steps 上限 | 最终 QA 评估 |
|---|---|---|---|
| `smoke` | 16／4 | 2 | dev，最多 2 条轨迹 |
| `pilot` | 256／32 | 20 | dev，最多 16 条轨迹 |
| `full`（默认） | 按各阶段预设／全部开发集 | 按预设训练，默认 1 epoch | 完整 test |

各档默认执行完整方法流程。先用 `smoke` 检查显存、日志和 SwanLab，再用 `pilot` 检查更多样本，最后切换 `full` 重新训练。试跑阶段的样本数取阶段预算与档位上限的较小值。

同一方法只加载一次模型：ICAE 的 AE＋LM → QA、动态方法的动作预热 → 策略训练在同一组训练进程中连续完成，阶段切换时保留模型权重、重置优化器。共享预训练和不同方法分别启动；最终评估独立运行。

用 `--method` 选择运行对象：

| 值 | 默认执行内容 |
|---|---|
| `all`（默认） | 三个 baseline 和两个动态方法；动态预训练只执行一次 |
| `icae_single`、`icae_multi` | 所选 ICAE 的 AE＋LM → QA → 评估 |
| `autocompressors` | 分段 LM → 评估 |
| `memory_change`、`information_loss` | 共享预训练 → 所选方法的动作预热 → 策略训练 → 评估 |
| `dynamic_pretrain` | 仅生成动态方法共用的预训练 checkpoint |

例如，只试跑 ICAE-single：

```bash
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode smoke --method icae_single --gpus 0,1
```

## 3. 复用共享预训练（可选）

需要分开调度动态方法时，先运行一次共享预训练：

```bash
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode full --method dynamic_pretrain --gpus 0,1 --run-id capacity-comparison_20261007-01
```

从 `artifacts/v3/capacity-comparison_20261007-01/plan/dynamic-pretrain-k512/result.json` 取得 checkpoint 路径，再执行：

```bash
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode full --method memory_change --gpus 0,1 \
  --init-checkpoint artifacts/v3/capacity-comparison_20261007-01/train/dynamic-pretrain-k512/pretrain/checkpoints/global_step_N
```

- 默认连续执行动作预热、策略训练和最终评估，此时只需要 QA 数据。
- 将方法改为 `information_loss` 并使用同一 checkpoint，可单独训练另一组。
- 保留原预训练运行目录及路径，包括 `pretrain/run.json` 和在线运行生成的 `swanlab.json`。运行标识自动继承，模型配置须与预训练一致。

## 4. 调整参数

通用命令行参数覆盖各阶段预设，预训练与动态专用参数仅应用于相应阶段。常用参数如下，完整列表通过 `bash src/latent_working_memory/v3/scripts/run_gpu.sh --help` 查看。

| 参数 | 用途／默认值 |
|---|---|
| `--config <文件路径>` | 替换所选单方法的 JSON 预设；未指定时读取 `configs/v3/` 中的默认文件 |
| `--gpus 0,1` | 使用的物理 GPU；默认 `0,1`，最终评估使用第一张卡 |
| `--micro-batch-size-per-gpu` | 每卡一次并行处理的样本／轨迹数；ICAE、AutoCompressors 和共享预训练默认 8，动态 warmup／policy 默认 4 |
| `--gradient-accumulation-steps` | 每次更新累积的 microbatch 数；ICAE、AutoCompressors 和共享预训练默认 1，动态 warmup／policy 默认 2 |
| `--qa-batch-size 8` | 每条轨迹一次读取的题数，默认 8 |
| `--append-slots 32` | 动态方法每次追加的 slots 数，默认 32；首次为 512，覆盖保持末块大小 |
| `--max-input-tokens 12288` | AE／LM 训练正文 token 上限；超限整条过滤，不计 memory／提示／目标；`0` 使用模型窗口 |
| `--max-qa-input-tokens 12288` | FactQA 训练正文 token 上限；超限整条过滤，不计 memory／提示／目标；`0` 使用模型窗口 |
| `--lm-ratio 0.5` | ICAE 与动态方法预训练选择 LM 的概率；AutoCompressors 保持 LM-only |
| `--lm-target-tokens 512` | 预训练 LM 的续文目标长度 |
| `--icae-min-segments 3`、`--icae-max-segments 6` | ICAE-multi 的随机块数范围，两个训练阶段均应用；总容量固定为 K |
| `--ac-num-segments 4` | AutoCompressors 的正文压缩段数；每段追加约 K/n slots，最终合计 K |
| `--bptt-steps` | BPTT 写入窗口；AutoCompressors 默认 2，动态 warmup／policy 默认完整；`0` 表示完整 BPTT。AutoCompressors 还用该值组织随机分段 |
| `--epochs` | 各阶段训练轮数 |
| `--save-total-limit 2` | 每阶段保留最近成功保存的 checkpoint 数；默认 2 |
| `--pretrain-train-samples` | 覆盖所有方法预训练阶段的训练样本上限，默认 12800 |
| `--qa-train-samples`、`--warmup-train-samples`、`--policy-train-samples` | 分别覆盖对应阶段的训练样本上限，默认不限 |
| `--dev-samples`、`--max-steps`、`--eval-trajectories` | 覆盖档位的开发集、每阶段步数和最终评估轨迹上限，`0` 表示不限 |
| `--threshold-i`、`--threshold-d`、`--threshold-g`、`--eta` | 动态策略阈值 |
| `--run-id`、`--group` | 运行标识默认按上海时间生成；group 默认直接使用 `run-id`，可单独覆盖 |
| `--output-root` | 产物父目录，默认 `artifacts/v3` |

全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数，双卡默认均为 16：

| 方法／阶段 | 每卡 microbatch | 梯度累积 |
|---|---:|---:|
| ICAE-single／ICAE-multi：预训练、QA | 8 | 1 |
| 动态共享预训练 | 8 | 1 |
| AutoCompressors | 8 | 1 |
| 动态 warmup／policy | 4 | 2 |

阶段训练样本参数的 `0` 表示不限；`smoke`／`pilot` 仍保留档位上限。`max_input_tokens` 与 `max_qa_input_tokens` 分别控制 AE／LM 和 FactQA 的训练正文上限；dev／test 不按长度上限过滤。配置值 `null` 表示训练上限采用模型窗口，模型另行检查实际调用窗口长度。

预设通过 `objective.stages` 声明阶段顺序，通过 `training.stage_max_train_samples` 分别设置各阶段训练预算：

| 预设 | 默认 `objective.stages` |
|---|---|
| `icae_single.json`、`icae_multi.json` | `["pretrain", "qa"]` |
| `memory_change.json`、`information_loss.json` | `["warmup", "policy"]` |
| `dynamic_pretrain.json` | `["pretrain"]` |
| `autocompressors.json` | `["pretrain"]` |

例如 ICAE 配置默认使用：

```json
"stage_max_train_samples": {"pretrain": 12800, "qa": null}
```

训练前按 seed 选取样本，`null` 表示全部；启动器展开后，每份阶段配置只保存该阶段的 `max_train_samples`。

复制对应默认 JSON 并修改后，指定一个方法和自定义配置文件，例如：

```bash
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode full \
  --method icae_single \
  --config configs/v3/icae_single_custom.json \
  --gpus 0,1 --micro-batch-size-per-gpu 4
```

- 文件中的 `objective.method` 须与所选方法一致，阶段使用 `objective.stages`；`--method dynamic_pretrain` 的文件使用 `"method": "dynamic"`，并仅声明 `["pretrain"]`。
- 显式超参数覆盖文件值；数据入口由 `--pretrain-data`／`--qa-data` 指定，阶段训练预算由配置及对应 CLI 参数决定，试跑档位再限制上限。产物与 SwanLab 身份由启动器生成。
- `--config` 用于单方法，不与 `all` 组合。`dynamic_pretrain` 可传入自己的预训练文件；动态后训练的自定义文件只替换 warmup／policy，共享预训练仍读取默认文件。更换动态模型或 slots 时，先生成匹配的预训练 checkpoint，再通过 `--init-checkpoint` 使用。

- 启动器按列表生成阶段配置；单元素列表如 `["pretrain"]` 只执行该阶段。正式实验保留默认完整流程。
- 仅运行后训练时，首阶段可用 `training.init_checkpoint` 指定来源；动态方法也支持 `--init-checkpoint`，未指定时先执行共享预训练。后继阶段自动继承前一阶段，单独 `warmup` 不触发最终评估。

预设不指定产物目录。启动器根据 `--run-id` 生成运行配置，写入 `plan/<method-dir>/<stage>.json`，每份配置用 `objective.stage` 明确当前阶段；训练保存的 `config.json` 也使用该单阶段字段。中断恢复使用阶段目录保存的 `config.json` 和 checkpoint，不直接运行预设。

`--bptt-steps` 统一控制写入窗口，`0` 对应配置中的 `null`，表示完整 BPTT。动态方法默认完整展开；设为 2 时每两轮反向并 detach 记忆，首次写入也计一轮。AutoCompressors 默认 2，在窗口完成后继文本监督后截断；完整 BPTT 时将全部 n 段作为一个随机切分组。各方法均在 global batch 结束后更新参数。

## 5. 查看结果

各档位产物统一保存到 `artifacts/v3/<run-id>/`，父目录可用 `--output-root` 修改。

```text
artifacts/v3/<run-id>/
├── plan/<method-dir>/             job.json、<stage>.json、train.log、eval.log、result.json
├── train/<method-dir>/<stage>/    config.json、metrics.jsonl、checkpoints/
├── eval/<method-dir>/<最终阶段>/   summary.json、trajectories.jsonl
└── compare/                      points.json、points.csv、compare.log、result.json
```

**`<method-dir>` 命名规则**

| 运行模式 | 命名 |
|---|---|
| `full` | `<method>-k<K>` |
| `smoke`、`pilot` | `<method>-k<K>_<mode>` |

- 方法名将 `_` 转为 `-`，如 `memory-change`；共享预训练使用 `dynamic-pretrain`。K 为 `model.memory_slots`，动态方法表示首次容量。
- SwanLab 名称为 `<method-dir>_<run-id>`；两个动态方法与共享预训练使用相同后缀关联来源。

| 方法 | 默认训练阶段 | 默认最终评估目录的阶段 |
|---|---|---|
| ICAE | `pretrain` → `qa` | `qa` |
| 动态方法 | `warmup` → `policy` | `policy` |
| 共享预训练 | `pretrain` | 无 |
| AutoCompressors | `pretrain` | `pretrain` |

**查看步骤**

1. **训练进度**：终端每个 optimizer step 打印一行摘要；完整指标查看阶段目录的 `metrics.jsonl`。
2. **SwanLab**：每个方法的训练、验证、评估共用一个 run；共享预训练独立记录，默认 `all` 共六个。阶段看 `train/stage`，阶段内进度看 `train/epoch`，最终质量与容量看柱状图；指标含义见 [v3 说明](../README.md#评估与产物)。
3. **方法比较**：本次调用产出至少两个评估结果时，统一导出 `points.json` 和 `points.csv`；参与的结果路径与执行状态见 `compare/result.json`。
4. **失败排查**：先看对应方法的 `plan/<method-dir>/result.json`，再看 `train.log` 或 `eval.log`；比较失败查看 `compare/result.json` 和同目录的 `compare.log`。

各阶段的 `checkpoints/` 默认保留最近两个 `global_step_<阶段step>/` 目录（`training.save_total_limit=2`）。目录内包含 `state.pt` 和各卡的 `data_<rank>.pt`；`--init-checkpoint`、`--resume` 与评估的 `--checkpoint` 均传目录路径。

已有产物不覆盖；独立试跑与正式运行使用不同 `run-id`，省略时自动生成。
