# 运行 v3 实验

使用 [run_gpu.sh](run_gpu.sh) 完成训练、评估与方法比较。默认使用 Qwen3-4B；基线每块 512 slots，动态方法首次 512 slots、每次追加 32 slots。方法与训练细节见 [v3 说明](../README.md)，预设位于 [configs/v3](../../../../configs/v3/)。

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

默认 AE／LM 多段文本数据**尚未构造，须先完成构造再训练**。

构造无放回分批读取到各划分配额，每篇代表原文可构造多条正文及续文均不重叠的轨迹，同篇全部轨迹归属同一划分。每条抽取 3–5 段，各段名义 token 长度 l 在 `[K,3K]` 内采样，分别扩为 `ceil(4 × l × α)` 个字符；尾部 `continuation` 独立取 `ceil(4 × Q × α)` 个字符。此处 Q 是构造时估算续文候选长度的参数，α 为 `content_reserve_ratio`，默认 1.5。字符分段在构造时固定；保存的 `estimated_tokens = len(text) / 4` 包含余量，实际 token 数取决于训练 tokenizer。

| 构造配置（α=1.5） | 每段名义 tokens | 每段保存字符数 | 正文保存字符数 | 尾部字符数（Q=512） |
|---|---:|---:|---:|---:|
| K512（默认） | 512–1536 | 3072–9216 | 9216–46080 | 3072 |
| [K64](../../../../configs/data_preparation/fineweb-multisegment/fineweb-multisegment-k64-seg1to3x_train32k.json) | 64–192 | 384–1152 | 1152–5760 | 3072 |

预训练目录根层保留 `preparation.json` 与 `train/dev/test.jsonl`，无需原始 FineWeb Parquet。加载使用 `multisegment_random_prefix`：按保存的字符段独立分词，计算不超过 8192 tokens 的最大连续前缀段数，再均匀随机选择段数；只有首段超限时才裁剪首段。

每条来源按 `--lm-ratio` 选择一个 AE 或 LM 目标，默认 LM 概率 0.5。LM 从选中前缀的终点取 Q-token 续文，可包含尚未选中的正文；训练 Q 由 `training.lm_target_tokens` 决定，默认 512，可通过 `--lm-target-tokens` 修改，不读取构造元信息中的候选长度。续文不足时转为 AE。采样使用训练 seed 与 `trajectory_id`，加载一次后所有 epoch 复用，结果不受 batch 或 GPU 数量影响。AutoCompressors 保持 LM-only，使用最多 Q-token 的可用续文；短流保留后续分段的监督，使写入器仍能训练。模型的 `memory_slots` 与构造配置的 K 分别设置，当前默认均为 512。

QA 默认使用统一格式的 FactQA，可从 [ModelScope 数据仓库](https://modelscope.cn/datasets/percyWeeei/latent-working-memory/files) 下载到表中目录。格式见 [FactQA 构造说明](../../data_preparation/fineweb_factqa/README.md)，训练读取 `preparation.json` 与三个 split 文件。

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
  --mode full --gpus 0,1 --run-id capacity-comparison_20261007-01
```

GPU 任务平台使用同一命令，将脚本位置替换为绝对路径即可。示例中的 `capacity-comparison_20261007-01` 是当前五方法记忆容量分配对照实验的自定义标识；省略 `--run-id` 时，默认按上海当前时间生成 `YYYYMMDD-HHMMSS`。

| 档位 | 每阶段训练／开发集样本上限 | 每阶段 optimizer steps 上限 | 最终 QA 评估 |
|---|---|---|---|
| `smoke`（默认） | 16／4 | 2 | dev，最多 2 条轨迹 |
| `pilot` | 256／32 | 20 | dev，最多 16 条轨迹 |
| `full` | 全部 | 按预设训练，默认 1 epoch | 完整 test |

各档均执行完整方法流程，QA 保留整条轨迹。先用 `smoke` 检查显存、日志和 SwanLab 展示，再切换 `full`；正式运行重新训练，不自动沿用试跑权重。提高 microbatch 后可先运行 `pilot`，检查更多长轨迹的显存峰值；少量 smoke 样本的显存不能代表完整数据。

同一方法只加载一次模型：ICAE 的 AE＋LM → QA、动态方法的动作预热 → 策略训练在同一组训练进程中连续完成，阶段切换时保留模型权重、重置优化器。共享预训练和不同方法分别启动；最终评估独立运行。

用 `--method` 选择运行对象：

| 值 | 执行内容 |
|---|---|
| `all`（默认） | 三个 baseline 和两个动态方法；动态预训练只执行一次 |
| `icae_single`、`icae_multi` | 所选 ICAE 的 AE＋LM → QA → 评估 |
| `autocompressors` | 分段 LM → 评估 |
| `memory_change`、`information_loss` | 共享预训练 → 所选方法的动作预热 → 策略训练 → 评估 |
| `dynamic` | 共享预训练一次，再分别完成两个动态方法 |
| `shared_pretrain` | 仅生成动态方法共用的预训练 checkpoint |

例如，只试跑 ICAE-single：

```bash
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode smoke --method icae_single --gpus 0,1
```

## 3. 复用共享预训练（可选）

需要分开调度动态方法时，先运行一次共享预训练：

```bash
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode full --method shared_pretrain --gpus 0,1 --run-id capacity-comparison_20261007-01
```

从 `artifacts/v3/capacity-comparison_20261007-01/plan/shared-pretrain/result.json` 取得 checkpoint 路径，再执行：

```bash
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode full --method memory_change --gpus 0,1 \
  --init-checkpoint /path/to/pretrain/checkpoints/step-NNNNNN.pt
```

- 自动连续执行动作预热、策略训练和最终评估，此时只需要 QA 数据。
- 将方法改为 `information_loss` 可单独训练另一组；改为 `dynamic` 可顺序完成两组。
- 保留原预训练运行目录及路径，包括 `pretrain/run.json` 和在线运行生成的 `swanlab.json`。运行标识自动继承，模型配置须与预训练一致。

## 4. 调整参数

通用命令行参数覆盖各阶段预设，预训练与动态专用参数仅应用于相应阶段。常用参数如下，完整列表通过 `bash src/latent_working_memory/v3/scripts/run_gpu.sh --help` 查看。

| 参数 | 用途／默认值 |
|---|---|
| `--gpus 0,1` | 使用的物理 GPU；默认 `0,1`，最终评估使用第一张卡 |
| `--micro-batch-size-per-gpu` | 每卡一次并行处理的样本／轨迹数；共享预训练默认 8，其余默认 4 |
| `--gradient-accumulation-steps` | 每次更新累积的 microbatch 数；共享预训练默认 1，其余默认 2 |
| `--qa-batch-size 8` | 每条轨迹一次读取的题数，默认 8 |
| `--append-slots 32` | 动态方法每次追加的 slots 数，默认 32；首次为 512，覆盖保持末块大小 |
| `--max-input-tokens 8192` | 预训练输入长度上限，默认 8192 tokens |
| `--lm-ratio 0.5` | ICAE 与动态共享预训练选择 LM 的概率；AutoCompressors 保持 LM-only |
| `--lm-target-tokens 512` | 预训练 LM 的续文目标长度，由训练配置决定 |
| `--bptt-steps 0` | 动态 warmup／policy 的 BPTT 窗口；0 表示默认的完整 BPTT，2 表示每两轮截断 |
| `--epochs` | 各阶段训练轮数 |
| `--train-samples`、`--dev-samples`、`--max-steps`、`--eval-trajectories` | 覆盖档位预算，`0` 表示不设该上限 |
| `--threshold-i`、`--threshold-d`、`--threshold-g`、`--eta` | 动态策略阈值 |
| `--run-id`、`--group` | 运行标识默认按上海时间生成；group 默认直接使用 `run-id`，可单独覆盖 |
| `--output-root` | 产物父目录，默认 `artifacts/v3` |

全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数。共享预训练双卡默认 **2 × 8 × 1 = 16**，其余阶段默认 **2 × 4 × 2 = 16**。`dynamic_pretrain.json` 只配置共享预训练，动态 QA 分别读取 `memory_change.json`、`information_loss.json`；三个 baseline 读取 `icae_single.json`、`icae_multi.json`、`autocompressors.json`。

当前先按完整 BPTT 运行；若要测量两轮截断的显存和质量，可执行：

```bash
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode pilot --method dynamic --gpus 0,1 --bptt-steps 2
```

首段也计一轮，每两轮反向传播后 detach 全部记忆，global batch 结束后仍只更新一次参数。该参数不影响 AutoCompressors 的 `ac_bptt_steps`。

动态容量按 `512 → 544 → 576` 逐次追加；覆盖只改写末块，不增加容量。共享预训练生成 512 slots，三个 baseline 的块大小不受 `--append-slots` 影响。

## 5. 查看结果

`full`、`smoke`、`pilot` 的产物统一位于 `artifacts/v3/<run-id>/`：

```text
plan/<method>/                运行计划、各方法的训练日志、result.json
train/<run-name>/<stage>/     配置、训练指标、checkpoints/
eval/<run-name>/<stage>/      评估汇总与逐题结果
compare/<method>/             多方法质量—容量比较
```

SwanLab 中，一个完整方法对应一个 run，训练、验证和最终评估共用；共享预训练单独记录，因此 `all` 共六个 runs。正式名称为 `<method>-k512_<run-id>`，试跑为 `<method>-k512_<mode>_<run-id>`，其中 `<mode>` 为 `smoke` 或 `pilot`。动态方法的 `k512` 表示首次容量，追加大小记录在 config 中。两个动态方法与共享预训练使用相同后缀关联来源；group 默认直接使用 `run-id`，`--group` 可覆盖。

阶段用 `train/stage` 曲线展示，最终质量和容量用合并柱状图展示；详细统计与样例保存在本地，不上传表格。指标含义见 [v3 说明](../README.md#评估与产物)。

终端每个 optimizer step 打印一行摘要：方法／阶段、epoch、阶段内步数、损失、容量、耗时和峰值显存；执行验证时附带 dev loss。完整指标保存在阶段目录的 `metrics.jsonl`。

相同 `run-id` 的不同档位共用 `plan/<method>/`；试跑与正式训练作为独立运行时，使用不同 `run-id` 或省略该参数自动生成。重复启动已有方法目录会报错；失败时查看 `plan/<method>/result.json` 和 `<run-name>-train.log`，同一方法的各阶段共用一个训练日志。
