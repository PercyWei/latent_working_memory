# 运行 v4 实验

使用 [run.sh](run.sh) 完成训练、恢复、新 run 续训和独立评估。默认 Qwen3-4B-Instruct-2507、每层 32 个 memory slots。方法与训练细节见 [v4 说明](../README.md)，默认配置为 [experiment.json](../../../../configs/v4/attention_matching/experiment.json)。

## 1. 准备环境、模型与数据

项目通过 uv 管理根 `.venv/`，与 v3 共用 Python 3.11、PyTorch 2.6.0 环境。先执行 `uv sync --frozen`，再参考 [.env.example](../../../../.env.example) 配置项目 `.env`。

| 内容 | 默认位置 | 覆盖参数 |
|---|---|---|
| Qwen3-4B-Instruct-2507 | `~/models/Qwen3-4B-Instruct-2507` | `--model-path` |
| QASPER 数据 | `data/delta-mem-qasper/` | `--data` |
| RULER Benchmark 数据 | `data/benchmarks/ruler-v1/<model-name>/` | `--benchmark-data` |
| LoCoMo Benchmark 数据 | `data/benchmarks/locomo/` | `--benchmark-data` |

### 1.1 训练与评估数据准备

QASPER 目录保留 `train/dev/test.jsonl`，可从 [ModelScope](https://modelscope.cn/datasets/percyWeeei/latent-working-memory/files) 下载 `delta-mem-qasper/`。数据格式与筛选规则见 [v4 数据说明](../README.md#二数据)，从本地 raw 构造数据见 [QASPER 构造说明](../../data_preparation/delta_mem_qasper/README.md)。

默认记录到 SwanLab 项目 `latent-working-memory-v4`，使用项目 `.env` 中的 `SWANLAB_API_KEY`。加 `--tracking disabled` 仅保存本地产物，无需 key。

### 1.2 Benchmark 数据准备

下载与生成使用独立入口，确保系统安装有 `git`、`git-lfs`、`curl`：

```bash
bash src/latent_working_memory/v4/scripts/download_benchmarks.sh --benchmark all
```

`--benchmark` 可选 `all`（默认）、`ruler`、`locomo`。LoCoMo 下载后可直接用于评估，RULER 下载的是源码、素材与分词资源，需要使用和模型训练相同的 tokenizer 生成样本。

NLTK 的 `punkt` 和 `punkt_tab` 固定下载到项目 `data/raw/nltk_data/`。通过 ModelScope 分发时，将它与 `data/raw/ruler-v1-source/`、`data/benchmarks/locomo/` 一起保存并按原相对路径还原，保留 `nltk_data/tokenizers/` 下的资源目录。还原后跳过下载脚本，直接运行下面的生成命令；生成无需 RULER 目录中的 `.git`。生成脚本自动设置项目目录的绝对 `NLTK_DATA`，其子进程共用此设置。

```bash
bash src/latent_working_memory/v4/scripts/generate_ruler.sh \
  --model-path "~/models/Qwen3-4B-Instruct-2507" \
  --lengths 8192 16384 32768 \
  --num-samples 100 --seed 42
```

- 默认三个长度（8k，16k，32k），每个长度、每项任务生成 100 条，共 13 项任务。
- 默认写入 `data/benchmarks/ruler-v1/<model-name>/`，可传入 `--output-dir` 更改输出目录。
- 已有长度目录拒绝覆盖；失败生成后，重试前将失败的长度目录移出输出目录，或指定新的 `--output-dir`。
- 不同模型自动分目录，原始素材共享，详情见 [benchmark 数据准备说明](../benchmarks/README.md#2-手动准备数据)。

## 2. 运行

`run.sh` 统一切到项目根目录，相对配置、数据和输出路径均以此为基准。

### 2.1 完整运行

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --config configs/v4/attention_matching/experiment.json \
  --gpus 0,1 \
  --run-id am-query_20261013
```

- 显式 CLI 训练参数覆盖配置。模型和局部记忆更新参数直接修改 JSON，见[配置文件说明](../README.md#三配置与执行)。

### 2.2 单独评估

**Test 评估**

评估一个方法 run 当前保留的全部 checkpoint：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only \
  --eval-type test \
  --gpus 0,1 \
  --eval-run artifacts/v4/am-query_20261013/train/conditioned-query-m32
```

也可显式选择一个或多个 checkpoint：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only \
  --eval-type test \
  --gpus 0,1 \
  --checkpoint \
    artifacts/v4/am-query_20261013/train/conditioned-query-m32/checkpoints/global_step_50 \
    artifacts/v4/am-query_20261013/train/conditioned-query-m32/checkpoints/global_step_100
```

- Test 评估时 `--eval-type` 须设置为 `test`。
- `--evaluate-only` 须搭配 `--eval-run` 或 `--checkpoint`，两者互斥。
- `--eval-run` 仅用于单独评估，指向 `artifacts/v4/<run-id>/train/<method-dir>/` 方法级目录，会扫描当前保留且包含 `state.pt` 的 checkpoint。
- `--checkpoint` 仅接受属于同一 run 下的单/多个 checkpoint 列表。
- 多 checkpoint 评估时支持设置多 GPU 并行计算，全部计算结束后，由父进程汇总结果，串行上传完整报告，checkpoint 按真实累计 step 排列。
- 默认沿用评估配置 `plan/<method-dir>/test.json`。缺少该文件时，必须传入 `--test-data` 以创建评估配置，`--max-test-samples`、`--max-new-tokens` 可使用默认值。模型、方法和写入配置从 checkpoint 读取。
- 每个 checkpoint 的结果独立保存。相同 checkpoint、相同配置复用已有结果和日志；在线发布将本次选中结果与此前已发布结果合并，按真实累计 step 排序重建完整报告；其他离线结果须明确选择后才发布。内容相同时仅检查、修复云端布局；需要更新时从缓存重建，不重新计算 QA，也无需保留此前 checkpoint 的权重。计算日志见各 checkpoint 的 `eval.log`，发布状态及错误见任务级 `status.json`。
- 默认追加到原训练 SwanLab run，须保留原训练目录与身份文件且云端 run 状态为 `FINISHED`；加 `--tracking disabled` 仅保存本地结果。
- 在线发布只更新当前任务报告，同时恢复同一 run 已成功发布的 test、RULER 和 LoCoMo 面板；其他任务复用已上传媒体，不重新计算或增加报告版本。

**Benchmark 评估**

```bash
# LoCoMo
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only \
  --eval-type benchmark \
  --benchmark locomo \
  --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-query-m32/checkpoints/global_step_N \
  --benchmark-data /data/benchmarks/locomo

# RULER
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only \
  --eval-type benchmark \
  --benchmark ruler \
  --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-query-m32/checkpoints/global_step_N \
  --benchmark-data data/benchmarks/ruler-v1/<model-name>/ \
  --lengths 8192 16384 32768
```

- Benchmark 评估时 `--eval-type` 须设置为 `benchmark`，同时指定 `--benchmark`。
- `--eval-run` 和 `--checkpoint` 的基本用法同 test。
- 默认沿用评估配置 `plan/<method-dir>/<benchmark>.json`。缺少该文件时，必须传入 `--benchmark-data` 以创建评估配置，`--lengths`（RULER 专属）可使用默认值。模型、方法和写入配置从 checkpoint 读取。
- checkpoint 的保存，结果的发布等同 test 评估。
- `--baseline-results` 按“标签＋目录”成对传入，标签不可重复，每次仅展示本次传入的基线。发布时读取当前 benchmark 的汇总，尚未计算的基线告警跳过；路径、格式或对照协议错误仍报错，不影响主方法计算结果的保存。

### 2.3 中断恢复

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --resume \
  --gpus 0,1 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-query-m32/checkpoints/global_step_N
```

- 训练配置等信息均沿用原运行，拒绝 CLI 参数覆盖。
- 可重新选择物理 GPU，卡数须与原任务一致。

### 2.4 新 run 续训

从运行 A 的任意已保存 checkpoint 创建新计划 B，无需 A 已完成，A 保持不变。例如额外训练 1 个 epoch：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --fork-train \
  --checkpoint artifacts/v4/old-plan/train/conditioned-query-m32/checkpoints/global_step_600 \
  --train-only \
  --gpus 0,1 \
  --run-id new-plan \
  --epochs 1
```

- 不传 `--config` 时沿用来源 checkpoint 的 seed、模型和训练配置，传入时使用新配置，CLI 参数会覆盖配置参数；然后校验最终模型配置（`model`）与来源的一致，否则报错，`training` 和 `seed` 允许调整。
- 仅继承可训练权重，优化器、RNG、数据加载器与样本计数重新开始；`--epochs` 表示 B 的新增预算。
- B 的本地 step 从 0 开始，验证和保存间隔按本地 step 计算；来源 checkpoint 为 step s 时，图表横轴和 checkpoint 名使用累计 step s+t。
- B 使用新 run-id、目录和 SwanLab 身份，将来源截至 s 的 train/dev 指标按原值重放后继续记录。`train/stage` 原始训练为 0，每次分叉取来源 checkpoint 所属阶段加 1，继承前缀保留原阶段。
- 启动时须保留来源 `state.pt` 和 `metrics.jsonl` 截至该 checkpoint 的完整前缀。
- B 使用自己的 checkpoint 和 `--resume` 恢复，不再依赖来源文件；test／benchmark 仅评估 B 自己的 checkpoint，不复制 A 的评估结果。

## 3. 调整参数

以下默认值以仓库预设为准。

### 3.1 训练输入与执行预算

| 参数 | 用途／默认值 |
|---|---|
| **运行方式** |
| `--train-only` |  只进行训练及 dev 验证，跳过 test 评估 |
| `--resume` | 从 checkpoint 恢复并沿用原配置继续训练及评估，须指定 `--checkpoint` |
| `--fork-train` | 从 checkpoint 新开 run 续训，须指定 `--checkpoint` |
| `--evaluate-only` | 仅评估已有 checkpoint，须指定 `--checkpoint` 或 `--eval-run` |
| **配置与模型** |
| `--config` | JSON 配置文件，默认 `configs/v4/attention_matching/experiment.json` |
| `--model-path` | 基座模型位置，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--seed` | 随机种子，控制初始化、样本选择及数据打乱，默认 42 |
| **目录与设备** |
| `--run-id` | 实验运行编号，默认 `YYYYMMDD-HHMMSS` |
| `--output-root` | 运行产物的父目录，默认 `artifacts/v4` |
| `--gpus` | 物理 GPU 编号，默认 `0,1` |
| **数据集与训练筛选** |
| `--data` | 完整数据集目录，默认 `data/delta-mem-qasper` |
| `--max-train-samples` | 训练样本数上限，默认 `null`（全部） |
| `--max-train-data-tokens` | 训练正文上限，默认 12288 |
| **训练时长** |
| `--epochs` | 训练轮数，默认 4 |
| **批处理** |
| `--micro-batch-size-per-gpu` | 每张 GPU 一次并行处理的样本数，默认 1 |
| `--gradient-accumulation-steps` | 每次优化器更新前累积的 microbatch 数，默认 8 |

- `--resume`、`--evaluate-only`、`--fork-train` 互斥；`--train-only` 可搭配 `--fork-train`，但不能搭配其他两个。
- 长度筛选仅用于 train 正文，不含系统提示、chat template 等。`--max-train-data-tokens` 接受正整数或 `null`：超限样本整条排除，`null` 跳过该长度筛选。除此之外，默认要求样本正文至少包含 `pending_size + recent_size + 1` 个 tokens，即能进行至少一次压缩。实际每次编码／解码仍须满足模型窗口限制。
- 全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数，默认按全局 batch 16 进行设置。

### 3.2 验证、测试与结果记录

| 参数 | 用途／默认值 |
|---|---|
| **Checkpoint** |
| `--save-every` | checkpoint 保存的正整数 optimizer step 间隔或 `after_each_epoch`（默认） |
| `--save-total-limit` | 保留最近成功保存的 checkpoint 数，默认 4 |
| `--checkpoint` | 用于恢复、续训或独立评估；恢复和续训接受一个目录，独立评估支持同一 run 的多个目录 |
| `--eval-run` | 仅用于独立评估，指定 `train/<method-dir>/`，评估其全部保留的 checkpoint |
| **Dev 验证** |
| `--max-dev-samples` | dev 验证样本数上限，默认 `null`（全部） |
| `--dev-every` | dev 验证的 optimizer step 间隔，默认 50 |
| **Test 和 benchmark 评估** |
| `--eval-type` | 评估类型，支持 `test`（默认）和 `benchmark` |
| `--test-data` | 用于完整运行或独立评估时覆盖 test 数据目录，默认 `null` |
| `--max-test-samples` | test 评估样本数上限，默认 `null`（全部） |
| `--max-new-tokens` | test 评估时 token 生成上限，默认 128 |
| `--benchmark` | benchmark 名称，默认 `null` |
| `--benchmark-data` | benchmark 数据目录，默认位置见[第 1 节](#1-准备环境模型与数据) |
| `--baseline-results` | 用于 baseline 对比展示的 “标签＋结果目录” 列表 |
| `--lengths` | 仅用于 RULER benchmark 评估，默认 `8192 16384 32768` |
| **SwanLab** |
| `--tracking` | SwanLab 记录方式，支持 `online`（默认）和 `disabled`（仅保存本地产物） |
| `--swanlab-project` | SwanLab 项目，默认 `latent-working-memory-v4` |
| `--group` | SwanLab 实验分组，默认当前 `run-id` |
| `--tag` | SwanLab 命名空间标签 |

- 除了 `--save-every` 和 `--dev-every` 所指定的 step，epoch 末会自动保存 checkpoint 及进行 dev 验证。

## 4. 查看结果

产物位于 `artifacts/v4/<run-id>/`：

```text
artifacts/v4/<run-id>/
├── plan/<method-dir>/
│   ├── config.json             公共模型配置与 seed
│   ├── train.json              训练设置
│   ├── test.json               test 评估配置
│   └── <benchmark>.json        不同 benchmark 评估配置
├── train/<method-dir>/
│   ├── run.json                训练实际配置、数据与运行信息
│   ├── status.json             训练流程状态记录
│   ├── swanlab.json            SwanLab 记录
│   ├── train.log               训练日志
│   ├── summary.json            训练完成情况
│   ├── metrics.jsonl           训练与验证指标
│   └── checkpoints/global_step_N/
└── eval/<method-dir>/
    ├── test/
    │   ├── status.json         评估流程状态记录
    │   ├── swanlab.json        SwanLab 记录
    │   └── global_step_N/
    │       ├── eval.json       评估配置快照
    │       ├── eval.log        评估日志
    │       ├── summary.json    评估指标与数据统计
    │       └── details.jsonl   逐样本详细评估记录
    └── <benchmark>/            不同 benchmark 单独保存，结构同 test/
```

- **目录命名**：`<method-dir>` 为 `<query_mode>-query-m<num_slots>`，例如 `conditioned-query-m32`；SwanLab 名称为 `<method-dir>_<run-id>`。
- **流程与结果**：`plan/` 仅保存流程设置和配置；`train/`、`eval/` 分别保存训练、评估的日志与执行结果。
- **实验配置**：`--config` 读取 `configs/v4/` 中的预设。`plan/<method-dir>/` 下，`config.json` 保存模型设置、seed 等通用配置，`train.json` 保存训练流程配置，`test.json` 保存评估流程配置。
- **训练与验证**：终端每 step 输出摘要，`metrics.jsonl` 保存完整指标；`run.json` 的 `data` 字段记录实际样本身份，其中 `data.selection` 保存筛选、抽样统计，新 run 续训的来源见 `run.json.continuation`；`metrics.jsonl` 包含来源前缀与本次训练指标。
- **评估结果**：每个 checkpoint 独立保存 `eval.json` 配置快照、`details.jsonl` 逐样本详细评估记录（每行一道 QA）和 `summary.json` 指标汇总；test 的 `data_statistics` 字段保存完整的原始样本信息、筛选数量和 token／长度统计。相同 checkpoint、相同配置复用已完成结果。
- **累计发布**：`eval/<method-dir>/<test|ruler|locomo>/swanlab.json` 保存最新成功报告的 run 身份、成员 `summary.json` 路径、报告签名、发布版本和面板信息；上传并核验成功后原子更新，失败时保留此前成功记录。结果与配置由各自文件保存。
- **Checkpoint 与恢复**：训练目录的 `summary.json` 记录完成情况、进度统计与最终 checkpoint 路径，恢复、续训和评估均向 `--checkpoint` 传入目录路径，tokenizer 从配置中的本地基座模型目录加载。
- **SwanLab 展示**：训练、dev 和评估共用原 run；train/dev 的写入 loss 图合并写入前后曲线。合并已发布与本次选中结果，按真实 optimizer step 重建完整报告：单 checkpoint 使用汇总图，多 checkpoint 曲线横轴为真实 step，滑块表示独立递增的报告发布版本。详细布局见 [v4 说明](../README.md#四评估与记录)。
- **运行状态**：训练 `status.json` 保存命令、GPU、起止时间、返回码和失败原因等执行状态。评估任务级 `status.json` 保存任务信息、本轮选择 `selected_checkpoints`、整体状态与进度 `progress`，以及 `checkpoints` 中以 `global_step_N` 为键的最近执行状态，包括命令、GPU、起止时间、返回码及计算／发布失败详情。补评估保留未选中的旧条目，整体状态与进度仅统计本轮选择。
- **失败重跑**：计算或发布失败时保留本地结果，重新执行原命令可复用已完成结果并补传。在线追加要求云端原训练 run 为 `FINISHED`。
