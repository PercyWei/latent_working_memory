# 20261013_v4 统一运行入口

创建时间：20261013 10:12:17 Asia/Shanghai（UTC+08:00）

最后修订时间：20261016 19:26:12 Asia/Shanghai（UTC+08:00）

使用 [run.sh](run.sh) 完成训练、恢复和独立评估。默认 Qwen3-4B-Instruct-2507、每层 32 个 memory slots；模型与训练目标见 [v4 说明](../README.md)，默认配置为 [experiment.json](../../../../configs/v4/attention_matching/experiment.json)。以下命令从项目根目录执行。

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

`--benchmark` 可选 `all`（默认）、`ruler`、`locomo`。LoCoMo 下载后可直接用于评估，RULER 下载的是素材，需要进一步生成样本（使用和模型训练相同的 tokenizer）：

```bash
bash src/latent_working_memory/v4/scripts/generate_ruler.sh \
  --model-path "~/models/Qwen3-4B-Instruct-2507" \
  --lengths 8192 16384 32768 \
  --num-samples 100 --seed 42
```

- 默认三个长度（8k，16k，32k），每个长度、每项任务生成 100 条，共 13 项任务。
- 默认写入 `data/benchmarks/ruler-v1/<model-name>/`，可传入 `--output-dir` 更改输出目录。
- 不同模型自动分目录，原始素材共享，详情见 [benchmark 数据准备说明](../benchmarks/README.md#2-手动准备数据)。

## 2. 运行

`run.sh` 统一切到项目根目录，相对配置、数据和输出路径均以此为基准。`--train-only`、`--resume`、`--evaluate-only` 互斥。恢复须提供一个 `--checkpoint` 目录；独立评估须指定 `--checkpoint` 或 `--eval-run`，两者互斥。

### 2.1 完整运行

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --config configs/v4/attention_matching/experiment.json \
  --gpus 0,1 --run-id am-query_20261013
```

- 显式 CLI 训练参数覆盖配置。模型和局部记忆更新参数直接修改 JSON，见[配置文件说明](../README.md#三配置与执行)。

### 2.2 单独评估

**Test 评估**

评估一个方法 run 当前保留的全部 checkpoint：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --eval-type test --gpus 0,1 \
  --eval-run artifacts/v4/am-query_20261013/train/conditioned-query-m32
```

也可显式选择一个或多个 checkpoint：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --eval-type test --gpus 0,1 \
  --checkpoint \
    artifacts/v4/am-query_20261013/train/conditioned-query-m32/checkpoints/global_step_50 \
    artifacts/v4/am-query_20261013/train/conditioned-query-m32/checkpoints/global_step_100
```

- Test 评估时 `--eval-type` 须设置为 `test`。
- `--evaluate-only` 须搭配 `--eval-run` 或 `--checkpoint`，两者互斥。
- `--eval-run` 仅用于单独评估，指向 `artifacts/v3/<run-id>/train/<method-dir>/` 方法级目录，会扫描其各阶段当前保留且包含 `state.pt` 的 checkpoint。
- `--checkpoint` 仅接受属于同一 run 下的单/多个 checkpoint 列表。
- 多 checkpoint 评估时支持设置多 GPU 并行计算，全部计算结束后，由父进程汇总结果，串行上传完整报告，checkpoint 按真实累计 step 排列。
- 默认沿用评估配置 `plan/<method-dir>/test/eval.json`。缺少该文件时，必须传入 `--eval-data` 以创建评估配置；`--seed`、`--max-test-samples`、`--max-new-tokens` 非必须，可使用默认值。模型、方法和写入配置从 checkpoint 读取。
- 每个 checkpoint 的结果独立保存。相同 checkpoint、相同配置复用已有结果和日志；在线发布将本次选中的结果与此前已发布的本地结果合并为完整报告，其他离线结果须明确选择后才发布。内容相同时仅检查、修复云端布局；需要更新时从缓存重建，补发日志追加到原 `eval.log`，不重新计算 QA，也无需保留此前 checkpoint 的权重。
- 默认追加到原训练 SwanLab run，须保留原训练目录与身份文件且云端 run 状态为 `FINISHED`；加 `--tracking disabled` 仅保存本地结果。

**Benchmark 评估**

```bash
# LoCoMo
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --eval-type benchmark --benchmark locomo --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-query-m32/checkpoints/global_step_N

# RULER
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --eval-type benchmark --benchmark ruler --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-query-m32/checkpoints/global_step_N \
  --lengths 8192 16384 32768
```

- Benchmark 评估时 `--eval-type` 须设置为 `benchmark`，同时指定 `--benchmark`。
- `--eval-run` 和 `checkpoint` 的使用同 test 评估。
- 默认沿用评估配置 `plan/<method-dir>/<benchmark>/eval.json`。缺少该文件时，必须传入 `--benchmark-data` 以创建评估配置；`--seed`、`--max-test-samples`、`--max-new-tokens` 等参数非必须，可使用默认值。模型、方法和写入配置从 checkpoint 读取。
- checkpoint 的保存，结果的发布等同 test 评估。

### 2.3 中断恢复

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --resume --gpus 0,1 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-query-m32/checkpoints/global_step_N
```

- 训练配置等信息均沿用原运行，忽略 CLI 参数覆盖。
- 可重新选择物理 GPU，卡数须与原任务一致。

## 3. 调整参数

以下默认值以仓库预设为准。

### 3.1 训练输入与执行预算

| 参数 | 用途／默认值 |
|---|---|
| **运行方式** |
| `--train-only` |  只进行训练及 dev 验证，跳过 test 评估 |
| `--resume` | 从 checkpoint 恢复并沿用原配置继续训练及评估，须指定 `--checkpoint` |
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
| `--epochs` | 训练轮数，默认 1 |
| **批处理** |
| `--micro-batch-size-per-gpu` | 每张 GPU 一次并行处理的样本数，默认 4 |
| `--gradient-accumulation-steps` | 每次优化器更新前累积的 microbatch 数，默认 2 |

- `--train-only`、`--resume`、`--evaluate-only` 互斥。
- 长度筛选仅用于 train 正文，不含系统提示、chat template 等。`--max-train-data-tokens` 接受正整数或 `null`：超限样本整条排除，`null` 跳过该长度筛选。除此之外，默认要求样本正文至少包含 `pending_size + recent_size + 1` 个 tokens，即能进行至少一次压缩。实际每次编码／解码仍须满足模型窗口限制。
- 全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数，默认按全局 batch 16 进行设置。

### 3.2 验证、测试与结果记录

| 参数 | 用途／默认值 |
|---|---|
| **Checkpoint** |
| `--save-every` | checkpoint 保存的正整数 optimizer step 间隔或 `after_each_epoch`，默认 50 |
| `--save-total-limit` | 保留最近成功保存的 checkpoint 数，默认 2 |
| `--checkpoint` | 用于训练恢复或独立评估，其中独立评估支持传入同一 run 的多个 checkpoint 目录 |
| `--eval-run` | 仅用于独立评估，指定 `train/<method-dir>/`，评估其全部保留的 checkpoint |
| **Dev 验证** |
| `--max-dev-samples` | dev 验证样本数上限，默认 `null`（全部） |
| `--dev-every` | dev 验证的 optimizer step 间隔，默认 50 |
| **Test 和 benchmark 评估** |
| `--eval-type` | 评估类型，支持 `test`（默认）和 `benchmark` |
| `--eval-data` | 用于完整运行或独立评估时覆盖 test 数据目录，默认 `null` |
| `--max-test-samples` | test 评估样本数上限，默认 `null`（全部） |
| `--max-new-tokens` | test 评估时 token 生成上限，默认 128 |
| `--benchmark` | benchmark 名称，默认 `null` |
| `--benchmark-data` | benchmark 数据目录，默认位置见[第 1 节](#1-准备环境模型与数据) |
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
├── plan/<method-dir>/            config.json、命令与执行状态、运行日志
├── train/<method-dir>/           run.json、tokenizer/、data-statistics.json、metrics.jsonl
│   ├── checkpoints/global_step_N/
│   ├── swanlab.json              在线运行身份
│   └── training-result.json      训练结果与最终 checkpoint
└── eval/<method-dir>/
    ├── test/                    QASPER test
    │   ├── eval.json            同类评估共用的计算协议
    │   └── global_step_N/       协议快照、逐题结果、汇总及发布记录
    └── ruler/、locomo/           同样保存 eval.json 及各 checkpoint 的结果
```

- `<method-dir>` 按 `<query_mode>-query-m<num_slots>` 生成，例如 `conditioned-query-m32`。
- **评估产物**：每个 checkpoint 的结果目录包含 `eval.json` 协议快照、`predictions.jsonl` 和计算完成标记 `summary.json`；QASPER 另存筛选统计，在线发布时保存 `publication.json`。运行日志和状态位于 `plan/<method-dir>/<evaluation-name>/global_step_N.log`、`global_step_N-result.json`，其中 `<evaluation-name>` 为 `evaluation`、`ruler-evaluation` 或 `locomo-evaluation`。批量评估另存 `plan/<method-dir>/<evaluation-name>-result.json` 汇总各 checkpoint 的结果与失败情况。
- **进度与图表**：终端展示 step、loss、耗时与显存，完整训练指标见 `metrics.jsonl`。训练、dev 和评估共用原 SwanLab run；`train/*`、`dev/*` 的写入 loss 图合并写入前后曲线，单 checkpoint 的 `evaluation/*` 展示汇总图，批量评估使用真实 checkpoint step 的原生增量曲线。布局见 [v4 评估与记录](../README.md#四评估与记录)。
- **SwanLab**：SwanLab 名称为 `<method-dir>_<run-id>`。
- **错误与状态**：训练查看 `plan/<method-dir>/train.log`、`result.json`；评估查看上述日志和状态文件。状态从 `running` 更新为 `finished`／`stopped`／`failed`，保存命令、GPU、起止时间、退出码及失败原因。
- **数据统计**：`data-statistics.json` 保存筛选、抽样统计及实际选中样本身份；恢复沿用原配置及所选子集，不接受数量上限或 seed 覆盖。
- **Checkpoint 与恢复**：默认保留最近两个 checkpoint，恢复与评估均传目录路径；具体 step 可从训练结果或 checkpoint 目录取得。中断恢复见[第 2.3 节](#23-中断恢复)。
