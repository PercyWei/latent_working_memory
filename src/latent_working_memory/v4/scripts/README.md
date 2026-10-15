# 20261013_v4 统一运行入口

创建时间：20261013 10:12:17 Asia/Shanghai（UTC+08:00）

最后修订时间：20261015 16:46:40 Asia/Shanghai（UTC+08:00）

使用 [run.sh](run.sh) 完成训练、恢复、独立评估和单条生成。默认 Qwen3-4B-Instruct-2507、每层 32 个 memory slots；模型与训练目标见 [v4 说明](../README.md)，默认配置为 [experiment.json](../../../../configs/v4/attention_matching/experiment.json)。以下命令从项目根目录执行。

## 1. 准备环境、模型与数据

项目通过 uv 管理根 `.venv/`，与 v3 共用 Python 3.11、PyTorch 2.6.0 环境。先执行 `uv sync --frozen`，再参考 [.env.example](../../../../.env.example) 配置项目 `.env`。

| 内容 | 默认位置 | 覆盖参数 |
|---|---|---|
| Qwen3-4B-Instruct-2507 | `~/models/Qwen3-4B-Instruct-2507` | `--model-path` |
| QASPER 数据 | `data/delta-mem-qasper/` | `--data-dir` |
| RULER Benchmark 数据 | `data/benchmarks/ruler-v1/<model-name>/` | `--data-dir` |
| LoCoMo Benchmark 数据 | `data/benchmarks/locomo/` | `--data-dir` |

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

`run.sh` 统一切到项目根目录，相对配置、数据和输出路径均以此为基准。`--resume`、`--evaluate-only`、`--generate-only` 互斥，均须提供一个 `--checkpoint` 目录。

### 2.1 新训练

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --config configs/v4/attention_matching/experiment.json \
  --gpus 0,1 --run-id am-query_20261013
```

- 显式 CLI 训练参数覆盖配置。模型和局部记忆更新参数直接修改 JSON，见[配置文件说明](../README.md#三配置与执行)。
- `--run-id` 默认采用格式为 `YYYYMMDD-HHMMSS` 的当前时间。
- 完成全部训练预算后，自动用最终 checkpoint 执行基础非 benchmark 评估，使用所选 GPU 中的第一张卡，生成上限为 128 tokens。提前停止或训练失败时不自动评估。

### 2.2 单独评估

**基础评估**

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-m32/checkpoints/global_step_N \
  --max-new-tokens 128
```

- `--benchmark` 默认 `null`，即读取原训练数据目录的 `test.jsonl`，进行与训练集配对的 test 评估。
- 计算答案 NLL 与 greedy 生成的 EM／F1。筛选、生成预算和评分规则见 [评估说明](../README.md#四评估与记录)。

**Benchmark 评估**

```bash
# LoCoMo
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --benchmark locomo --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-m32/checkpoints/global_step_N

# RULER
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --benchmark ruler --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-m32/checkpoints/global_step_N \
  --lengths 8192 16384 32768
```

- 传入 `--benchmark ruler` 或 `locomo` 选择对应评估。RULER 的 `--lengths` 须与已生成数据一致；LoCoMo 不接受该参数。
- RULER 默认按 checkpoint 中原始模型路径的末级名称查找数据。人工确认新模型可复用已有样本时，用 `--data-dir data/benchmarks/ruler-v1/<model-name>` 指定对应目录。
- 每题独立初始化记忆，执行流式读取、局部 TTT 和 greedy 生成，沿用官方评分与任务生成预算。协议见 [benchmark 说明](../benchmarks/README.md)。

两类评估均从 checkpoint 读取模型配置、实际 revision 和保存的 tokenizer，可用 `--data-dir`、`--output-dir`、`--tracking` 调整本次评估。输出目录须不存在，重跑时指定新目录。在线评估仅向原训练 run 追加最终 checkpoint 的结果，云端 run 须为 `FINISHED`；加 `--tracking disabled` 仅保存本地。

### 2.3 中断恢复

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --resume --gpus 0,1 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-m32/checkpoints/global_step_N
```

- 沿用原训练配置、预算、输出目录和 SwanLab 身份，不接受训练参数覆盖；可通过 `--stop-after-steps N` 指定本次绝对停止点。
- GPU 编号可以重新选择，卡数、数据和配置须与原 run 一致；恢复 optimizer、加载器位置和各 rank RNG。
- 保留原 `plan/<method-dir>/config.json`、训练目录的 `run.json`、`tokenizer/`、checkpoint，以及在线运行的 `swanlab.json`。续训完成预算后同样自动执行 QASPER test。

### 2.4 单条生成

准备 UTF-8 纯文本 prompt：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --generate-only --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-m32/checkpoints/global_step_N \
  --prompt-file data/attention-matching/prompt.txt \
  --max-new-tokens 128
```

直接编码 prompt，不额外应用 chat template；需要完整提示格式时由输入文件提供。prompt 与请求的新增 token 上限之和须满足模型窗口限制；该入口只保存生成结果，不计算 QA 分数。

## 3. 调整参数

以下默认值以仓库预设为准。新训练的 CLI 参数覆盖 JSON，恢复沿用原训练配置；完整参数通过 `bash src/latent_working_memory/v4/scripts/run.sh --help` 查看。

### 3.1 训练输入与执行预算

**配置与数据**

| 参数 | 用途／默认值 |
|---|---|
| `--config` | 训练 JSON，默认 `configs/v4/attention_matching/experiment.json` |
| `--model-path` | 覆盖本地基座模型位置，默认见[第 1 节](#1-准备环境模型与数据)；HF repo ID 和 revision 在 JSON 中设置 |
| `--data-dir` | 训练数据目录，默认 `data/delta-mem-qasper` |
| `--seed` | 训练随机种子，默认 42；接受非负整数，0 有效 |
| `--max-seq-length` | train 完整序列的人为上限，默认 12288；`null` 关闭该上限 |

完整序列包含正文、问题、答案和模板。超限样本整条过滤，不截断；dev/test 不使用该人为上限。QASPER 的 train/dev/test 均要求答案前 prompt 至少为 `pending_size + recent_size + 1` tokens，完整序列不得超过模型窗口。

**批处理与训练时长**

| 参数 | 用途／默认值 |
|---|---|
| `--micro-batch-size-per-gpu` | 每卡一次前向的样本数，默认 4 |
| `--gradient-accumulation-steps` | 每次优化器更新前累积的 microbatch 数，默认 2 |
| `--epochs` | 训练轮数，默认 1 |
| `--stop-after-steps` | 本次执行的绝对 optimizer step 停止点，不改变原 epoch 预算 |

全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数，双卡默认 16。单卡保持全局 16 时使用 `--gradient-accumulation-steps 4`。

### 3.2 验证与 checkpoint 保存

| 参数 | 用途／默认值 |
|---|---|
| `--eval-every` | dev 验证间隔，默认 50 optimizer steps |
| `--save-every` | 正整数保存间隔或 `after_each_epoch`，默认 50 steps |
| `--save-total-limit` | 保留最近成功保存的 checkpoint 数，默认 2 |

epoch 结束和达到 `--stop-after-steps` 时也会验证、保存。例如，`--epochs 3 --save-every after_each_epoch --save-total-limit 3` 保留每轮末的 checkpoint。

### 3.3 独立评估与生成

| 参数 | 用途／默认值 |
|---|---|
| `--checkpoint` | 原运行的 `checkpoints/global_step_N/` 目录；恢复、评估和生成均必填 |
| `--benchmark` | 仅用于独立评估，默认 `null`（不传时为 QASPER test）；显式值为 `ruler` 或 `locomo` |
| `--data-dir` | 覆盖评估数据目录；QASPER 默认沿用训练目录，benchmark 默认位置见[第 1 节](#1-准备环境模型与数据) |
| `--lengths` | 仅用于 RULER，默认 `8192 16384 32768` |
| `--max-new-tokens` | QASPER test 或单条生成的 token 上限，默认 128；benchmark 不接受此参数 |
| `--output-dir` | 评估结果目录，默认 `eval/<method-dir>/test/`、`ruler/` 或 `locomo/` |
| `--prompt-file` | 单条生成的纯文本输入，必填 |
| `--output` | 单条生成的结果 JSON，默认 `eval/<method-dir>/greedy-sample.json` |

checkpoint 须保留原运行目录结构。模型和记忆设置从保存配置读取，不能通过评估或生成命令覆盖。

### 3.4 运行目录与跟踪

| 参数 | 用途／默认值 |
|---|---|
| `--gpus` | 物理 GPU，训练默认 `0,1`；生成和独立评估选择 `0` 或 `1` |
| `--run-id` | 新训练的实验系列编号，默认上海时间 `YYYYMMDD-HHMMSS` |
| `--output-root` | 新训练产物的父目录，默认 `artifacts/v4` |
| `--tracking` | `online` 或 `disabled`；新训练沿用配置，预设为在线，评估沿用原训练设置 |
| `--swanlab-project` | 新训练的 SwanLab 项目，默认 `latent-working-memory-v4` |
| `--group` | 新训练的实验分组，省略时沿用配置或当前 `run-id` |
| `--tag` | 新训练追加的命名空间标签，可重复传入，如 `--tag study:query-generation` |

默认 QASPER 配置包含 `data:qasper`，代码自动附加 `scope:main`、`method:attention-matching`。模型、优化器及局部更新的具体超参数保存在 JSON 和 SwanLab config 中。

## 4. 查看结果

产物位于 `artifacts/v4/<run-id>/`，`<method-dir>` 按 `<query_mode>-m<num_slots>` 生成，例如 `conditioned-m32`；SwanLab 名称为 `<method-dir>_<run-id>`。

```text
artifacts/v4/<run-id>/
├── plan/<method-dir>/            config.json、命令与执行状态、运行日志
├── train/<method-dir>/           run.json、tokenizer/、data-statistics.json、metrics.jsonl
│   ├── checkpoints/global_step_N/
│   ├── swanlab.json              在线运行身份
│   └── training-result.json      训练结果与最终 checkpoint
└── eval/<method-dir>/
    ├── greedy-sample.json        单条生成结果
    ├── test/                    QASPER 评估配置、逐题结果、汇总及发布记录
    └── ruler/、locomo/           benchmark 逐题结果、汇总及发布记录
```

- **评估产物**：逐题预测和评分汇总保存到 `--output-dir` 指定目录，主要文件为 `predictions.jsonl`、`summary.json`；QASPER 另存筛选统计，在线发布时保存 `publication.json`。运行日志和执行状态写入 `plan/<method-dir>/` 下的 `evaluation.log`／`evaluation-result.json`，benchmark 对应 `<benchmark>-evaluation.log`／`<benchmark>-evaluation-result.json`。
- **进度与图表**：终端展示 step、loss、耗时与显存，完整训练指标见 `metrics.jsonl`。训练、dev 和最终评估共用原 SwanLab run；`train/*`、`dev/*` 的写入 loss 图合并写入前后曲线，`evaluation/*` 展示 QASPER 指标及 RULER／LoCoMo 汇总。布局见 [v4 评估与记录](../README.md#四评估与记录)。
- **错误与状态**：训练查看 `plan/<method-dir>/train.log`、`result.json`；评估查看上述日志和状态文件，生成查看 `generate.log`、`generate-result.json`。状态从 `running` 更新为 `finished`／`stopped`／`failed`，保存命令、GPU、起止时间、退出码及失败原因。
- **Checkpoint 与恢复**：默认保留最近两个 checkpoint，恢复与评估均传目录路径；具体 step 可从训练结果或 checkpoint 目录取得。中断恢复见[第 2.3 节](#23-中断恢复)。
