# 20261013_v4 统一运行入口

创建时间：20261013 10:12:17 Asia/Shanghai（UTC+08:00）

最后修订时间：20261014 16:34:33 Asia/Shanghai（UTC+08:00）

使用 [run.sh](run.sh) 启动训练、恢复、QASPER test、benchmark 评估或单条生成。模型与训练目标见 [v4 说明](../README.md)，默认配置见 [experiment.json](../../../../configs/v4/attention_matching/experiment.json)。以下命令均从项目根目录执行。

## 1. 准备环境、模型与数据

- 使用与 v3 共用的根 `.venv/`，依赖由 `pyproject.toml` 和 `uv.lock` 管理；执行 `uv sync --frozen` 安装环境。默认依赖组为 `cuda124`，使用 Python 3.11、PyTorch 2.6.0、Transformers 4.57.6 和 verl 0.8.0。
- 模型预设为 `~/models/Qwen3-4B-Instruct-2507`。可用 `--model-path` 覆盖，或在 JSON 中设置 HF repo ID 和固定 `revision`。
- `training.dataset_dir` 默认 `data/delta-mem-qasper`，固定读取其中的 `train.jsonl`、`dev.jsonl`、`test.jsonl`；`--data-dir` 可在新训练或 QASPER test 评估时覆盖该目录。数据可从 [ModelScope](https://modelscope.cn/datasets/percyWeeei/latent-working-memory/files) 的 `delta-mem-qasper/` 获取。
- 数据格式、独立 QA 样本及长度上下限见 [v4 数据说明](../README.md#二数据)。划分文件保留全部样本，加载时才筛选；训练和评估均不截断输入。
- 从本地 raw 构造数据时执行：

```bash
uv run --frozen python -m latent_working_memory.data_preparation.delta_mem_qasper \
  --config configs/data_preparation/delta-mem-qasper.json
```

- 默认在线记录到 `latent-working-memory-v4`，使用与 v3 相同的凭据：有项目 `.env` 时读取其中的 `SWANLAB_API_KEY`，否则读取终端环境变量。加 `--tracking disabled` 仅保存本地产物，无需 key。

## 2. 新训练

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --config configs/v4/attention_matching/experiment.json \
  --gpus 0,1 \
  --run-id am-query_20261013
```

- 默认使用物理 GPU 0、1；单卡使用 `--gpus 0` 或 `--gpus 1`。
- 完成全部训练预算后自动执行最终 QASPER test 评估：等待训练进程退出，使用所选 GPU 中的第一张卡，生成上限为 128 tokens；在线结果追加到原 SwanLab run 的 `evaluation/*`。
- 显式 CLI 参数覆盖配置。相对数据、输出路径以运行目录为基准；`run.sh` 统一切到项目根目录，因此 JSON 和 CLI 使用相同的路径规则。
- `--run-id` 是唯一的用户命名入口，未指定时采用上海时间 `YYYYMMDD-HHMMSS`。方法目录自动使用 `<query_mode>-m<num_slots>`，例如 `conditioned-m32`；SwanLab 名称为 `<method-dir>_<run-id>`。
- 新训练的输出目录由入口派生，写入运行配置；预设 JSON 无需设置 `output_dir`。不覆盖已有运行目录。
- `--stop-after-steps 10` 在绝对 optimizer step 10 停止，执行 dev 验证和 checkpoint 保存，不改变原 epoch 预算；未完成全部预算时不执行 test。
- `--swanlab-project 项目名` 可覆盖配置中的项目。`group` 依次使用 CLI、配置或 run-id；可重复传入 `--tag study:名称`。默认 QASPER 配置包含 `data:qasper`。

## 3. 恢复训练

从 `train/<method-dir>/training-result.json` 的 `checkpoint` 字段取得实际路径，替换下例的 `global_step_N`：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --resume --gpus 0,1 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-m32/checkpoints/global_step_N
```

- 沿用原配置、训练预算、输出目录和 SwanLab 身份，不接受训练参数覆盖；可以设置新的绝对停止点 `--stop-after-steps N`。
- GPU 编号可以重新选择，卡数必须与原训练一致。
- 保留原 `plan/<method-dir>/config.json`、`run.json`、`tokenizer/`、checkpoint，以及在线运行的 `swanlab.json`。checkpoint 须位于原 `<实验系列>/train/<method-dir>/checkpoints/` 下。
- 恢复新增模块权重、optimizer、加载器状态和各 rank RNG；保存点在完整全局 batch 结束后，每条 QA 样本仍独立初始化记忆。
- 入口从 `run.json` 读取元数据，训练进程沿用原 `config.json`、检查运行配置一致性并加载 checkpoint；日志追加到 `train.log`。
- 续训完成全部预算后，同样自动执行最终 QASPER test 评估。

## 4. 独立 QASPER test 评估

为已完成的训练补评估，或以不同生成预算重新评估时，使用最后一个 checkpoint：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-m32/checkpoints/global_step_N \
  --max-new-tokens 128
```

- 模型配置、实际 revision 和 tokenizer 均取自原训练；默认读取 checkpoint 中 `dataset_dir` 下的 `test.jsonl`，可用 `--data-dir` 切换目录。生成预算及输出位置可单独设置。
- 默认输出 `eval/<method-dir>/test/`；输出目录必须不存在，重跑时用 `--output-dir` 指定新目录。
- 按同一数据接口展开 QA，检查原始 test 正文与训练时的 train/dev 不重叠，应用固定 prompt 下限和模型窗口上限。
- 分别从初始记忆执行答案 NLL 与仅 prompt 的 greedy 生成。生成时仍进行局部 TTT，参考答案仅用于 NLL 和结果评分。
- `--max-new-tokens` 是生成预算，不是输入筛选条件；实际预算受剩余模型窗口限制。每题保存实际预算与停止原因。
- 本地输出 `data-statistics.json`、`predictions.jsonl` 和 `summary.json`；汇总包含完整评估配置、NLL／EM／F1、记忆容量及耗时／显存，逐题结果单独保存。评分口径见 [v4 评估说明](../README.md#四评估与记录)。
- 自动与独立评估共用本节流程。评估失败时查看 `plan/<method-dir>/evaluation.log` 和 `evaluation-result.json`；已完成的训练结果和 checkpoint 保留，无需重新训练。
- 默认沿用原训练是否在线记录的设置；`--tracking disabled` 仅保存本地。在线评估只向已完成训练的原 run 发布最终 checkpoint 结果，保存 `publication.json`；不改变训练配置或新建 run。

## 5. Benchmark 评估

```bash
# 下载 benchmarks，支持单独下载
bash src/latent_working_memory/v4/scripts/download_benchmarks.sh --benchmark all

# 为 RULER 创建实际用于评估的样本
bash src/latent_working_memory/v4/scripts/generate_ruler.sh \
  --model-path "~/models/Qwen3-4B-Instruct-2507" \
  --lengths 8192 16384 32768 \
  --num-samples 100 \
  --seed 42
```

- Benchmark 准备：使用 [download_benchmarks.sh](download_benchmarks.sh) 下载原始数据。LoCoMo 下载后可直接用于评估，RULER 需要进一步生成可用样本。
- RULER 后处理：使用 [generate_ruler.sh](generate_ruler.sh) 从素材生成样本，默认三个长度（8k，16k，32k）、每长度／任务 100 条。
- 环境依赖及参数填法见 [benchmark 数据准备说明](../benchmarks/README.md#2-手动准备数据)。

```bash
# LoCoMo 评估
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --benchmark locomo --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-m32/checkpoints/global_step_N

# RULER 评估
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --benchmark ruler --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-m32/checkpoints/global_step_N \
  --lengths 8192 16384 32768
```

- `--benchmark` 是 benchmark 评估时的必须参数，不传该参数时执行[第 4 节的 QASPER test 评估](#4-独立-qasper-test-评估)。
- benchmark 默认目录为 `data/benchmarks/<benchmark-name>/`，可用 `--data-dir` 覆盖。
- `--lengths` 是 RULER 评估专用参数，默认 8192、16384、32768。
- 每题独立初始化记忆，使用 greedy 生成和各任务规定的生成预算，不接受 `--max-new-tokens` 覆盖。
- 评估产物：逐题预测和评分汇总保存到 `--output-dir` 指定目录（默认 `eval/<method-dir>/<benchmark-name>/`），运行日志和执行状态分别写入 `plan/<method-dir>/` 目录下的 `<benchmark-name>-evaluation.log` 和 `<benchmark-name>-evaluation-result.json`。
- 在线结果追加到原训练 run，`--tracking disabled` 仅保存本地。

## 6. 单条生成

准备已组织好的 UTF-8 纯文本 prompt：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --generate-only --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/conditioned-m32/checkpoints/global_step_N \
  --prompt-file data/attention-matching/prompt.txt \
  --max-new-tokens 128
```

- 该入口直接编码文本，不额外应用 chat template；如需与 QA 训练提示一致，应准备对应的完整 prompt。
- 默认写入 `eval/<method-dir>/greedy-sample.json`，可用 `--output` 指定文件。结果包含 prompt、生成文本及 token IDs。
- prompt 与请求的新增 token 上限之和不得超过模型窗口；单条生成不计算 QA 指标。

## 7. 常用参数

**新训练的配置与执行**

| 参数 | 用途／默认值 |
|---|---|
| `--config` | 默认 `configs/v4/attention_matching/experiment.json` |
| `--model-path` | 本地基座模型位置；未传时使用配置 |
| `--data-dir` | 覆盖数据目录；固定使用 `train.jsonl`、`dev.jsonl`、`test.jsonl` |
| `--run-id`／`--output-root` | 实验系列及产物父目录；父目录默认 `artifacts/v4` |
| `--epochs`／`--seed` | 训练轮数与随机种子，预设 1／42；seed 允许 0 |
| `--micro-batch-size-per-gpu`／`--gradient-accumulation-steps` | 每卡单次前向样本数与梯度累积次数，预设 4／2 |
| `--max-seq-length` | train 完整输入的人为上限，预设 12288；`null` 关闭，固定下限与模型窗口始终生效 |

- 全局 batch = GPU 数 × 每卡 microbatch × 梯度累积步数。预设双卡全局 batch 为 16，每卡累积 2 次 microbatch 后更新优化器；尾批按实际有效答案 token 数归一化。
- 单卡沿用预设时全局 batch 为 8；若需保持 16，设置 `--gradient-accumulation-steps 4`。

**验证、保存与跟踪**

| 参数 | 用途／默认值 |
|---|---|
| `--eval-every` | dev 验证间隔，预设 50 optimizer steps |
| `--save-every` | 正整数保存间隔或 `after_each_epoch`，预设 50 |
| `--save-total-limit` | 保留最近成功保存的 checkpoint 数，预设 2 |
| `--stop-after-steps` | 本次训练／恢复的绝对停止点，不改变 epoch 预算 |
| `--tracking` | `disabled`／`online`；默认预设开启在线记录，独立评估沿用原训练设置 |
| `--swanlab-project`／`--group`／`--tag` | 项目、实验分组和可重复的命名空间标签；仅用于新训练，项目预设为 `latent-working-memory-v4` |

- epoch 结束及主动停止时始终验证、保存。仅按 epoch 保存可使用 `--save-every after_each_epoch`。
- 模型、记忆与局部更新参数直接修改 JSON，见 [v4 配置说明](../README.md#三配置与执行)。
- 生成与评估必须显式选一张卡。`--resume`、`--generate-only`、`--evaluate-only` 互斥，均须提供 `--checkpoint`。`--benchmark` 仅用于 `--evaluate-only`，默认 `null`；显式传入时仅接受 `ruler` 或 `locomo`。
- 完整参数：`bash src/latent_working_memory/v4/scripts/run.sh --help`。

## 8. 产物与进度

```text
artifacts/v4/<run-id>/
├── plan/<method-dir>/
│   ├── config.json                         运行配置
│   ├── result.json                         训练／恢复命令与状态
│   ├── generate-result.json                生成命令与状态
│   ├── evaluation-result.json              评估命令与状态
│   ├── <benchmark>-evaluation-result.json  benchmark 命令与状态
│   └── train.log、generate.log、evaluation.log
├── train/<method-dir>/
│   ├── run.json、tokenizer/                运行配置与 tokenizer
│   ├── data-statistics.json、metrics.jsonl 数据筛选与完整指标
│   ├── checkpoints/global_step_N/         新增权重、optimizer、加载器与 RNG
│   ├── swanlab.json                        在线运行身份
│   └── training-result.json                训练结果与最后 checkpoint
└── eval/<method-dir>/
    ├── greedy-sample.json
    ├── test/                              评估配置、逐题结果、汇总及发布记录
    └── ruler/、locomo/                    benchmark 逐题结果、汇总及发布记录
```

- 状态文件保存执行命令、GPU 和起止时间；训练与评估分别记录在 `result.json`、`evaluation-result.json`。启动时为 `running`，结束后更新为 `finished`／`stopped`／`failed`，失败时记录原因与子进程退出码。
- 终端展示简洁的 epoch、step、loss、耗时及显存；完整指标保存在 `metrics.jsonl`。训练与 dev 追加到同一 SwanLab run。
- 新训练及续训完成后自动执行 QASPER test；`--evaluate-only` 和 `--generate-only` 只执行对应动作。
