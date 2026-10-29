# 20261027_v4.1 运行指南

创建时间：20261027 16:34:06 Asia/Shanghai（UTC+08:00）

最后修订时间：20261029 15:21:11 Asia/Shanghai（UTC+08:00）

计算协议、数据格式与参数含义见[方法说明](../README.md)。以下命令使用项目根目录 `.venv`；训练采用单卡、microbatch 1、梯度累积 8，离线准备支持多卡。正式实验先检查[配置](../../../../configs/v4-1/online_ttt_memory.json)中的基座与数据路径。

## 1. 环境与入口

```bash
uv sync --frozen
bash src/latent_working_memory/v4_1/scripts/run.sh --help
```

默认 BF16 离线准备使用 Linux x86_64、支持 BF16 的 NVIDIA GPU（SM80+）。默认 `cuda124` 依赖组固定 PyTorch 2.6／CUDA 12.4 与 FlashInfer `0.2.5+cu124torch2.6`，由上述命令安装；`torch214` 组不适用于此准备入口。

`run.sh` 切换到项目根目录，使用 `uv run --frozen`，直接继承 `CUDA_VISIBLE_DEVICES`。离线准备自动使用所有可见 GPU；训练与评估保持单进程，使用首张可见 GPU（进程内 `cuda:0`）。本项目选择物理卡 0、1，不使用 `torchrun` 或额外设备参数。模块参数原样传递，可在操作名后使用 `--help`。

本文示例约定：

| 内容 | 路径 |
|---|---|
| 基座与 tokenizer | `~/models/Qwen3-4B-Instruct-2507` |
| QA 数据 | `data/delta-mem-qasper/{train,dev,test}.jsonl` |
| Teacher 缓存 | `artifacts/v4-1/online-ttt-memory/teacher/p1024-r256` |
| 训练目录 | `artifacts/v4-1/online-ttt-memory/train/am-recent_m256-d512_p1024-r256_inner3_20261029` |
| 独立评估 | `artifacts/v4-1/online-ttt-memory/eval/<任务名>` |

## 2. 离线准备 teacher

为 train/dev 准备 teacher 参考缓存；test 和 benchmark 无需此步骤。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v4_1/scripts/run.sh prepare \
  --config configs/v4-1/online_ttt_memory.json \
  --batch-size 4 \
  --splits train dev
```

- **并行与批量**：每张可见 GPU 加载一份 teacher。`--batch-size` 控制每卡批量；省略时沿用配置，默认 4。
- **数据划分**：`--splits` 指定 train/dev，可分别运行。
- **缓存复用**：同文档 QA 共用缓存；同格式、同配置的已完成边界直接复用。
- **进度显示**：汇总文档／已保存边界进度（含缓存命中）及每卡状态；终端原地刷新，重定向输出时每 30 秒打印摘要。

输出目录优先级：`--output-dir` > 配置中的非空 `training.teacher_cache_dir` > 自动目录。配置默认 `null`，prepare 与 train 自动使用：

```text
artifacts/v4-1/online-ttt-memory/teacher/p{pending_size}-r{recent_size}
```

修改窗口后自动切换目录，需重新准备缓存。若用 `--output-dir <目录>` 自定义路径，后续训练须将 `training.teacher_cache_dir` 设为同一路径。

| 缓存产物 | 内容 |
|---|---|
| `identity.json` | 全局提示文本与 token IDs、缓存配置身份 |
| `documents/<document_id>/write_<j>.safetensors` | 第 j 次压缩的响应 token IDs、偏移、候选位置与标准化尺度 |

计算过程与张量格式见[离线参考池说明](../README.md#22-离线参考池)。

## 3. 训练与恢复

### 3.1 启动训练

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v4_1/scripts/run.sh train \
  --config configs/v4-1/online_ttt_memory.json \
  --micro-batch-size-per-gpu 1 \
  --gradient-accumulation-steps 8 \
  --teacher-batch-size 4
```

当前仅支持单卡训练，teacher 与 student 在同一卡上分别加载。以下参数省略时沿用配置：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--micro-batch-size-per-gpu` | 1 | 每个 microbatch 的 QA 样本数，当前仅支持 1 |
| `--gradient-accumulation-steps` | 8 | 每次 optimizer step 累积的 microbatch 数 |
| `--teacher-batch-size` | 4 | Teacher forcing 每批处理的提示／响应分支数 |

默认有效训练 batch 为 $1\times8=8$ 个样本，epoch 尾批可能更小；teacher 批量不计入训练 batch。

输出目录默认按方法设置与启动日期构造：

```text
artifacts/v4-1/online-ttt-memory/train/am-recent_m{num_slots}-d{memory_dim}_p{pending_size}-r{recent_size}_inner{inner_steps}_{YYYYMMDD}
```

- **目录优先级**：`--output-dir` > 非空 `training.output_dir` > 自动目录。
- **日期与路径**：按 `Asia/Shanghai` 时区的启动日期命名；最终绝对路径在启动时打印并写入实际配置。下文以 `20261029` 启动的 run 为例。
- **重复运行**：新训练遇到非空目录会报错；继续原 run 使用 `--resume`，新实验使用其他目录。

### 3.2 恢复训练

使用已保存的实际配置，指定本 run 最新完整 checkpoint；以下以第 100 步为例：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v4_1/scripts/run.sh train \
  --config artifacts/v4-1/online-ttt-memory/train/am-recent_m256-d512_p1024-r256_inner3_20261029/config.json \
  --resume artifacts/v4-1/online-ttt-memory/train/am-recent_m256-d512_p1024-r256_inner3_20261029/checkpoints/global_step_100
```

- **恢复状态**：慢参数、optimizer、scheduler、数据游标及随机状态。
- **一致性要求**：实际配置（含命令行覆盖值）、数据、teacher 缓存与输出目录须匹配 checkpoint；跨日期恢复沿用原目录。修改方法或训练设置应启动新 run。
- **保存规则**：仅在 optimizer step 完成后保存，默认保留最近两个 checkpoint。

### 3.3 产物与记录

| 训练产物 | 内容 |
|---|---|
| `config.json` | 实际配置，包含解析后的路径与命令行覆盖值 |
| `data.json` | 数据选样与绑定身份 |
| `metrics.jsonl` | 每个 optimizer step 的 train/dev 指标；恢复时保留至 checkpoint step |
| `summary.json` | 最终 step、checkpoint 与 output_dir |
| `checkpoints/global_step_<N>/state.pt` | 慢参数与完整恢复状态，供恢复和评估使用 |
| `swanlab.json` | SwanLab run 的 id/url，供恢复和评估续记 |

进度打印到终端，逐步指标写入 `metrics.jsonl`。完整格式见[方法说明第 4 节](../README.md#4-代码与结果)。

SwanLab 默认开启，运行前需完成登录配置：

- **项目与名称**：项目为 `latent-working-memory-v4`，run 名称使用训练目录名。
- **实验分组**：`training.group` 显式指定；当前配置为 `online-ttt-memory_20261028`，新实验系列应相应修改。
- **指标组织**：train/dev 记录在同一 run，独立评估续记 `evaluation/*`；设 `tracking: "disabled"` 可关闭上传。

## 4. 独立评估与报告

以下使用默认训练结束时的第 1000 步 checkpoint。

### 4.1 QASPER test

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v4_1/scripts/run.sh evaluate \
  --checkpoint artifacts/v4-1/online-ttt-memory/train/am-recent_m256-d512_p1024-r256_inner3_20261029/checkpoints/global_step_1000 \
  --task test --data data/delta-mem-qasper \
  --output-dir artifacts/v4-1/online-ttt-memory/eval/qasper-test \
  --max-new-tokens 128
```

独立样本重新初始化 memory，先处理正文，再固定 memory 回答问题。评估只加载 student，不读取 teacher 缓存，报告单参考答案 EM／token F1 和按答案 token 加权的 NLL；NLL 包含 assistant 结束符，与训练时先逐样本平均的汇总口径区分。

### 4.2 LoCoMo 与 RULER

使用项目已有的 benchmark 数据准备结果，数据布局与准备入口见[公共 benchmark 指南](../../v4/benchmarks/README.md)。将下面的 `<已准备数据目录>` 替换为对应实际目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v4_1/scripts/run.sh evaluate \
  --checkpoint artifacts/v4-1/online-ttt-memory/train/am-recent_m256-d512_p1024-r256_inner3_20261029/checkpoints/global_step_1000 \
  --task locomo --data '<已准备数据目录>' \
  --output-dir artifacts/v4-1/online-ttt-memory/eval/locomo

CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v4_1/scripts/run.sh evaluate \
  --checkpoint artifacts/v4-1/online-ttt-memory/train/am-recent_m256-d512_p1024-r256_inner3_20261029/checkpoints/global_step_1000 \
  --task ruler --data '<已准备数据目录>' --lengths 8192 16384 32768 \
  --output-dir artifacts/v4-1/online-ttt-memory/eval/ruler
```

Benchmark 默认采用原任务 prompt 与生成预算，完整 prompt 在线压缩后固定 memory 生成，沿用公共评分。`--max-new-tokens` 可显式覆盖生成预算，并记录在评估协议中。`--max-samples` 对 RULER 是每个任务／长度的上限，对其他数据是总体 QA 上限；正式结果保留默认完整评估。

### 4.3 查看与重新生成报告

每次评估输出 `summary.json`、`details.jsonl` 和 `report/index.html`。Summary 保存实验身份与汇总，details 每题一行保存原始输出；HTML 可重建，并明确显示换行、特殊空白和异常字符。Summary 的 `checkpoint` 指向绝对 `state.pt` 路径，`training_run` 记录训练目录与 SwanLab 身份快照。原训练启用 SwanLab 时，评估通过训练目录的 `swanlab.json` 续记同一 run 的 `evaluation/*`。

已有结果无需重新运行模型即可生成报告：

```bash
bash src/latent_working_memory/v4_1/scripts/run.sh report \
  --results artifacts/v4-1/online-ttt-memory/eval/qasper-test
```

`--output <文件.html>` 可改变报告位置。报告生成不使用 GPU。

## 5. 配置变更与缓存复用

| 变更 | 操作 |
|---|---|
| Teacher／tokenizer、输入模板、提示库、窗口边界、响应预算、候选池、尺度下限、seed | 使用新缓存目录重新准备，再启动新训练 |
| 扩大数据选样 | 为新增文档补齐同身份缓存；新训练记录实际数据身份 |
| Slots／维度、解码器、LoRA、inner 步数／初始步长、outer 优化设置 | 新训练目录；teacher 身份未变时可复用缓存 |
| 每次 outer AM 采样位置数 | 新训练目录；不超过原池容量时可复用完整池 |
| 离线 GPU 数量、`teacher.batch_size` | 可复用缓存；恢复训练仍遵循完整配置一致的要求 |
| 仅重新展示既有评估 | 运行 `report`，不执行模型 |

离线缓存以 token IDs、索引与固定尺度为主。按 36 层、32 query heads、2571 个边界估算，每条响应均达到 24 tokens 时，主要张量总量约为 439 MB。同格式、同配置的已有结果直接复用；不同 attention 内核、GPU 或批量的数值舍入可能影响重新生成的响应与尺度。训练需要额外的 teacher 前向，显存包含两份模型权重、teacher prefix KV 与 student 元训练图；实际成本和效果以运行结果为准。

## 6. 从逐位置张量缓存迁移

旧缓存若保存 `queries`、`outputs` 和 `log_mass`，不能直接用于当前实现。保留原目录，按第 2 节命令在对应窗口目录重新准备（如 `artifacts/v4-1/online-ttt-memory/teacher/p1024-r256`）；若该路径已存旧格式缓存，先将旧目录另存。

已有训练 checkpoint 绑定旧缓存身份，迁移后使用新的训练输出目录开始训练，不以旧 checkpoint 执行 `--resume`。模型超参数、480／24／4096／128 查询预算及 CLI 入口保持不变。
