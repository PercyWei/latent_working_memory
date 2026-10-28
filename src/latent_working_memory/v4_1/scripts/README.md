# 20261027_v4.1 运行指南

创建时间：20261027 16:34:06 Asia/Shanghai（UTC+08:00）

最后修订时间：20261028 20:38:00 Asia/Shanghai（UTC+08:00）

计算协议、数据格式与参数含义见[方法说明](../README.md)。以下命令使用项目根目录 `.venv`；训练采用单卡、microbatch 1、梯度累积 8，离线准备支持多卡。正式实验先检查[配置](../../../../configs/v4-1/online_ttt_memory.json)中的基座与数据路径。

## 1. 环境与入口

```bash
uv sync
bash src/latent_working_memory/v4_1/scripts/run.sh --help
```

`run.sh` 切换到项目根目录，使用 `uv run --frozen`，直接继承 `CUDA_VISIBLE_DEVICES`。离线准备自动使用所有可见 GPU；训练与评估保持单进程，使用首张可见 GPU（进程内 `cuda:0`）。本项目选择物理卡 0、1，不使用 `torchrun` 或额外设备参数。模块参数原样传递，可在操作名后使用 `--help`。

本文示例约定：

| 内容 | 路径 |
|---|---|
| 基座与 tokenizer | `~/models/Qwen3-4B-Instruct-2507` |
| QA 数据 | `data/delta-mem-qasper/{train,dev,test}.jsonl` |
| Teacher 缓存 | `artifacts/v4-1/online-ttt-memory/teacher` |
| 训练目录 | `artifacts/v4-1/online-ttt-memory/train/recent-m256` |
| 独立评估 | `artifacts/v4-1/online-ttt-memory/eval/<任务名>` |

## 2. 离线准备 teacher

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v4_1/scripts/run.sh prepare \
  --config configs/v4-1/online_ttt_memory.json \
  --output-dir artifacts/v4-1/online-ttt-memory/teacher \
  --batch-size 4 \
  --splits train dev
```

每张卡加载一份冻结 teacher，按预计计算量分配文档，同篇文档的所有边界由同一进程处理。原文 prefill 每次处理一篇；`--batch-size` 控制每卡 teacher 生成与统计的批处理大小，覆盖本次运行的 `teacher.batch_size`。省略参数时沿用配置，默认 4。

每个文档、每个压缩边界固定 480 条提示的响应、4096 个候选位置及完整池上的逐层尺度，并输出进度。边界缓存只保存响应 token IDs、偏移、位置与尺度，放在 `documents/<document_id>/write_<j>.safetensors`；全局提示 token IDs 与身份约束保存在 `identity.json`。同文档 QA 共用缓存；同配置重复执行会复用已有边界，train/dev 也可分开准备。

尺度统计并入响应生成：逐层将局部 attention 输出缩为通道均值与方差，并保留 log-mass；每批完成后按原提示／位置顺序维护 4096 位置候选池，最后计算尺度。最后生成的 token（含 EOS）补一次前向取得 query，不继续生成答案。

GPU 仅暂存当前批统计，CPU 固定候选池的三项 FP64 统计约占 108 MiB／进程（36 层、32 query heads），不含模型、KV 与当前批工作区。训练时仍从池中抽取 128 个位置，通过 teacher forcing 重算当次 AM 目标，尺度沿用离线结果。

`--output-dir` 仅指定本次写入位置；如改用其他目录，同步修改训练配置的 `training.teacher_cache_dir`。Teacher 不需要覆盖 test 或 benchmark。

## 3. 训练与恢复

新训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v4_1/scripts/run.sh train \
  --config configs/v4-1/online_ttt_memory.json \
  --batch-size 4 \
  --output-dir artifacts/v4-1/online-ttt-memory/train/recent-m256
```

Train/dev 在同一卡上加载独立冻结 teacher 与 student，顺序执行。每轮均匀、无放回抽取 128 个位置，只回放涉及分支到最远选中位置，重算 AM 参照；teacher 原文 prefix KV 在当前 QA episode 内递增，跨 QA／epoch 重算。

训练的 `--batch-size` 控制 teacher forcing 批量，不改变训练 microbatch；省略时沿用配置，默认 4。它可与离线准备的批量不同，缓存仍可复用。覆盖值写入训练目录的实际配置、checkpoint 与 SwanLab config。

从同一 run 的已保存 step 恢复，例如第 100 步：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v4_1/scripts/run.sh train \
  --config artifacts/v4-1/online-ttt-memory/train/recent-m256/config.json \
  --output-dir artifacts/v4-1/online-ttt-memory/train/recent-m256 \
  --resume artifacts/v4-1/online-ttt-memory/train/recent-m256/checkpoints/global_step_100
```

恢复会还原慢参数、optimizer、scheduler、数据顺序、查询采样及各随机状态；实际配置、数据和 teacher 缓存必须与 checkpoint 绑定一致，包括 teacher 批处理大小、`max_steps` 和输出目录。示例直接读取已保存的实际配置；若使用原配置文件，须传入与首次训练相同的 `--batch-size` 覆盖值。

Checkpoint 只在 optimizer step 完成后保存，默认保留最近两个；恢复须指定本 run 最新完整 checkpoint，示例适用于最新已完成第 100 步的情况。改变方法或训练设置时使用新训练目录。

| 训练产物 | 内容 |
|---|---|
| `config.json` | 本次实际配置 |
| `data.json` | 数据选样与身份 |
| `metrics.jsonl` | 每个 optimizer step 一行的 train/dev 指标；恢复时截至 checkpoint step |
| `summary.json` | 最终 `step`、`checkpoint`、`output_dir`；checkpoint 为 `state.pt` 绝对路径 |
| `checkpoints/global_step_<N>/state.pt` | 慢参数、配置、绑定身份与完整恢复状态，作为恢复及评估依据 |
| `swanlab.json` | 启用上传时保存 `id/url`，恢复与评估续记所需 |

字段、索引与单位统一见[方法说明第 4 节](../README.md#4-代码与结果)。展示报告可以重建，checkpoint 和原始记录应保留。

进度直接打印到终端。`metrics.jsonl` 是可供分析的结构化指标，不另存终端输出副本。默认开启 SwanLab，使用既有 `latent-working-memory-v4` 项目，正式配置显式指定 `group: "online-ttt-memory_20261028"`；train/dev 在同一 run 内记录，独立评估续记该 run。运行前配置 SwanLab 登录凭据；新实验系列修改配置中的 group，关闭上传时设 `tracking: "disabled"`。

## 4. 独立评估与报告

以下使用默认训练结束时的第 1000 步 checkpoint。

### 4.1 QASPER test

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v4_1/scripts/run.sh evaluate \
  --checkpoint artifacts/v4-1/online-ttt-memory/train/recent-m256/checkpoints/global_step_1000 \
  --task test --data data/delta-mem-qasper \
  --output-dir artifacts/v4-1/online-ttt-memory/eval/qasper-test \
  --max-new-tokens 128
```

独立样本重新初始化 memory，先处理正文，再固定 memory 回答问题。评估只加载 student，不读取 teacher 缓存，报告单参考答案 EM／token F1 和按答案 token 加权的 NLL；NLL 包含 assistant 结束符，与训练时先逐样本平均的汇总口径区分。

### 4.2 LoCoMo 与 RULER

使用项目已有的 benchmark 数据准备结果，数据布局与准备入口见[公共 benchmark 指南](../../v4/benchmarks/README.md)。将下面的 `<已准备数据目录>` 替换为对应实际目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v4_1/scripts/run.sh evaluate \
  --checkpoint artifacts/v4-1/online-ttt-memory/train/recent-m256/checkpoints/global_step_1000 \
  --task locomo --data '<已准备数据目录>' \
  --output-dir artifacts/v4-1/online-ttt-memory/eval/locomo

CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v4_1/scripts/run.sh evaluate \
  --checkpoint artifacts/v4-1/online-ttt-memory/train/recent-m256/checkpoints/global_step_1000 \
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

离线缓存以 token IDs、索引与固定尺度为主。按 36 层、32 query heads、2571 个边界估算，每条响应均达到 24 tokens 时，主要张量总量约为 439 MB。GPU 数量和执行批量不改变缓存格式，已有结果直接复用；重新生成时可能因浮点计算产生数值差异。训练需要额外的 teacher 前向，显存包含两份模型权重、teacher prefix KV 与 student 元训练图；实际成本和效果以运行结果为准。

## 6. 从逐位置张量缓存迁移

旧缓存若保存 `queries`、`outputs` 和 `log_mass`，不能直接用于当前实现。保留原目录，选择新的缓存目录（如 `artifacts/v4-1/online-ttt-memory/teacher-tokens`），将配置的 `training.teacher_cache_dir` 改为该路径，再用第 2 节命令将 `--output-dir` 指向同一路径重新准备。

已有训练 checkpoint 绑定旧缓存身份，迁移后使用新的训练输出目录开始训练，不以旧 checkpoint 执行 `--resume`。模型超参数、480／24／4096／128 查询预算及 CLI 入口保持不变。
