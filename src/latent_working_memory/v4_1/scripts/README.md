# 20261027_v4.1 运行指南

创建时间：20261027 16:34:06 Asia/Shanghai（UTC+08:00）

最后修订时间：20261027 16:50:38 Asia/Shanghai（UTC+08:00）

计算协议、数据格式与参数含义见[方法说明](../README.md)。以下命令使用项目根目录 `.venv`，默认单卡、microbatch 1、梯度累积 8；正式实验先检查[配置](../../../../configs/v4-1/online_ttt_memory.json)中的基座与数据路径。

## 1. 环境与入口

```bash
uv sync
bash src/latent_working_memory/v4_1/scripts/run.sh --help
```

`run.sh` 切换到项目根目录，使用 `uv run --frozen`。`--gpu 0`／`--gpu 1` 指定物理卡，进程内均为 `cuda:0`；默认物理卡 0。模块参数原样传递，可在操作名后使用 `--help`。当前入口支持单进程，不使用 `torchrun`。

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
bash src/latent_working_memory/v4_1/scripts/run.sh prepare --gpu 0 \
  --config configs/v4-1/online_ttt_memory.json \
  --output-dir artifacts/v4-1/online-ttt-memory/teacher \
  --device cuda:0 --splits train dev
```

每个文档、每个压缩边界准备完整候选池，并输出进度。缓存放在 `documents/<document_id>/write_<j>.safetensors`，身份约束保存在 `identity.json`。同文档 QA 共用缓存；同配置重复执行会复用已有边界，train/dev 也可分开准备。

`--output-dir` 仅指定本次写入位置；如改用其他目录，同步修改训练配置的 `training.teacher_cache_dir`。Teacher 不需要覆盖 test 或 benchmark。

## 3. 训练与恢复

新训练：

```bash
bash src/latent_working_memory/v4_1/scripts/run.sh train --gpu 0 \
  --config configs/v4-1/online_ttt_memory.json \
  --output-dir artifacts/v4-1/online-ttt-memory/train/recent-m256 \
  --device cuda:0
```

从同一 run 的已保存 step 恢复，例如第 100 步：

```bash
bash src/latent_working_memory/v4_1/scripts/run.sh train --gpu 0 \
  --config configs/v4-1/online_ttt_memory.json \
  --output-dir artifacts/v4-1/online-ttt-memory/train/recent-m256 \
  --device cuda:0 \
  --resume artifacts/v4-1/online-ttt-memory/train/recent-m256/checkpoints/global_step_100
```

恢复会还原慢参数、optimizer、scheduler、数据顺序、查询采样及各随机状态；配置、数据和 teacher 缓存必须与 checkpoint 绑定一致，包括 `max_steps` 和输出目录。Checkpoint 只在 optimizer step 完成后保存，默认保留最近两个；恢复须指定本 run 最新完整 checkpoint，示例适用于最新已完成第 100 步的情况。改变方法或训练设置时使用新训练目录。

| 训练产物 | 内容 |
|---|---|
| `config.json` | 本次实际配置 |
| `data.json` | 数据选样与身份 |
| `metrics.jsonl` | 按 optimizer step 追加的 train/dev 指标 |
| `summary.json` | 训练汇总 |
| `checkpoints/global_step_<N>/state.pt` | 慢参数与完整恢复状态 |

进度直接打印到终端。`metrics.jsonl` 是可供分析的结构化指标，不另存终端输出副本。默认关闭 SwanLab；需要上传时在配置中设 `tracking: "swanlab"` 并给出 `group`，使用既有 `latent-working-memory-v4` 项目；train/dev 在同一 run 内记录。

## 4. 独立评估与报告

以下使用默认训练结束时的第 1000 步 checkpoint。

### 4.1 QASPER test

```bash
bash src/latent_working_memory/v4_1/scripts/run.sh evaluate --gpu 0 \
  --checkpoint artifacts/v4-1/online-ttt-memory/train/recent-m256/checkpoints/global_step_1000 \
  --task test --data data/delta-mem-qasper \
  --output-dir artifacts/v4-1/online-ttt-memory/eval/qasper-test \
  --device cuda:0 --max-new-tokens 128
```

独立样本重新初始化 memory，先处理正文，再固定 memory 回答问题。评估不读取 teacher 缓存，报告单参考答案 EM／token F1 和按答案 token 加权的 NLL；NLL 包含 assistant 结束符，与训练时先逐样本平均的汇总口径区分。

### 4.2 LoCoMo 与 RULER

使用项目已有的 benchmark 数据准备结果，数据布局与准备入口见[公共 benchmark 指南](../../v4/benchmarks/README.md)。将下面的 `<已准备数据目录>` 替换为对应实际目录：

```bash
bash src/latent_working_memory/v4_1/scripts/run.sh evaluate --gpu 0 \
  --checkpoint artifacts/v4-1/online-ttt-memory/train/recent-m256/checkpoints/global_step_1000 \
  --task locomo --data '<已准备数据目录>' \
  --output-dir artifacts/v4-1/online-ttt-memory/eval/locomo \
  --device cuda:0

bash src/latent_working_memory/v4_1/scripts/run.sh evaluate --gpu 0 \
  --checkpoint artifacts/v4-1/online-ttt-memory/train/recent-m256/checkpoints/global_step_1000 \
  --task ruler --data '<已准备数据目录>' --lengths 8192 16384 32768 \
  --output-dir artifacts/v4-1/online-ttt-memory/eval/ruler \
  --device cuda:0
```

Benchmark 默认采用原任务 prompt 与生成预算，完整 prompt 在线压缩后固定 memory 生成，沿用公共评分。`--max-new-tokens` 可显式覆盖生成预算，并记录在评估协议中。`--max-samples` 对 RULER 是每个任务／长度的上限，对其他数据是总体 QA 上限；正式结果保留默认完整评估。

### 4.3 查看与重新生成报告

每次评估输出 `summary.json`、`details.jsonl` 和 `report/index.html`。汇总查看任务得分、生成与写入统计；逐题结果保留原始输出，可在 HTML 中查看显式换行、特殊空白和异常字符。原训练启用 SwanLab 时，评估追加到同一 run 的 `evaluation/*`。

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
| 仅重新展示既有评估 | 运行 `report`，不执行模型 |

GPU 训练、teacher 准备和 benchmark 的实际显存、耗时及效果尚需正式运行确认。离线缓存按层与位置保存 FP32 监督，磁盘预算随文档数与压缩轮数增长。
