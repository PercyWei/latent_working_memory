# 20261027_v4.1 运行指南

创建时间：20261027 16:34:06 Asia/Shanghai（UTC+08:00）

最后修订时间：20261031 15:33:32 Asia/Shanghai（UTC+08:00）

计算协议与数据格式见[方法说明](../README.md)，命令行参数集中见[第 5 节](#5-调整参数)。以下命令使用项目根目录 `.venv`；正式实验先检查[配置](../../../../configs/v4-1/online_ttt_memory.json)中的基座与数据路径。

## 1. 环境与入口

```bash
uv sync --frozen
bash src/latent_working_memory/v4_1/scripts/run.sh --help
```

默认 BF16 离线准备使用 Linux x86_64、支持 BF16 的 NVIDIA GPU（SM80+）。默认 `cuda124` 依赖组固定 PyTorch 2.6／CUDA 12.4 与 FlashInfer `0.2.5+cu124torch2.6`，由上述命令安装；`torch214` 组不适用于此准备入口。

`run.sh` 默认执行完整运行，使用 `--train-only` 仅训练、`--resume` 恢复、`--evaluate-only` 仅评估；离线准备使用 `prepare`，评估看板通过 `dashboard.sh` 启动。各入口支持 `--help`。

脚本切换到项目根目录，使用 `uv run --frozen`，直接继承 `CUDA_VISIBLE_DEVICES`。离线准备自动分配到所有可见 GPU；训练通过 `torchrun` 为每张可见卡启动一个进程；评估在可见 GPU 间分片同一 checkpoint 的待评估样本，多个 checkpoint 按 step 依次执行。本项目双卡设为 `0,1`，单卡设为 `0` 或 `1`，无需额外设备参数。

本文示例约定：

| 内容 | 路径 |
|---|---|
| 基座与 tokenizer | `~/models/Qwen3-4B-Instruct-2507` |
| QA 数据 | `data/delta-mem-qasper/{train,dev,test}.jsonl` |
| 产物父目录 | `artifacts/v4-1` |

Teacher 缓存由不同 run 共享；运行目录与各阶段产物见 [第 7 节](#7-产物目录与记录)。

## 2. 离线准备 teacher

为 train/dev 准备 teacher 参考缓存；test 和 benchmark 无需此步骤。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v4_1/scripts/run.sh prepare \
  --config configs/v4-1/online_ttt_memory.json \
  --prepare-data data/delta-mem-qasper \
  --prepare-batch-size-per-gpu 64
```

- **并行与批量**：每张可见 GPU 加载一份 teacher，`--prepare-batch-size-per-gpu` 控制每卡离线批量。
- **缓存复用**：同文档 QA 共用缓存；同格式、同配置的已完成边界直接复用。
- **进度显示**：汇总文档／已保存边界进度（含缓存命中）及每卡状态；终端原地刷新，重定向输出时每 30 秒打印摘要。

Teacher 缓存目录按产物父目录与窗口设置自动创建：

```text
<output_root>/teacher/p{pending_size}-r{recent_size}
```

修改窗口后自动切换目录，需重新准备缓存。自定义 `--output-root` 时，prepare 与 train 使用同一父目录。

| 缓存产物 | 内容 |
|---|---|
| `identity.json` | 全局提示文本与 token IDs、缓存配置身份 |
| `documents/<document_id>/write_<j>.safetensors` | 第 j 次压缩的响应 token IDs、偏移、候选位置与标准化尺度 |

计算过程与张量格式见[离线参考池说明](../README.md#22-离线参考池)。

## 3. 完整运行与恢复

### 3.1 完整运行

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v4_1/scripts/run.sh \
  --config configs/v4-1/online_ttt_memory.json \
  --run-id 01_20261029 \
  --train-data 'data/delta-mem-qasper' \
  --epochs 4 \
  --micro-batch-size-per-gpu 2 \
  --gradient-accumulation-steps 8 \
  --teacher-batch-size-per-gpu 64
```

- **配置覆盖**：显式 CLI 参数覆盖 JSON；模型结构和记忆更新设置直接修改 JSON，参数说明见[第 5 节](#5-调整参数)。
- **训练批量**：`--micro-batch-size-per-gpu` 控制每卡正文与 TTT 写入的并行样本数；全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数。`--teacher-batch-size-per-gpu` 独立控制 teacher forcing 批量，不计入训练 batch。
- **Teacher 显存**：每个样本先准备全部固定参照，再执行 student；提示分支共享完整原文 KV，仅保留各自的短 suffix KV。Teacher 批量可独立调大，离线缓存可继续复用。
- **执行流程**：先保存训练与自动 test 计划；训练和 dev 成功结束、全部训练进程退出后，在可见 GPU 间按样本分片评估最后一个 checkpoint。Test 默认使用训练数据目录。

### 3.2 仅训练

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v4_1/scripts/run.sh \
  --train-only \
  --run-id 02_20261029
```

- `--train-only` 先保存训练计划，保留 dev 验证与 checkpoint 保存，不创建或执行自动 test 计划；不能同时传入 test 设置。

### 3.3 中断恢复

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v4_1/scripts/run.sh \
  --resume \
  --checkpoint 'artifacts/v4-1/am-recent_m256-d512_p1024-r256_inner3_01_20261029/train/checkpoints/global_step_<N>'
```

- **恢复起点**：仅接受当前 run 最新的完整 checkpoint。
- **恢复状态**：慢参数、optimizer、scheduler、全局数据游标及各 rank 的随机状态。
- **一致性要求**：原计划与 checkpoint 保存的配置须一致，不接受 CLI 配置覆盖；GPU 数量、数据和 teacher 缓存须保持一致。修改训练设置应启动新 run。
- **执行计划**：沿用原训练安排和已保存的 test 计划；训练已完成时只继续或复用最终 test，不重新训练。
- **保存规则**：仅在 optimizer step 完成后保存；保存频率与保留数量见[保存参数](#52-验证评估与结果记录)。

## 4. 独立评估

### 4.1 QASPER test

评估一个方法 run 当前保留的全部 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v4_1/scripts/run.sh \
  --evaluate-only \
  --eval-id auto \
  --eval-type test \
  --eval-run 'artifacts/v4-1/am-recent_m256-d512_p1024-r256_inner3_01_20261029/train'
```

也可显式选择一个或多个 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v4_1/scripts/run.sh \
  --evaluate-only \
  --eval-id auto \
  --eval-type test \
  --checkpoint \
    'artifacts/v4-1/am-recent_m256-d512_p1024-r256_inner3_01_20261029/train/checkpoints/global_step_<N>' \
    'artifacts/v4-1/am-recent_m256-d512_p1024-r256_inner3_01_20261029/train/checkpoints/global_step_<M>'
```

- Test 评估使用 `--eval-type test`（默认）。
- `--evaluate-only` 须搭配 `--eval-run` 或 `--checkpoint`，两者互斥。
- `--eval-run` 仅用于独立评估，指向 `<运行目录>/train/`，扫描当前保留且包含 `state.pt` 的 checkpoint。
- `--checkpoint` 仅接受属于同一 run 下的单个或多个 checkpoint。
- 独立评估的 `--eval-id` 默认 `YYYYMMDD-HHMMSS`；训练后的自动 test 默认使用 `auto`。同一标识下可包含 test、RULER 和 LoCoMo 等不同任务，各任务分别固定设置。
- 单个 checkpoint 的未完成样本分配到可见 GPU，各卡每次处理 1 个样本；多个 checkpoint 按 step 串行执行，当前项结束并汇总后再处理下一项。自动 test 使用同一流程，无需额外设置评估 batch size。
- 默认沿用评估配置 `<运行目录>/plan/eval-<eval-id>/test.json`。缺少该文件时，必须传入 `--test-data` 以创建评估配置，`--max-test-samples`、`--max-new-tokens` 可使用默认值。模型、方法和写入配置从 checkpoint 读取。
- 对选中的 checkpoint 进行评估时，已有完整且设置一致的结果直接复用，设置不一致则报错；未完成时复用逐题进度，重新分片剩余样本。GPU 数量只控制并行度，不绑定缓存。计算日志见各 checkpoint 的 `eval.log`，执行状态及错误见任务级 `status.json`。
- 评估仅保存本地结果并在终端输出重要指标，图表通过[评估结果与看板](#43-评估结果与看板)查看，不依赖 SwanLab。

### 4.2 Benchmark

使用项目已有的 benchmark 数据准备结果，数据布局与准备入口见[公共 benchmark 指南](../../v4/benchmarks/README.md)。

```bash
# LoCoMo
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v4_1/scripts/run.sh \
  --evaluate-only \
  --eval-id standard_20261030 \
  --eval-type benchmark \
  --benchmark locomo \
  --checkpoint 'artifacts/v4-1/am-recent_m256-d512_p1024-r256_inner3_01_20261029/train/checkpoints/global_step_<N>' \
  --benchmark-data '/data/benchmarks/locomo' \

# RULER
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v4_1/scripts/run.sh \
  --evaluate-only \
  --eval-id standard_20261030 \
  --eval-type benchmark \
  --benchmark ruler \
  --checkpoint 'artifacts/v4-1/am-recent_m256-d512_p1024-r256_inner3_01_20261029/train/checkpoints/global_step_<N>' \
  --benchmark-data 'data/benchmarks/ruler-v1/<model-name>' \
  --lengths 8192 16384 32768
```

- Benchmark 评估时 `--eval-type` 须设置为 `benchmark`，同时指定 `--benchmark`。
- `--eval-run` 和 `--checkpoint` 的基本用法同 test。
- 默认沿用评估配置 `<运行目录>/plan/eval-<eval-id>/<benchmark>.json`。缺少该文件时，必须传入 `--benchmark-data` 以创建评估配置，`--lengths`（RULER 专属）可使用默认值。模型、方法和写入配置从 checkpoint 读取。
- 结果保存与复用及多 GPU 并行方式同 test 评估。
- 同一 `eval-id` 的 RULER `--lengths` 允许保持原集合或扩展为超集，顺序无关；扩展后更新目标长度列表，复用每个 checkpoint 已完成的长度及逐题进度，仅对缺失样本分片计算。已有长度的样本及其余设置须一致，缩减或替换集合时报错。

### 4.3 评估结果与看板

下表路径均相对于运行根目录：

| 评估产物 | 内容 |
|---|---|
| `plan/eval-<eval-id>/<task>.json` | 该任务固定的数据与评估设置 |
| `eval/<eval-id>/<task>/global_step_<N>/eval.json` | 当前 checkpoint 的评估设置快照 |
| `eval/<eval-id>/<task>/global_step_<N>/summary.json` | 实验身份、协议与指标汇总 |
| `eval/<eval-id>/<task>/global_step_<N>/details.jsonl` | 按原始样本顺序汇总的回答与评分 |
| `eval/<eval-id>/<task>/global_step_<N>/work.json` | 本次执行的待评估样本分工 |
| `eval/<eval-id>/<task>/global_step_<N>/progress/worker_<rank>.jsonl` | 各分片逐题保存的完成进度与资源记录 |

各分片逐题保存进度；只有样本完整并通过校验后才提交最终 summary 与 details，部分进度不表示评估完成。RULER 在 `lengths/<长度>/` 保存已完整覆盖长度的 summary 与 details，以支持增量评估。

Summary 的 `checkpoint` 指向绝对 `state.pt` 路径，`training_run.directory` 记录实际 `train/` 路径。进度直接打印到终端。

通过 Streamlit + ECharts 查看单次 run 的本地评估结果。`--dashboard-run-dir` 必填；启动后访问 `http://127.0.0.1:8501`：

```bash
bash src/latent_working_memory/v4_1/scripts/dashboard.sh \
  --dashboard-run-dir artifacts/v4-1/am-recent_m256-d512_p1024-r256_inner3_01_20261029 \
  --dashboard-baselines \
    full-context artifacts/baselines/full-context/qwen3-4b-inst-2507_20261019/eval
```

- 在页面中选择 eval-id 和 checkpoint，在不同评估任务页签间切换；仅读取和绘制当前任务，点击“刷新结果”重新读取本地结果。
- 单 checkpoint 展示指标汇总，多 checkpoint 按真实累计 step 升序绘图；指标表与逐样本记录可在页面查看，基线排在当前方法之前。
- RULER 用折线图比较各长度的 Mean，选择长度后用分组柱图展示 13 项任务，只显示实际完成的长度，同一方法保持同色。任务横轴使用 `single_1/2/3`、`multik_1/2/3`、`multiv`、`multiq` 等缩写，`vt`、`cwe`、`fwe`、`qa_1/2` 保持原名；悬停显示完整任务名。
- RULER 多 checkpoint 可选择 Overall 或具体长度查看 step 曲线，再选择一个 checkpoint 查看长度总览和任务明细。共同长度可逐项比较；基线 Overall 按对应 checkpoint 的长度范围重算，缺任一长度时不显示该点的基线 Overall，缺失不补零。
- 要展示 baseline 结果时，追加 `--dashboard-baselines <标签> <结果目录> ...`，可传入多组，标签不可重复。支持 `eval/<benchmark>/summary.json` 布局的 `eval/` 目录，或直接传入含 `summary.json` 的单任务／单 checkpoint 结果目录。页面可勾选基线，不保存绑定。
- 缺少当前 benchmark 汇总时跳过；路径、格式或比较设置不一致时在页面提示，不影响已有结果。
- **质量比较**：核对基座模型标识、实际输入与参考答案、输入／解码／生成预算及评分口径。基座按模型路径末级名称识别，父目录可以不同；算法、窗口、精度、attention 实现、设备和批量不作为质量比较限制。
- **数据核对**：使用顺序无关的比较指纹，RULER 按长度分别核对。旧结果可沿用原数据指纹；缺少分组指纹且覆盖范围不同时，提示无法核实对应范围。
- **资源统计**：与质量指标分区展示，保留各结果的精度、attention 实现、设备、累计生成时间、峰值显存及调度墙钟时间等记录，不自动计算加速比。成本比较需另外统一硬件、批量和测量范围；重试的调度墙钟时间仅覆盖本次执行。
- **缓存与恢复**：跨方法展示允许上述差异；结果复用、失败重试和训练恢复仍严格核对各自原有的运行身份与设置。
- 地址和端口保存在 [`.streamlit/config.toml`](../../../../.streamlit/config.toml)，修改后须重启看板；临时更换端口时，在启动命令前加 `STREAMLIT_SERVER_PORT=8502`。
- 异地查看只需同步并保留 `<run-id>/eval/` 目录结构，仍用 `--dashboard-run-dir` 指向 run 根目录，无需 `plan/`、训练记录、权重或原始数据。

## 5. 调整参数

本节汇总完整运行、恢复、独立评估、离线准备与看板的公开参数。默认值以[正式配置](../../../../configs/v4-1/online_ttt_memory.json)为准：新训练与离线准备的显式 CLI 值在校验前覆盖 JSON，省略时沿用配置。模型结构、记忆更新及优化器设置直接修改 JSON，见[方法说明](../README.md)。

### 5.1 训练输入与执行预算

| 参数 | 用途／默认值 |
|---|---|
| **运行方式** |
| `--train-only` | 只进行训练及 dev 验证，跳过 test 评估 |
| `--resume` | 从 checkpoint 恢复并沿用原配置继续训练及评估，须指定 `--checkpoint` |
| `--evaluate-only` | 仅评估已有 checkpoint，须指定 `--checkpoint` 或 `--eval-run` |
| **配置与模型** |
| `--config` | JSON 配置文件，默认 `configs/v4-1/online_ttt_memory.json` |
| `--model-path` | 基座与 tokenizer 路径，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--seed` | 随机种子，默认 42，控制初始化、选样与打乱 |
| **目录与运行标识** |
| `--run-id` | 训练运行编号，如 `01_20261029`；默认 `YYYYMMDD-HHMMSS` |
| `--output-root` | 产物父目录，默认 `artifacts/v4-1` |
| **Teacher 数据准备** |
| `--prepare-data` | 离线准备的数据目录，默认 `data/delta-mem-qasper` |
| `--prepare-batch-size-per-gpu` | 每卡离线生成与统计批量，默认 64 |
| **数据集与筛选** |
| `--train-data` | 训练与 dev 验证的数据目录，默认 `data/delta-mem-qasper` |
| `--max-train-samples` | 训练样本数上限，默认 `null`（全部） |
| `--max-train-data-tokens` | 训练正文上限，默认 12288 |
| **训练时长** |
| `--epochs` | 训练轮数，默认 4 |
| **批处理** |
| `--micro-batch-size-per-gpu` | 每卡一次并行处理的样本批量，默认 2 |
| `--gradient-accumulation-steps` | 每次优化器更新前累积的 microbatch 数，默认 8 |
| `--teacher-batch-size-per-gpu` | 每卡运行时 teacher forcing 的提示／响应分支批量，默认 64 |

- `--train-only`、`--resume`、`--evaluate-only` 互斥。
- 长度筛选仅用于 train 正文，不含系统提示、chat template 等。超限样本整条排除，`null` 跳过该长度筛选。除此之外，默认要求正文至少包含 `pending_size + recent_size` 个 tokens，实际编码／解码仍须满足模型窗口限制。
- 全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数，默认单卡 16、双卡 32。

### 5.2 验证、评估与结果记录

| 参数 | 用途／默认值 |
|---|---|
| **Checkpoint** |
| `--save-every` | checkpoint 保存的正整数 optimizer step 间隔或 `after_each_epoch`（默认） |
| `--save-total-limit` | 保留最近成功保存的 checkpoint 数，默认 4 |
| `--checkpoint` | 用于恢复或独立评估；其中独立评估支持同一 run 的多个目录 |
| `--eval-run` | 指定运行目录下的 `train/`，选择其中仍保留 `state.pt` 的全部 checkpoint |
| **Dev 验证** |
| `--max-dev-samples` | dev 验证样本数上限，默认 `null`（全部） |
| `--dev-every` | dev 验证的 optimizer step 间隔，默认 50 |
| **Test 与 benchmark 评估** |
| `--eval-id` | 独立评估标识，默认 `YYYYMMDD-HHMMSS` |
| `--eval-type` | 评估类型，支持 `test`（默认）和 `benchmark` |
| `--test-data` | 用于完整运行或独立评估时覆盖 test 数据目录，默认 `null` |
| `--max-test-samples` | test 评估样本数上限，默认 `null`（全部） |
| `--max-new-tokens` | test 评估时 token 生成上限，默认 128 |
| `--benchmark` | benchmark 名称，默认 `null` |
| `--benchmark-data` | benchmark 数据目录 |
| `--lengths` | 仅用于 RULER benchmark 评估，默认 `8192 16384 32768` |
| **SwanLab** |
| `--tracking` | SwanLab 记录方式，支持 `online`（默认）或 `disabled`（仅保存本地产物） |
| `--swanlab-project` | SwanLab 项目，默认 `latent-working-memory-v4.1` |
| `--group` | SwanLab 实验分组，默认当前运行目录名 |
| `--tag` | SwanLab 命名空间标签 |
| **看板** |
| `--dashboard-run-dir` | 必填，指向运行根目录；仅需保留 `eval/` 目录结构 |
| `--dashboard-baselines` | 可选，按“标签 结果目录”成对传入；支持 benchmark 的 `eval/` 或直接含 `summary.json` 的目录 |

- 除了 `--save-every` 和 `--dev-every` 所指定的 step，epoch 末会自动保存 checkpoint 及进行 dev 验证。

## 6. 配置变更与缓存复用

| 变更 | 操作 |
|---|---|
| Teacher／tokenizer、输入模板、提示库、窗口边界、响应预算、候选池、尺度下限、seed | 使用新缓存目录重新准备，再启动新训练 |
| 扩大数据选样 | 为新增文档补齐同身份缓存；新训练记录实际数据身份 |
| Slots／维度、解码器、LoRA、inner 步数／初始步长、outer 优化设置 | 新训练目录；teacher 身份未变时可复用缓存 |
| 每次 outer AM 采样位置数 | 新训练目录；不超过原池容量时可复用完整池 |
| 离线 GPU 数量、`teacher.prepare_batch_size`／`teacher.batch_size` | 可复用缓存；恢复训练仍遵循完整配置一致的要求 |
| 评估重试、补评 checkpoint 或更换评估 GPU 数量 | 沿用原任务计划，复用完整结果及逐题进度，重新分片未完成样本 |
| 仅重新展示既有评估 | 启动或刷新独立看板，不执行模型 |

离线缓存以 token IDs、索引与固定尺度为主。按 36 层、32 query heads、2571 个边界估算，每条响应均达到 24 tokens 时，主要张量总量约为 439 MB。同格式、同配置的已有结果直接复用；不同 attention 内核、GPU 或批量的数值舍入可能影响重新生成的响应与尺度。训练时两份模型权重常驻；teacher 原文 KV、回放临时张量与 student 元训练图分阶段占用显存，实际成本和效果以运行结果为准。

## 7. 产物目录与记录

产物默认保存在 `artifacts/v4-1/`，可通过 `--output-root` 指定父目录。Teacher 缓存共享，各 run 的计划、训练和评估分别归档：

```text
<output_root>/
├── teacher/p{pending_size}-r{recent_size}/
│   ├── identity.json                         Teacher 缓存身份与提示库
│   └── documents/<document_id>/
│       └── write_<j>.safetensors             每轮参考响应、位置与尺度
└── <方法设置>_<run_id>/
    ├── plan/
    │   ├── config.json                       公共模型、teacher 设置与 seed
    │   ├── train.json                        训练设置与执行安排
    │   └── eval-<eval-id>/<task>.json         各任务的评估计划
    ├── train/
    │   ├── config.json                       实际训练配置
    │   ├── data.json                         数据选样与绑定身份
    │   ├── metrics.jsonl                     逐 step 的训练与验证指标
    │   ├── summary.json                      完成步数与最终 checkpoint
    │   ├── swanlab.json                      SwanLab run 身份
    │   ├── train.log                         训练命令与日志
    │   ├── status.json                       训练执行状态
    │   └── checkpoints/global_step_<N>/
    │       └── state.pt                      慢参数与完整恢复状态
    └── eval/<eval-id>/<task>/
        ├── status.json                       本轮评估进度与各 checkpoint 状态
        └── global_step_<N>/
            ├── eval.json                     评估配置快照
            ├── eval.log                      各 GPU／分片的评估命令与日志
            ├── work.json                     本轮待评估样本分工
            ├── progress/worker_<rank>.jsonl    已完成样本与资源记录
            ├── summary.json                  指标、数据、比较指纹与资源统计
            ├── details.jsonl                 逐题回答与评分
            └── lengths/<长度>/               RULER 各长度的评估结果
```

- **目录命名**：`<方法设置>` 为 `am-recent_m{num_slots}-d{memory_dim}_p{pending_size}-r{recent_size}_inner{inner_steps}`；`run_id` 默认 `YYYYMMDD-HHMMSS`，可显式指定。新训练拒绝覆盖已有计划或产物。
- **计划与配置**：`plan/` 保存执行前确定的设置；`train/config.json` 保存实际训练配置，评估结果目录的 `eval.json` 保存评估设置快照。自动 test 默认使用 `plan/eval-auto/test.json`；独立评估沿用对应任务计划，RULER 允许扩展长度。
- **训练与恢复**：Rank 0 统一写入产物，train/dev 指标汇总全部 GPU 的实际样本。`state.pt` 保存慢参数、优化器、scheduler、数据游标及各 rank 随机状态；恢复时与原计划核对。完整格式见[方法说明第 4 节](../README.md#4-代码与结果)。
- **评估结果**：各卡结果按原始样本顺序汇总；Test NLL 按答案 token 加权，EM／F1 按样本平均，benchmark 沿用官方汇总。完整结果直接复用；未完成时保留逐题进度，重试可更换 GPU 数，只重新分片剩余样本。RULER 继续复用已完成长度。
- **比较指纹**：Benchmark summary 的 `comparison_data` 按原始输入、参考答案、生成预算与任务生成，忽略执行顺序，RULER 按长度分组；仅供跨方法展示，不参与 `eval.json`、缓存或恢复校验。
- **日志与状态**：日志同时写入文件和终端，评估输出带 GPU／分片前缀；`status.json` 记录 checkpoint 与各分片的设备、命令、起止时间、返回码和错误。整体进度仅统计本轮选择。
- **展示方式**：SwanLab 默认开启，记录 train/dev、训练资源及配置；`--tracking disabled` 仅保留本地产物，恢复训练沿用 `swanlab.json` 中的 run。评估由独立 dashboard 展示，不读取 SwanLab 身份或凭据。
