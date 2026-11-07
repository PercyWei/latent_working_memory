# 20261102_v3.1.1 运行指南

创建时间：20261102 17:04:01 Asia/Shanghai

最后修订时间：20261107 11:39:43 Asia/Shanghai

计算协议与数据格式见[方法说明](../README.md)，命令行参数集中见[第 5 节](#5-调整参数)。以下命令使用项目根目录 `.venv`；正式实验先检查[K256 预训练配置](../../../../configs/v3_1_1/pretrain_k256.json)和[K256 后训练配置](../../../../configs/v3_1_1/post_k256.json)中的基座与数据路径。

## 1. 环境与入口

```bash
uv sync --frozen
bash src/latent_working_memory/v3_1_1/scripts/run.sh --help
```

`run.sh` 默认执行训练和 dev 验证，`--evaluate-only` 执行独立评估，`--resume` 用于恢复训练或搭配 `--evaluate-only` 续跑已有评估；离线准备使用 `prepare`，评估看板通过 `dashboard.sh` 启动。各入口支持 `--help`。

离线准备先合并原窗口的 `text + continuation`，按字符数 ÷ 4 估算 tokens，裁剪完整样本并做长度分层；训练通过 `torchrun` 为每张可见卡启动一个进程；评估在可见 GPU 间分片同一 checkpoint 的待评估样本，每卡批量处理，多个 checkpoint 按 step 依次执行。

本文示例约定：

| 内容 | 路径 |
|---|---|
| 基座与 tokenizer | `~/models/Qwen3-4B-Instruct-2507` |
| 原始窗口 | `data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/{train,dev}.jsonl` |
| AE／LM 准备数据 | `data/fineweb_20261104/{train,dev}.jsonl` |
| QA 数据 | `data/delta-mem-qasper/{train,dev,test}.jsonl` |
| 产物父目录 | `artifacts/v3-1-1` |

运行目录与各阶段产物见 [第 7 节](#7-产物目录与记录)。

## 2. 离线数据准备

```bash
bash src/latent_working_memory/v3_1_1/scripts/run.sh prepare \
  --prepare-data data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/train.jsonl \
  --output-file data/fineweb_20261104/train.jsonl \
  --target-samples null \
  --min-tokens null \
  --max-tokens 4096 \
  --length-bins 10 \
  --length-distribution uniform
```

将输入、输出文件名都改为 `dev.jsonl`，再运行一次以准备 dev 数据。

- **来源与范围**：`--prepare-data` 必须指向标准命名的 `train.jsonl` 或 `dev.jsonl`。先合并原窗口的 `text + continuation`，以完整原文的 `floor(len / 4)` 统计最短、最长估算长度与容量。`--min-tokens`、`--max-tokens` 均必填，接受正整数或 `null`；实际长度范围为请求范围与该数据集合并原文长度范围的交集。先打印支持范围与可用数量，再直接构造。
- **长度分布**：`--length-bins` 默认 10，实际整数长度不足时减少区间数。`--length-distribution` 支持 `uniform`（默认）和 `random`，两种方式均在区间内随机选择整数长度，长度配额受来源容量约束。
- **目标数量**：`--target-samples` 指定目标生成数据数量。正整数指定生成条数，超过可用数量时报错；`null`（默认）不设构造数量上限。
- **裁剪**：从合并后的完整原文随机裁出一条连续 `text`，长度范围约束整条输出样本。
- **随机设置与输出**：内部固定 seed 为 42。数据写入必填的 `--output-file` 指定路径，保留来源 split；元信息保存为同目录的 `preparation-{split}.json`。输出文件已存在时直接报错。

## 3. 训练

新训练必须同时指定阶段 `--stage` 和配置文件 `--config`，配置中的 `stage` 必须与 `--stage` 一致。两阶段共用以下运行规则：

- **配置覆盖**：显式 CLI 参数覆盖 JSON；模型结构和记忆更新设置直接修改 JSON，参数说明见[第 5 节](#5-调整参数)。
- **训练批量**：`--micro-batch-size-per-gpu` 控制每卡 microbatch 的样本数；全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数。预训练按窗口、后训练按文档计数；两阶段示例均为 `2 × 2 × 4 = 16`。
- **执行流程**：先保存训练计划，再执行训练、dev 与 checkpoint 保存。
- **覆盖训练**：追加 `--overwrite` 会清理本次 `run-id` 对应的整个 run，包括计划、训练和评估结果，再保存新计划重新训练；云端历史保留。

### 3.1 单次压缩预训练

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --stage pretrain \
  --config configs/v3_1_1/pretrain_k256.json \
  --train-data data/fineweb_20261104 \
  --run-id pretrain_k256_20261104 \
  --epochs 1 \
  --micro-batch-size-per-gpu 2 \
  --gradient-accumulation-steps 4
```

- **运行目录**：`run_id` 默认为 `pretrain_k{memory_slots}_{YYYYMMDD-HHMM}`，其中 `memory_slots` 取自 `model.memory_slots`。
- **输入与目标**：读取准备后的完整 `text`，先按完整样本的实际 token 数执行长度筛选，再依据 seed 与 `trajectory_id` 固定选择 AE 或 LM，跨 epoch 不变。AE 压缩并重建完整样本；LM 以前缀为压缩输入，预测样本末尾的 512 tokens（`objective.lm_target_tokens` 默认值），若剩余前缀长度不大于 K 则改为 AE。默认 LM 概率 0.5，目标均追加 EOS。
- **可训练参数**：压缩器 LoRA 和 memory embeddings；基座与读取器冻结，读取器保留损失到 memory 的梯度。预训练不更新边界模块。

### 3.2 后训练（QA／GRPO 联合训练）

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --stage post \
  --config configs/v3_1_1/post_k256.json \
  --run-id post_k256_20261104 \
  --init-checkpoint 'artifacts/v3-1-1/pretrain_k256_20261104/train/checkpoints/global_step_<N>' \
  --epochs 1 \
  --micro-batch-size-per-gpu 2 \
  --gradient-accumulation-steps 4
```

- **运行目录**：`run_id` 默认为 `post_k{memory_slots}_{YYYYMMDD-HHMM}`，其中 `memory_slots` 取自 `model.memory_slots`。
- **初始化**：后训练需通过 `--init-checkpoint` 指定预训练来源。源 checkpoint 阶段须为 `pretrain`，两阶段 `model` 配置一致。新 run 只继承压缩器，重新创建边界模块、optimizer、scheduler、游标和边界随机流。实际来源随训练记录保存。
- **训练单位**：每行一篇文档及其全部 QA。一篇文档采样 G 条完整硬分段，每条路径共享 memory 评价同一套 QA；G 不计入全局 batch。
- **一次联合更新**：固定参数采样评价，按记录动作带梯度回放，累计整批后更新一次；下一批刷新轨迹。QA 损失训练记忆内容，停止梯度后的组内优势训练边界选择。
- **归约与尾批**：损失按实际文档数平均，尾批不补齐、不丢弃、不跨 epoch 拼接；空 rank 同样参加 collective。

### 3.3 中断恢复

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --resume \
  --checkpoint 'artifacts/v3-1-1/pretrain_k256_20261104/train/checkpoints/global_step_<N>'
```

- **恢复起点**：仅接受当前 run 最新的完整 checkpoint。
- **恢复状态**：可训练参数、optimizer、scheduler、全局数据游标及各 rank 的随机状态。
- **参数来源**：仅用 `--checkpoint` 定位原训练计划，恢复全部实验设置；其他已知 CLI 实验设置忽略，启动时集中提示被忽略的选项。
- **一致性要求**：原计划与 checkpoint 保存的配置须一致；GPU 数量和数据须保持一致。修改训练设置应启动新 run，恢复后训练不依赖预训练来源文件。
- **执行计划**：沿用原训练计划；预算已完成时直接返回，不读取或启动评估计划。

## 4. 评估

Test、LoCoMo 和 RULER 共用独立评估入口，仅接受后训练 checkpoint，模型与方法设置从 checkpoint 读取。新建时确定数据、评估预算和 checkpoint 范围；中断后从已有计划继续。

### 4.1 启动评估

以下以 QASPER test 为例，评估原运行的 step 25、50：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --evaluate-only \
  --eval-run 'artifacts/v3-1-1/post_k256_20261104' \
  --eval-id standard_20261104 \
  --eval-task test \
  --eval-data 'data/delta-mem-qasper' \
  --checkpoint global_step_25 global_step_50 \
  --eval-batch-size-per-gpu 8
```

切换任务时，替换 `--eval-task` 与 `--eval-data`，并按需设置该任务的专属参数：

| `--eval-task` | `--eval-data` | 专属参数 |
|---|---|---|
| `test`（默认） | `data/delta-mem-qasper` | `--max-test-samples` 限制文档数，`--max-new-tokens` 设置每题生成上限 |
| `locomo` | `data/benchmarks/locomo` | |
| `ruler` | `data/benchmarks/ruler-v1/Qwen3-4B-Instruct-2507` | `--lengths` 选择长度 |

- **Benchmark 准备**：Benchmark 数据须提前准备，数据布局与入口见[公共 benchmark 指南](../../benchmarks/README.md#2-数据准备)。
- **Test 协议**：固定读取数据目录的完整 `test.jsonl`，不套用训练长度筛选，默认每题最多生成 128 tokens；同文档全部 QA 共用记忆。
- **运行与范围**：`--eval-run` 必填，指向运行根目录。`--checkpoint` 可选，只填一个或多个 `global_step_<N>` 名称，自动查找 `<eval-run>/train/checkpoints/<名称>/state.pt`；省略时选择当时保存的全部完整 checkpoint。
- **评估计划**：新建时须提供 `--eval-data`，默认先将选定范围与评估设置保存到 `<eval-run>/plan/eval-<eval-id>/<task>.json`，再执行评估，结果保存到 `<eval-run>/eval/<eval-id>/<task>/`。`--eval-id` 默认 `YYYYMMDD-HHMMSS`；同一标识下各任务独立管理，目标任务计划已存在时直接报错。
- **并行与批量**：多张可见 GPU 共同处理一个 checkpoint 的样本，每卡一份模型；`--eval-batch-size-per-gpu` 控制每卡批量，默认 8。多个 checkpoint 按 step 升序依次评估。 Test 按文档组批，benchmark 按题组批，RULER 按 length 分组。
- **覆盖评估**：追加 `--overwrite`，按本次参数重建当前评估任务（`<eval-run>` `<eval-id>` `<eval-task>`）的计划和全部 checkpoint 结果，其他任务、其他 eval-id、其他运行的评估结果保留。目标正在使用时直接报错。

### 4.2 中断恢复或补充

使用已有计划继续评估，也可选择本次范围、补评新 checkpoint。以下假设原 test 计划包含 step 25、50，训练又保存了 step 75，本次只评估 step 50、75：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --evaluate-only \
  --resume \
  --eval-run 'artifacts/v3-1-1/post_k256_20261104' \
  --eval-id standard_20261104 \
  --eval-task test \
  --checkpoint global_step_50 global_step_75 \
  --eval-batch-size-per-gpu 4
```

- **恢复定位**：须显式提供 `--eval-run`、`--eval-id` 和 `--eval-task`，对应任务计划必须存在。若原评估通过 `--output-root` 重定向，仍须传入原值以定位计划。
- **本次范围**：省略 `--checkpoint` 时评估计划中的全部 checkpoint；传入名称列表时只调度指定项。完整且身份一致的结果直接复用，未完成结果保留逐文档／题目进度，只补缺失样本；产物身份不一致时报错。
- **计划更新**：新 checkpoint 先校验归属与配置，再并入计划的 `checkpoints` 列表，按 step 排序去重，不缩减旧范围；仅有新增时原子更新计划。上述命令使累计范围变为 25、50、75，任务状态的 `selected_checkpoints` 仅记录本次的 50、75。
- **参数来源**：数据、预算和 RULER 长度沿用原计划；对应 CLI 选项忽略，启动时集中提示。
- **执行资源**：GPU 数量与 `--eval-batch-size-per-gpu` 可调整；未完成样本按本次资源重新分片、组批，已提交记录保留。

### 4.3 评估结果与看板

- **产物路径**：下表路径均相对于运行根目录。

| 评估产物 | 内容 |
|---|---|
| `plan/eval-<eval-id>/<task>.json` | 该任务累计的 checkpoint 列表、固定数据与评估设置 |
| `eval/<eval-id>/<task>/global_step_<N>/eval.json` | 当前 checkpoint 的评估设置快照 |
| `eval/<eval-id>/<task>/global_step_<N>/summary.json` | 实验身份、协议与指标汇总 |
| `eval/<eval-id>/<task>/global_step_<N>/details.jsonl` | 按原始样本顺序汇总的回答与评分 |
| `eval/<eval-id>/<task>/global_step_<N>/work.json` | 本次执行的待评估样本分工 |
| `eval/<eval-id>/<task>/global_step_<N>/progress/worker_<rank>.jsonl` | 各分片逐文档／题目保存的完成进度与资源记录 |

- **结果管理**：

  - **结果提交**：各分片在一篇 Test 文档的全部 QA 或一道 benchmark 题目完成后保存进度，样本齐全并通过校验后才提交最终 details 与 summary。RULER 另在 `lengths/<长度>/` 保存计划中各长度的完成结果，续跑复用已有记录。
  - **运行关联**：Summary 的 `checkpoint` 指向绝对 `state.pt` 路径，`training_run.directory` 记录实际 `train/` 路径；执行进度直接打印到终端。

- **启动看板**：通过 Streamlit + ECharts 查看本地结果，`--dashboard-run-dir` 指向运行根目录；启动后访问 `http://127.0.0.1:8501`。

```bash
bash src/latent_working_memory/v3_1_1/scripts/dashboard.sh \
  --dashboard-run-dir artifacts/v3-1-1/post_k256_20261104 \
  --dashboard-eval-id standard_20261104 \
  --dashboard-baselines \
    full-context artifacts/baselines/full-context/qwen3-4b-inst-2507_20261019/eval
```

- **结果浏览**：

  - **页面入口**：看板包含“训练与验证”和“独立评估”页签，评估结果位于“独立评估”。
  - **评估选择**：`--dashboard-eval-id` 指定启动时选中的评估，省略时默认列表首项；页面仍可切换其他 eval-id。指定评估不存在或没有已完成结果时提示错误。
  - **任务切换**：选择 checkpoint 和任务页签，仅读取当前任务；点击“刷新结果”重新加载本地结果。
  - **指标展示**：单 checkpoint 展示指标汇总，多 checkpoint 按实际 optimizer step 升序绘图；可查看指标表与逐样本记录，基线排在当前方法之前。

- **RULER 展示**：

  - **长度与任务**：用折线图比较各长度的 Mean，选择长度后用分组柱图展示 13 项任务；只显示已完成的长度，同一方法保持同色。
  - **Checkpoint 趋势**：多 checkpoint 可选择 Overall 或具体长度查看 step 曲线，再查看单个 checkpoint 的长度总览与任务明细。基线 Overall 按对应 checkpoint 的长度范围重算，缺任一长度时不显示该点，不补零。
  - **逐题响应**：开启“查看逐样本记录”，选择 checkpoint 后按长度和子任务筛选，再分页查看参考答案、模型响应与评分。
  - **完整题面**：展开“完整输入题面”，从原始 `test.jsonl` 读取全文；已有评估结果可直接使用，无需重新评估。

- **基线接入**：

  - **添加方式**：通过 `--dashboard-baselines <标签> <结果目录> ...` 添加多组基线，标签须唯一。支持 `eval/<benchmark>/summary.json` 布局的 `eval/` 目录，或含 `summary.json` 的单任务／单 checkpoint 目录；页面可勾选基线，不保存绑定。
  - **异常提示**：缺少当前 benchmark 汇总时跳过该基线；路径、格式或比较设置不一致时在页面提示，不影响已有结果。

- **比较规则**：

  - **质量比较**：核对基座模型标识、实际输入与参考答案、输入／解码／生成预算及评分口径。基座按模型路径末级名称识别，父目录可以不同；算法、窗口、精度、attention 实现、设备和批量不作为质量比较限制。
  - **数据核对**：使用顺序无关的比较指纹，RULER 按长度分别核对。旧结果可沿用原数据指纹；缺少分组指纹且覆盖范围不同时，提示无法核实对应范围。
  - **资源与成本**：资源统计与质量指标分区展示。批量计算时间按参与样本分摊后累计，不等同于每条请求的独立等待时延；峰值显存按 worker／批次范围观测。比较成本须统一硬件、实际评估批量及测量范围，不自动计算加速比；重试墙钟时间仅覆盖本次执行。
  - **缓存校验**：跨方法展示允许实现与运行环境不同；结果复用、失败重试和训练恢复仍严格核对原有身份与设置。

- **访问方式**：

  - **访问配置**：地址和端口见 [`.streamlit/config.toml`](../../../../.streamlit/config.toml)，修改后须重启看板；临时更换端口可在启动命令前加 `STREAMLIT_SERVER_PORT=8502`。
  - **异地查看**：同步 `<运行目录>/eval/` 并保留目录结构，将 `--dashboard-run-dir` 指向运行根目录。查看 RULER 完整题面还需对应原始数据，在页面“RULER 数据目录”填写本地路径，目录下应包含 `<长度>/<子任务>/test.jsonl`。查看训练曲线还需同步 `train/metrics.jsonl`。

## 5. 调整参数

本节汇总训练、恢复、独立评估、离线准备与看板的公开参数。训练默认值以[预训练配置](../../../../configs/v3_1_1/pretrain_default.json)和[后训练配置](../../../../configs/v3_1_1/post_default.json)为准：新训练的显式 CLI 值在校验前覆盖 JSON，省略时沿用配置。模型结构、方法目标及优化器设置直接修改 JSON，见[方法说明](../README.md)。本文示例改用 K256 配置，预训练完整样本上限为 4096；`*_default.json` 保留 K128 设置。

### 5.1 训练输入与执行预算

| 参数 | 用途／默认值 |
|---|---|
| **运行方式** | |
| `--evaluate-only` | 启动评估 |
| `--resume` | 恢复训练或补充评估 |
| `--overwrite` | 覆盖式训练或评估 |
| **配置与模型** | |
| `--stage` | 训练阶段，新训练必填，支持 `pretrain` 或 `post` |
| `--config` | JSON 配置文件，新训练必填 |
| `--model-path` | 基座与 tokenizer 路径，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--seed` | 训练随机种子，默认 42，控制初始化、选样与打乱 |
| `--init-checkpoint` | 后训练初始化所用的预训练 checkpoint，后训练新 run 必填 |
| **目录与运行标识** | |
| `--output-root` | 训练产物父目录，默认 `artifacts/v3-1-1` |
| `--run-id` | 训练运行目录名，默认见[目录命名](#7-产物目录与记录) |
| **离线数据准备** | |
| `--prepare-data` | 来源 `train.jsonl` 或 `dev.jsonl` 文件，必填；文件名确定 split |
| `--target-samples` | 目标生成条数；正整数超过可用数量时报错，`null`（默认）不设数量上限，使用满足长度条件的全部数据 |
| `--min-tokens` | 离线准备的完整样本估算 tokens 下限，必填；正整数或 `null`，`null` 使用合并原文最短长度 |
| `--max-tokens` | 离线准备的完整样本估算 tokens 上限，必填；正整数或 `null`，`null` 使用合并原文最长长度 |
| `--output-file` | 离线准备的完整输出文件路径，必填，文件名与来源一致；文件已存在时报错 |
| `--length-bins` | 离线准备的长度区间数，默认 10，整数长度不足时减少 |
| `--length-distribution` | 离线准备的区间分配方式，`uniform`（默认）尽量等配额，`random` 逐样本等概率选择区间 |
| **数据集与筛选** | |
| `--train-data` | 训练与 dev 验证的数据目录，默认使用配置的 `training.dataset_dir` |
| `--max-train-samples` | 训练样本数上限，默认 `null`（全部） |
| `--max-train-data-tokens` | 预训练完整样本／后训练正文的实际 token 数上限，默认 12288；K256 预训练示例为 4096 |
| **训练时长** | |
| `--epochs` | 训练轮数，默认 1 |
| **批处理** | |
| `--micro-batch-size-per-gpu` | 每卡 microbatch 的样本数，默认 2 |
| `--gradient-accumulation-steps` | 每次优化器更新前累积的 microbatch 数，默认 4 |

- **数据筛选**：预训练按完整样本 `text`、后训练按正文的实际 token 数筛选，不含系统提示、chat template 等。train 和 dev 均须严格大于 `model.memory_slots`；上限仅用于 train，超限样本整条排除，`null` 取消该上限。LM 划分后的压缩前缀若不大于 K 则改为 AE，实际编码／解码仍须满足模型窗口限制。
- **批量与预算**：全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数，默认按总批次 16 进行配置。

### 5.2 验证、评估与结果记录

| 参数 | 用途／默认值 |
|---|---|
| **Checkpoint** | |
| `--save-every` | checkpoint 保存的正整数 optimizer step 间隔或 `after_each_epoch`（默认） |
| `--save-total-limit` | 保留最近成功保存的 checkpoint 数，默认 2 |
| `--checkpoint` | 训练恢复用单个完整路径；评估只填 `global_step_<N>` 名称，支持多个；新建或覆盖时省略选当时全部，续跑时省略选累计列表全部，传入则仅调度指定项并累计新项 |
| **Dev 验证** | |
| `--max-dev-samples` | dev 验证样本数上限，默认 64；`null` 使用全部 |
| `--dev-every` | dev 验证的 optimizer step 间隔，默认 100 |
| **Test 与 benchmark 评估** | |
| `--eval-run` | 指向运行根目录，用于保存评估计划和 checkpoint 确认等 |
| `--eval-id` | 评估标识，默认 `YYYYMMDD-HHMMSS` |
| `--eval-task` | 评估任务，支持 `test`（默认）、`locomo` 和 `ruler` |
| `--eval-data` | 当前评估任务的数据目录 |
| `--eval-batch-size-per-gpu` | 每卡评估批量，默认 8 |
| `--max-test-samples` | test 评估文档数上限，默认 `null`（全部） |
| `--max-new-tokens` | test 评估时 token 生成上限，默认 128 |
| `--lengths` | RULER 评估时所用数据的长度，默认 `8192 16384 32768` |
| **SwanLab** | |
| `--tracking` | SwanLab 记录方式，支持 `online` 或 `disabled`（默认，仅保存本地产物） |
| `--swanlab-project` | SwanLab 项目，启用记录时显式指定 |
| `--group` | SwanLab 实验分组，默认 `run_id` |
| `--tag` | SwanLab 命名空间标签 |
| **看板** | |
| `--dashboard-run-dir` | 必填，指向运行根目录；仅需保留 `eval/` 目录结构 |
| `--dashboard-eval-id` | 可选，启动时选中的 eval-id；省略时默认列表首项，页面可切换 |
| `--dashboard-baselines` | 可选，按“标签 结果目录”成对传入；支持 benchmark 的 `eval/` 或直接含 `summary.json` 的目录 |

- **验证与保存**：除了 `--save-every` 和 `--dev-every` 所指定的 step，epoch 末和最终步也保存 checkpoint 及进行 dev 验证。

## 6. 配置变更与结果复用

| 变更 | 操作 |
|---|---|
| 预训练准备的来源文件、目标数量、完整样本估算长度范围、区间数或分配方式 | 修改准备参数，准备对应 split；训练仍按实际 token 数筛选，再启动新训练 |
| 从预训练进入后训练阶段 | 新 run，通过 `--init-checkpoint` 载入兼容压缩器 |
| 模型结构、计算、方法目标或优化器设置 | 修改基础 JSON，再启动新训练 |
| 数据、batch、轮数或保存等运行设置 | 新训练使用 CLI 覆盖或修改基础 JSON |
| 评估重试、调整评估批量或 GPU 数量 | 使用 `--evaluate-only --resume`，以原 `--eval-run`、`--eval-id` 与 `--eval-task` 定位计划，只重新分片并组批原计划的未完成样本 |
| 选择本次评估 checkpoint、增加 checkpoint | 续跑时传 `--checkpoint` 名称列表，仅调度指定项；新项并入累计计划，不缩减旧列表 |
| 调整 RULER 长度 | 另开 eval-id 新建计划，或用 `--overwrite` 重建当前任务 |
| 重新创建当前评估任务 | 显式 `--overwrite`，重新提供 `--eval-data` 与本次设置；当前 task 的旧计划及结果清空 |
| 仅重新展示既有评估 | 启动或刷新独立看板，不执行模型 |

## 7. 产物目录与记录

离线准备数据写入 `<output_file>`，元信息保存为同目录的 `preparation-{split}.json`，与训练产物分开：

```text
data/fineweb_20261104/
├── train.jsonl
├── preparation-train.json
├── dev.jsonl
└── preparation-dev.json
```

产物默认保存在 `artifacts/v3-1-1/`，可通过 `--output-root` 指定父目录。各 run 的计划、训练和评估分别归档：

```text
<output_root>/
└── <run_id>/
    ├── plan/
    │   ├── config.json                       stage、seed、model 与 objective
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
    │       └── state.pt                      可训练参数与完整恢复状态
    └── eval/<eval-id>/<task>/
        ├── status.json                       本轮评估进度与各 checkpoint 状态
        └── global_step_<N>/
            ├── eval.json                     评估配置快照
            ├── eval.log                      各 GPU／分片的评估命令与日志
            ├── work.json                     本轮待评估样本分工
            ├── progress/worker_<rank>.jsonl    已完成样本与资源记录
            ├── summary.json                  指标、数据、比较指纹与资源统计
            ├── details.jsonl                 逐文档／题目回答与评分
            └── lengths/<长度>/               RULER 各长度的评估结果
```

- **目录命名**：完整训练路径为 `<output_root>/<run_id>/train`。`run_id` 默认为 `pretrain_k{memory_slots}_{YYYYMMDD-HHMM}`（预训练）或 `post_k{memory_slots}_{YYYYMMDD-HHMM}`（后训练），时间戳采用 `Asia/Shanghai` 时区、精确到分钟。显式指定的 `run_id` 原样作为目录名。新训练默认拒绝覆盖已有计划或产物，显式 `--overwrite` 才清理并重建；恢复训练沿用原目录。
- **计划与配置**：`plan/` 保存执行前确定的设置，最终解析生成的 `run_id` 写入计划配置；`train/config.json` 保存实际训练配置，评估结果目录的 `eval.json` 保存评估设置快照。训练只创建训练计划；独立评估默认新建不存在的 task 计划，固定数据、预算与长度，`checkpoints` 保存累计范围。续跑可选择本次 checkpoint 子集并加入新项，仅有新增时原子更新计划；`--overwrite` 重建当前 task。
- **覆盖与互斥**：校验通过后，持删除范围之外的稳定锁执行清理与重建；锁覆盖 worker 生命周期，占用时拒绝覆盖。锁文件不属于实验结果。
- **训练与恢复**：Rank 0 统一写入产物，train/dev 指标汇总全部 GPU 的实际样本。`state.pt` 保存可训练的压缩器 LoRA、memory embeddings、边界参数、优化器、scheduler、数据游标及各 rank 随机状态；恢复时严格核对训练计划与绑定状态，不依赖评估计划。完整格式见[方法说明](../README.md#代码与产物)。
- **评估结果**：各卡结果按原始样本顺序汇总；Test 的 `qa_nll` 先按题、再按文档平均，`token_nll` 按答案 token 加权，EM／F1 按题平均，benchmark 沿用官方汇总。评估续跑复用完整结果；未完成时保留逐文档／题目进度，可调整 GPU 数和评估批量，只重新分片、组批剩余样本。RULER 继续复用已完成长度。
- **比较指纹**：Benchmark summary 的 `comparison_data` 按原始输入、参考答案、生成预算与任务生成，忽略执行顺序，RULER 按长度分组；仅供跨方法展示，不参与 `eval.json`、缓存或恢复校验。
- **日志与状态**：日志同时写入文件和终端，评估输出带 GPU／分片前缀；任务 `status.json` 的 `eval_run` 指向运行根目录，`selected_checkpoints` 是本次调度范围，与计划 `checkpoints` 的累计范围区分。状态记录 checkpoint 与各分片的设备、命令、起止时间、返回码和错误，整体进度仅统计本轮选择。
- **展示方式**：SwanLab 默认关闭；启用后记录 train/dev、训练资源及配置，project 须明确配置，group 未指定时默认 `run_id`。`--tracking disabled` 仅保留本地产物，恢复训练沿用 `swanlab.json` 中的 run。评估由独立 dashboard 展示，不读取 SwanLab 身份或凭据。
