# 20261102_v3.1.1 运行指南

创建时间：20261102 17:04:01 Asia/Shanghai

最后修订时间：20261112 15:55:39 Asia/Shanghai

计算协议与数据格式见[方法说明](../README.md)，命令行参数集中见[第 5 节](#5-调整参数)。以下命令使用项目根目录 `.venv`；正式实验先检查[K256 预训练配置](../../../../configs/v3_1_1/pretrain_k256.json)和[K256 后训练配置](../../../../configs/v3_1_1/post_k256.json)中的基座与数据路径。

## 1. 环境与入口

```bash
uv sync --frozen
bash src/latent_working_memory/v3_1_1/scripts/run.sh --help
```

- 数据准备：预训练数据准备使用 `prepare-pre`，后训练数据准备使用 `prepare-post`，长度分档使用 `prepare-length`。
- 训练与评估：`run.sh` 默认执行训练和 dev 验证，`--evaluate-only` 执行独立评估，`--resume` 用于恢复训练或搭配 `--evaluate-only` 续跑已有评估，评估看板通过 `dashboard.sh` 启动。

本文示例约定：

| 内容 | 路径 |
|---|---|
| 基座与 tokenizer | `/root/rivermind-data/models/Qwen3-4B-Instruct-2507` |
| AE / LM 数据来源 | `data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/{train,dev,test}.jsonl` |
| QA 数据来源 | `data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008/{train,dev,test}.jsonl` |
| 产物父目录 | `artifacts/v3-1` |

运行目录与各阶段产物见 [第 7 节](#7-产物目录与记录)。

## 2. 离线数据准备

`prepare-pre` 与 `prepare-post` 共用以下规则：

- `--target-samples` 默认 `null`，使用全部合条件数据；指定正整数时随机选取对应数量，超过可用数量时报错。
- `--seed` 默认 42，相同输入和参数可复现生成结果。
- `--output-file` 作为数据输出文件一般为 `<split>.jsonl`，构造参数与统计保存为同目录的元信息文件 `preparation-<split>.json`。

### 2.1 预训练数据

```bash
bash src/latent_working_memory/v3_1_1/scripts/run.sh prepare-pre \
  --prepare-data data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/train.jsonl \
  --output-file data/fineweb_train5k_20261108/train.jsonl \
  --target-samples 5000 \
  --min-tokens null \
  --max-tokens 4096 \
  --length-bins 10 \
  --length-distribution uniform \
  --seed 42
```

- **数据格式**：JSONL 每行仅包含 `{"text":"完整文本"}`。
- **正文与长度**：合并 Multisegment 的 `text + continuation`，按字符数 ÷ 4 估算长度。`--min-tokens`、`--max-tokens` 默认 `null`（采用对应来源边界），也可指定正整数；请求范围与来源长度范围取交集，再从完整原文随机裁出一条连续文本。
- **长度分布**：`--length-bins` 默认划分 10 个等宽区间，整数长度不足时减少区间数。`--length-distribution` 的 `uniform`（默认）尽量均衡分配，`random` 逐条等概率选择区间；均在区间内随机选长度，实际分布受来源容量约束。

### 2.2 后训练数据

```bash
bash src/latent_working_memory/v3_1_1/scripts/run.sh prepare-post \
  --prepare-data data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008/train.jsonl \
  --output-file data/fineweb-factqa_q10_train1k_20261108/train.jsonl \
  --target-samples null \
  --min-tokens null \
  --max-tokens null \
  --qas-per-sample 10 \
  --seed 42
```

- **数据格式**：JSONL 每行仅包含 `{"text":"完整正文","qas":[{"question":"问题","answer":"答案","segment_id":"seg0","segment_char_span":[0,4]}]}`。`segment_char_span` 为来源段在正文中的字符区间，左闭右开。
- **正文筛选**：完整保留 FactQA 正文；`--min-tokens`、`--max-tokens` 默认 `null`，也可指定正整数，按字符数 ÷ 4 过滤整条文本，`null` 取消对应界限。
- **QA 选择**：仅使用 `task` 题，`--qas-per-sample` 默认最多 10 题，题数不足时全保留，`null` 保留全部。按保留题数等分全文，依次选择答案证据最靠近各份中心的题，每题最多一次，最后按证据在正文中的顺序输出。

### 2.3 长度分档

两阶段均可对已准备数据集中的指定 split 执行长度分档。以下以后训练的 train split 为例：

```bash
bash src/latent_working_memory/v3_1_1/scripts/run.sh prepare-length \
  --prepare-data data/fineweb-factqa_q10_train1k_20261108/ \
  --output-dir data/fineweb-factqa_q10_train1k_length4_20261112 \
  --model-path /root/rivermind-data/models/Qwen3-4B-Instruct-2507 \
  --split train \
  --length-bins 4
```

- **数据格式**：读取来源目录中的 `<split>.jsonl`，根据数据正文长度进行分别保存，保留原数据格式；其余 split 的 JSONL 文件原样复制到输出目录。
- **长度分档**：使用与训练相同的 tokenizer，按完整正文的实际 token 数划分等宽区间，不含系统提示和 chat 前缀。`--length-bins` 默认 4。
- **输出与统计**：输出目录须不存在。分档写入 `<split>/bin_XX.jsonl`，终端与 `preparation-<split>.json` 展示各档边界、样本数及实际长度的最小值、最大值和均值，方便在训练前查看并调整数据。

```text
fineweb-factqa_q10_train1k_length4_20261112/
├── train/
│   ├── bin_00.jsonl
│   ├── bin_01.jsonl
│   ├── bin_02.jsonl
│   └── bin_03.jsonl
├── dev.jsonl
└── preparation-train.json
```

## 3. 训练

新训练必须同时指定阶段 `--stage` 和配置文件 `--config`，配置中的 `stage` 必须与 `--stage` 一致。两阶段共用以下运行规则：

- **配置覆盖**：显式 CLI 参数覆盖 JSON；模型结构和记忆更新设置直接修改 JSON，参数说明见[第 5 节](#5-调整参数)。
- **计算设置**：统一使用 SDPA，默认 FP32。`--mixed-precision` 和 `--save-on-cpu` 分别开启标准 BF16 autocast（参数精度保持 FP32）和反向保存张量的 CPU 暂存，两者默认关闭。恢复与独立评估沿用已保存设置，详细计算方式见[方法说明](../README.md#配置与运行)。
- **K256 配置**：两阶段使用上述同一基座及各自数据目录；预训练／后训练的 `warmup_steps` 分别为 20／5，均每 20 个 optimizer steps 验证和保存。预训练不设实际 token 数上限，后训练上限为 12288；准备阶段的 4096 是字符估算长度上限。
- **训练批量**：`--micro-batch-size-per-gpu` 控制每卡 microbatch 的样本数；全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数。预训练按窗口、后训练按文档计数；两阶段示例均为 `2 × 2 × 4 = 16`。
- **长度课程**：完成[长度分档](#23-长度分档)后，在原训练命令中追加 `--length-curriculum --curriculum-stage-epochs 1 --curriculum-replay-ratio 0.2`。按档位从短到长推进，每轮完整遍历本档，并从此前各档尽量均衡地随机无放回补充本档数量的 20%，数量四舍五入，历史不足时取全部；每轮重新抽取，混合打乱后统一切 batch，全部档位完成后结束。关闭课程时，各档合并后按 `--epochs` 全量随机训练；dev 和 test 保持原协议。
- **执行流程**：先保存训练计划，再执行训练、dev 与 checkpoint 保存。
- **覆盖训练**：追加 `--overwrite` 会清理本次 `run-id` 对应的整个 run，包括计划、训练和评估结果，再保存新计划重新训练；云端历史保留。

### 3.1 单次压缩预训练

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --stage pretrain \
  --config configs/v3_1_1/pretrain_k256.json \
  --run-id pretrain_k256_20261104 \
  --epochs 1 \
  --micro-batch-size-per-gpu 2 \
  --gradient-accumulation-steps 4
```

- **运行目录**：`run_id` 默认为 `pretrain_k{memory_slots}_{YYYYMMDD-HHMM}`，其中 `memory_slots` 为记忆槽位数。
- **输入与目标**：读取准备后的完整 `text`，先按完整样本的实际 token 数执行长度筛选，再依据 seed 与文本内容标识固定选择 AE 或 LM，跨 epoch 不变。AE 压缩并重建完整样本；LM 以前缀为压缩输入，默认预测样本末尾的 512 tokens，若剩余前缀长度不大于 K 则改为 AE。默认 LM 概率 0.5，目标均追加 EOS。
- **可训练参数**：压缩器 LoRA 和 memory embeddings；基座与读取器冻结，读取器保留损失到 memory 的梯度。预训练不更新边界模块。

### 3.2 后训练（QA／GRPO 联合训练）

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --stage post \
  --config configs/v3_1_1/post_k256.json \
  --run-id policy-post_k256_20261104 \
  --init-checkpoint 'artifacts/v3-1/pretrain_k256_20261104/train/checkpoints/global_step_<N>' \
  --epochs 1 \
  --micro-batch-size-per-gpu 2 \
  --gradient-accumulation-steps 4
```

- **运行目录**：`run_id` 默认为 `policy-post_k{memory_slots}_{YYYYMMDD-HHMM}`，其中 `memory_slots` 为记忆槽位数。
- **初始化**：后训练需通过 `--init-checkpoint` 指定预训练来源。源 checkpoint 阶段须为 `pretrain`，两阶段基座来源、记忆槽位数、LoRA 及梯度检查点设置须一致；精度、混合精度和 CPU 暂存可独立设置，可从 BF16 预训练初始化 FP32 后训练。边界模块隐藏维度由后训练独立设置；新 run 只继承压缩器，重新创建边界模块、optimizer、scheduler、游标和边界随机流。实际来源随训练记录保存。
- **训练单位**：每行一篇文档及其全部 QA。一篇文档采样 G 条完整硬分段，每条路径共享 memory 评价同一套 QA；G 不计入全局 batch。
- **一次联合更新**：固定参数采样并评价整组路径，逐路径回放；QA 逐题累积 memory 梯度并释放计算图，再与策略项一起回传压缩器，累计整批后更新一次。同一提交段内的边界判断共享因果前向，压缩排除仅用于判断的下一单元。QA 损失训练记忆内容，停止梯度后的组内优势训练边界选择。
- **反传跨度**：在 JSON 中设置 `training.backward.bptt_steps`，支持 `null`（保留完整跨段反传）或正整数。例如设为 `2`，每两次实际写入为一组，组内正常反传，组间截断后续压缩与边界判断使用的历史梯度。全部历史 memory 仍参与前向，所有块保留直接 QA 监督；预训练、无梯度采样和评估不受影响。
- **归约与尾批**：损失按实际文档数平均，尾批不补齐、不丢弃、不跨 epoch 拼接；空 rank 同样参加 collective。

### 3.3 中断恢复

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --resume \
  --checkpoint 'artifacts/v3-1/pretrain_k256_20261104/train/checkpoints/global_step_<N>'
```

- **恢复起点**：仅接受当前 run 最新的完整 checkpoint。
- **恢复状态**：可训练参数、optimizer、scheduler、数据游标、课程阶段与阶段内轮数及各 rank 的随机状态。
- **参数来源**：仅用 `--checkpoint` 定位原训练计划，恢复全部实验设置；其他已知 CLI 实验设置忽略，启动时集中提示被忽略的选项。
- **一致性要求**：原计划与 checkpoint 保存的配置须一致，GPU 数量和数据须保持一致。修改训练设置应启动新 run，恢复后训练不依赖预训练来源文件。
- **执行计划**：沿用原训练计划；预算已完成时直接返回，不读取或启动评估计划。

## 4. 评估

Test、LoCoMo 和 RULER 共用独立评估入口，仅接受后训练 checkpoint，模型与方法设置从 checkpoint 读取。新建时确定数据、评估预算和 checkpoint 范围；中断后从已有计划继续。

### 4.1 启动评估

以下以 QASPER test 为例，评估原运行的 step 25、50：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --evaluate-only \
  --eval-run 'artifacts/v3-1/policy-post_k256_20261104' \
  --eval-id standard_20261104 \
  --eval-task test \
  --eval-data 'data/delta-mem-qasper' \
  --checkpoint global_step_25 global_step_50 \
  --eval-batch-size-per-gpu 8
```

切换任务时，替换 `--eval-task` 与 `--eval-data`，并按需设置该任务的专属参数：

| `--eval-task` | `--eval-data` | 专属参数 |
|---|---|---|
| `test` | `data/delta-mem-qasper` | `--max-test-samples` 限制文档数，`--max-new-tokens` 设置每题生成上限 |
| `locomo` | `data/benchmarks/locomo` | |
| `ruler` | `data/benchmarks/ruler-v1/Qwen3-4B-Instruct-2507` | `--lengths` 选择长度 |

- **Benchmark 准备**：Benchmark 数据须提前准备，数据布局与入口见[公共 benchmark 指南](../../benchmarks/README.md#2-数据准备)。
- **Test 协议**：固定读取数据目录的完整 `test.jsonl`，不套用训练长度筛选，默认每题最多生成 128 tokens；同文档全部 QA 共用记忆。
- **运行与范围**：`--eval-run` 必填，指向运行根目录。`--checkpoint` 可选，只填一个或多个 `global_step_<N>` 名称，自动查找 `<eval-run>/train/checkpoints/<名称>/state.pt`；省略时选择当时保存的全部完整 checkpoint。
- **评估计划**：新建时须提供 `--eval-data`，默认先将选定范围与评估设置保存到 `<eval-run>/plan/eval-<eval-id>/<task>.json`，再执行评估，结果保存到 `<eval-run>/eval/<eval-id>/<task>/`。`--eval-id` 默认 `YYYYMMDD-HHMMSS`；同一标识下各任务独立管理，目标任务计划已存在时直接报错。
- **并行与批量**：多张可见 GPU 共同处理一个 checkpoint 的样本，每卡一份模型；`--eval-batch-size-per-gpu` 控制每卡批量，默认 1。多个 checkpoint 按 step 升序依次评估。 Test 按文档组批，benchmark 按题组批，RULER 按 length 分组。
- **覆盖评估**：追加 `--overwrite`，按本次参数重建当前评估任务（`<eval-run>` `<eval-id>` `<eval-task>`）的计划和全部 checkpoint 结果，其他任务、其他 eval-id、其他运行的评估结果保留。目标正在使用时直接报错。

### 4.2 中断恢复或补充

使用已有计划继续评估，也可选择本次范围、补评新 checkpoint。以下假设原 test 计划包含 step 25、50，训练又保存了 step 75，本次只评估 step 50、75：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --evaluate-only \
  --resume \
  --eval-run 'artifacts/v3-1/policy-post_k256_20261104' \
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
  --dashboard-run-dir artifacts/v3-1/policy-post_k256_20261104 \
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

  - **质量比较**：核对基座模型标识、实际输入与参考答案、输入／解码／生成预算及评分口径。基座按模型路径末级名称识别，父目录可以不同；算法、窗口、精度、设备和批量不作为质量比较限制。
  - **数据核对**：使用顺序无关的比较指纹，RULER 按长度分别核对。旧结果可沿用原数据指纹；缺少分组指纹且覆盖范围不同时，提示无法核实对应范围。
  - **资源与成本**：资源统计与质量指标分区展示。批量计算时间按参与样本分摊后累计，不等同于每条请求的独立等待时延；峰值显存按 worker／批次范围观测。比较成本须统一硬件、实际评估批量及测量范围，不自动计算加速比；重试墙钟时间仅覆盖本次执行。
  - **缓存校验**：跨方法展示允许实现与运行环境不同；结果复用、失败重试和训练恢复仍严格核对原有身份与设置。

- **访问方式**：

  - **访问配置**：地址和端口见 [`.streamlit/config.toml`](../../../../.streamlit/config.toml)，修改后须重启看板；临时更换端口可在启动命令前加 `STREAMLIT_SERVER_PORT=8502`。
  - **异地查看**：同步 `<运行目录>/eval/` 并保留目录结构，将 `--dashboard-run-dir` 指向运行根目录。查看 RULER 完整题面还需对应原始数据，在页面“RULER 数据目录”填写本地路径，目录下应包含 `<长度>/<子任务>/test.jsonl`。查看训练曲线还需同步 `train/metrics.jsonl`。

## 5. 调整参数

本节汇总训练、恢复、独立评估、离线准备与看板的公开参数。训练默认值以[预训练配置](../../../../configs/v3_1_1/pretrain_default.json)和[后训练配置](../../../../configs/v3_1_1/post_default.json)为准：新训练的显式 CLI 值在校验前覆盖 JSON，省略时沿用配置。模型结构、方法目标及优化器设置直接修改 JSON，见[配置与运行](../README.md#配置与运行)。本文示例改用 K256 配置：准备阶段上限为估算 4096 tokens，预训练不设实际 token 数上限，后训练上限为 12288；`*_default.json` 保留 K128 设置。

### 5.1 训练输入与执行预算

| 参数 | 用途／默认值 |
|---|---|
| **运行方式** | |
| `--evaluate-only` | 启动评估，默认不开启 |
| `--resume` | 恢复训练或补充评估，默认不开启 |
| `--overwrite` | 覆盖式训练或评估，默认不开启 |
| **配置与模型** | |
| `--stage` | 训练必填，训练阶段，支持 `pretrain` 和 `post` |
| `--config` | 训练必填，JSON 配置文件 |
| `--model-path` | 训练必填，基座与 tokenizer 路径 |
| `--seed` | 非负随机种子，默认 42 |
| `--init-checkpoint` | 后训练必填，后训练初始化所用的预训练 checkpoint |
| **目录与运行标识** | |
| `--output-root` | 训练产物父目录，默认 `artifacts/v3-1` |
| `--run-id` | 训练运行目录名，默认见[目录命名](#7-产物目录与记录) |
| **离线数据准备** | |
| `--prepare-data` | 离线数据准备必填；`prepare-pre`／`prepare-post` 为来源数据的 JSONL 文件，`prepare-length` 为来源数据目录 |
| `--output-file` | `prepare-pre`／`prepare-post` 必填，完整输出文件路径 |
| `--output-dir` | `prepare-length` 必填，完整输出数据集目录 |
| `--split` | `prepare-length` 必填，处理对应 split，支持 `train`、`dev` 和 `test` |
| `--target-samples` | `prepare-pre`／`prepare-post` 的目标生成条数，支持正整数和 `null`（默认） |
| `--min-tokens` | `prepare-pre`／`prepare-post` 的估算长度下限，支持正整数和 `null`（默认） |
| `--max-tokens` | `prepare-pre`／`prepare-post` 的估算长度上限，支持正整数和 `null`（默认） |
| `--length-bins` | `prepare-pre` 的长度区间数，默认 10；`prepare-length` 的长度分档数，默认 4 |
| `--length-distribution` | 仅 `prepare-pre`，不同长度区间样本数分布，支持 `uniform`（默认）和 `random` |
| `--qas-per-sample` | 仅 `prepare-post`，每条数据 QA 数上限，支持正整数和 `null`，默认 10 |
| **数据集与筛选** | |
| `--train-data` | 训练必填，训练与 dev 验证的数据目录 |
| `--max-train-samples` | 训练样本数上限，支持正整数和 `null`（默认） |
| `--max-train-data-tokens` | 预训练完整样本／后训练正文的实际 token 数上限，支持正整数和 `null`（默认） |
| **训练时长** | |
| `--epochs` | 普通训练轮数，仅关闭长度课程时生效，默认 1 |
| `--length-curriculum` | 按长度档位从短到长分阶段训练，默认不开启 |
| `--curriculum-stage-epochs` | 长度课程每阶段完整遍历本档数据的轮数，默认 1 |
| `--curriculum-replay-ratio` | 长度课程的历史补充数量／本档样本数，支持 `[0, 1]`，默认 0.2 |
| **批处理** | |
| `--micro-batch-size-per-gpu` | 每卡 microbatch 的样本数，默认 1 |
| `--gradient-accumulation-steps` | 每次优化器更新前累积的 microbatch 数，默认 16 |
| **计算设置** | |
| `--mixed-precision` | FP32 参数与标准 BF16 autocast，默认不开启 |
| `--save-on-cpu` | 反向所需的保存张量暂存到 CPU，默认不开启 |

- **数据筛选**：预训练按完整样本 `text`、后训练按正文的实际 token 数筛选，不含系统提示、chat template 等。train 和 dev 的长度均须严格大于记忆槽位数 K；上限仅用于 train，超限样本整条排除，`null` 取消该上限。LM 划分后的压缩前缀若不大于 K 则改为 AE，实际编码／解码仍须满足模型窗口限制。

### 5.2 验证、评估与结果记录

| 参数 | 用途／默认值 |
|---|---|
| **Checkpoint** | |
| `--save-every` | checkpoint 保存的正整数 optimizer step 间隔或 `after_each_epoch`（默认） |
| `--save-total-limit` | 保留最近成功保存的 checkpoint 数，默认 2 |
| `--checkpoint` | 训练恢复用单个完整路径；评估只填 `global_step_<N>` 名称，支持多个；新建或覆盖时省略选当时全部，续跑时省略选累计列表全部，传入则仅调度指定项并累计新项 |
| **Dev 验证** | |
| `--max-dev-samples` | dev 验证样本数上限，支持正整数和 `null`（默认） |
| `--dev-every` | dev 验证的正整数 optimizer step 间隔或 `after_each_epoch`（默认） |
| **Test 与 benchmark 评估** | |
| `--eval-run` | 评估必填，指向运行根目录，用于保存评估计划和 checkpoint 确认等 |
| `--eval-id` | 评估标识，默认 `YYYYMMDD-HHMMSS` |
| `--eval-task` | 评估必填，评估任务，支持 `test`、`locomo` 和 `ruler` |
| `--eval-data` | 评估必填，当前评估任务的数据目录 |
| `--eval-batch-size-per-gpu` | 每卡评估批量，默认 1 |
| `--max-test-samples` | test 评估文档数上限，支持正整数和 `null`（默认） |
| `--max-new-tokens` | test 评估时 token 生成上限，默认 128 |
| `--lengths` | RULER 评估时所用数据的长度，默认 `8192 16384 32768` |
| **SwanLab** | |
| `--tracking` | SwanLab 记录，支持 `online`（默认）和 `offline` |
| **看板** | |
| `--dashboard-run-dir` | 看板必填，指向运行根目录 |
| `--dashboard-eval-id` | 看板启动时选中的 eval-id；省略时默认列表首项，页面可切换 |
| `--dashboard-baselines` | 按“标签 结果目录”成对传入；支持 benchmark 的 `eval/` 或直接含 `summary.json` 的目录 |

- **验证与保存**：除了 `--save-every` 和 `--dev-every` 所指定的 step，每轮末和最终步也保存 checkpoint 及进行 dev 验证；长度课程中一轮指当前档位的一轮训练，`after_each_epoch` 同样适用。

## 6. 配置变更与结果复用

| 变更 | 操作 |
|---|---|
| 预训练准备的来源文件、目标数量、完整样本估算长度范围、区间数或分配方式 | 修改准备参数，准备对应 split；训练仍按实际 token 数筛选，再启动新训练 |
| 后训练准备的来源文件、目标数量、正文过滤范围或每条 QA 上限 | 调整 `prepare-post` 参数，准备对应 split，再启动新训练 |
| 长度分档的正文、QA、tokenizer 或档位数量 | 重新执行 `prepare-length`，检查分布后启动新训练 |
| 从预训练进入后训练阶段 | 新 run，通过 `--init-checkpoint` 载入兼容压缩器 |
| 模型结构、计算、方法目标或优化器设置 | 修改基础 JSON，再启动新训练 |
| 数据、batch、轮数或保存等运行设置 | 新训练使用 CLI 覆盖或修改基础 JSON |
| 评估重试、调整评估批量或 GPU 数量 | 使用 `--evaluate-only --resume`，以原 `--eval-run`、`--eval-id` 与 `--eval-task` 定位计划，只重新分片并组批原计划的未完成样本 |
| 选择本次评估 checkpoint、增加 checkpoint | 续跑时传 `--checkpoint` 名称列表，仅调度指定项；新项并入累计计划，不缩减旧列表 |
| 调整 RULER 长度 | 另开 eval-id 新建计划，或用 `--overwrite` 重建当前任务 |
| 重新创建当前评估任务 | 显式 `--overwrite`，重新提供 `--eval-data` 与本次设置；当前 task 的旧计划及结果清空 |
| 仅重新展示既有评估 | 启动或刷新独立看板，不执行模型 |

## 7. 产物目录与记录

产物默认保存在 `artifacts/v3-1/`，可通过 `--output-root` 指定父目录。各 run 的计划、训练和评估分别归档：

```text
<output_root>/
└── <run_id>/
    ├── plan/
    │   ├── config.json                       stage、seed、model 与 method
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

- **目录命名**：完整训练路径为 `<output_root>/<run_id>/train`。`run_id` 默认为 `pretrain_k{memory_slots}_{YYYYMMDD-HHMM}`（预训练）或 `policy-post_k{memory_slots}_{YYYYMMDD-HHMM}`（后训练），时间戳采用 `Asia/Shanghai` 时区、精确到分钟。显式指定的 `run_id` 原样作为目录名。新训练默认拒绝覆盖已有计划或产物，显式 `--overwrite` 才清理并重建；恢复训练沿用原目录。
- **计划与配置**：`plan/` 保存执行前确定的设置，最终解析生成的 `run_id` 写入计划配置；`train/config.json` 保存实际训练配置，评估结果目录的 `eval.json` 保存评估设置快照。训练只创建训练计划；独立评估默认新建不存在的 task 计划，固定数据、预算与长度，`checkpoints` 保存累计范围。续跑可选择本次 checkpoint 子集并加入新项，仅有新增时原子更新计划；`--overwrite` 重建当前 task。
- **覆盖与互斥**：校验通过后，持删除范围之外的稳定锁执行清理与重建；锁覆盖 worker 生命周期，占用时拒绝覆盖。锁文件不属于实验结果。
- **训练与恢复**：Rank 0 统一写入产物，train/dev 指标汇总全部 GPU 的实际样本。`state.pt` 保存可训练的压缩器 LoRA、memory embeddings、边界参数、优化器、scheduler、数据游标及各 rank 随机状态；恢复时严格核对训练计划与绑定状态，不依赖评估计划。完整格式见[方法说明](../README.md#代码与产物)。
- **评估结果**：各卡结果按原始样本顺序汇总；Test 的 `qa_nll` 先按题、再按文档平均，`token_nll` 按答案 token 加权，EM／F1 按题平均，benchmark 沿用官方汇总。评估续跑复用完整结果；未完成时保留逐文档／题目进度，可调整 GPU 数和评估批量，只重新分片、组批剩余样本。RULER 继续复用已完成长度。
- **比较指纹**：Benchmark summary 的 `comparison_data` 按原始输入、参考答案、生成预算与任务生成，忽略执行顺序，RULER 按长度分组；仅供跨方法展示，不参与 `eval.json`、缓存或恢复校验。
- **日志与状态**：日志同时写入文件和终端，评估输出带 GPU／分片前缀；任务 `status.json` 的 `eval_run` 指向运行根目录，`selected_checkpoints` 是本次调度范围，与计划 `checkpoints` 的累计范围区分。状态记录 checkpoint 与各分片的设备、命令、起止时间、返回码和错误，整体进度仅统计本轮选择。
- **展示方式**：SwanLab 默认开启，记录 train/dev、训练资源及配置。project、group 和 tags 通过配置文件设置；project 默认为 `latent-working-memory-v3.1`，group 未指定时默认 `run_id`。`--tracking offline` 关闭 SwanLab 记录，仅保留本地产物；恢复训练沿用 `swanlab.json` 中的 run。评估由独立 dashboard 展示，不读取 SwanLab 身份或凭据。
