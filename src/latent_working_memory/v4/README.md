# 20261003_v4 逐层 Attention Matching 记忆

创建时间：20261003 21:51:43 Asia/Shanghai（UTC+08:00）

最后修订时间：20261019 15:44:38 Asia/Shanghai（UTC+08:00）

冻结基座，在每层维护 latent memory，通过 Attention Matching 局部更新记忆，并由答案 CE 学习如何生成压缩查询和读写记忆。

[运行指南](scripts/README.md) · [Benchmark 数据准备与评估](benchmarks/README.md) · [默认配置](../../../configs/v4/attention_matching/experiment.json) · [框架笔记](../../../notes/v4/20261003_attention_matching_meta_memory.md)

## 一、模型与训练

### 1.1 记忆组织与读取

| 部分 | 当前实现 |
|---|---|
| 基座 | 冻结的 full-attention Llama／Qwen2／Qwen3；使用一份模型权重 |
| 序列 | 已压缩历史、待压缩 pending、最新 recent 三部分；后两部分保留原始 KV |
| 记忆 | 每层独立保存固定数量的 latent slots，每条 QA 样本独立初始化 |
| 读取 | 由当前隐藏状态生成读取 query，读出 memory 响应，分别修正 attention 的 Q projection 与 output projection |
| 写入 | 原始缓存达到 `pending_size + recent_size` 时压缩最早的 pending，移除对应 KV，保留 recent |

- pending 的逐层输入经投影得到源表示，保存在独立源缓存中；首次写入前不启用记忆读取修正。
- 写入后的 memory 用于后续前向；保留的 recent KV 不重算，绝对位置不重置。
- `query_mode` 可选 `fixed`（可学习、输入无关的查询）或 `conditioned`（根据当前 pending 生成查询）。
- `query_normalization` 与生成方式独立，可选 `none` 或 `fixed_norm`；后者约束读取查询，以及压缩生成器的初始查询和最终输出范数。

### 1.2 局部更新与外层训练

对任意一层，记 pending 源表示为 $S$、当前记忆为 $M$、压缩查询为 $Q=G_\theta(S)$。源表示与记忆使用同一个局部 reader：

$$
\mathcal L_{\mathrm{write}}
=\frac{1}{2R}\sum_{r=1}^{R}
\left\|\operatorname{Read}_\theta(Q_r,M)-\operatorname{Read}_\theta(Q_r,S)\right\|_2^2,
\qquad
M^{j+1}=M^j-\eta\nabla_{M^j}\mathcal L_{\mathrm{write}}.
$$

$R$ 为压缩查询数；每次写入固定执行 `inner_steps` 步，查询与源响应在该次内循环中固定。

| 阶段 | 更新对象与目标 |
|---|---|
| Inner loop | 只更新当前样本的 slots，最小化局部 Attention Matching 损失 |
| Outer loop | 用答案 token CE 学习初始 slots、查询生成器、投影、共享 reader 及 Q/O 修正模块；基座冻结 |
| 验证／推理 | 学到的模块参数固定，仍执行相同的局部记忆更新 |

- 外层 CE 按全局有效答案 token 数平均；训练保留 teacher、query generator 和跨写入边界的完整 meta-gradient。
- 每层 memory 与局部计算使用 FP32，显式 attention reader 支持二阶梯度；不使用一阶近似或截断 BPTT。
- 验证／生成关闭外层梯度，但局部 TTT 仍需 autograd，不能使用 `torch.inference_mode()`。
- 当前不包含独立历史约束、自适应扩容或 query generator 自身的 TTT。

## 二、数据

### 2.1 格式约定

- UTF-8 JSONL，每行一篇完整正文及其 QA；同一文件内正文唯一。
- 包含 `text` 和 `qas`，每个 QA 包含 `question` 和 `answer`，其中文本字段均为非空字符串，`qas` 为非空列表。

```json
{
  "text": "完整正文",
  "qas": [
    {"question": "问题一", "answer": "答案一"},
    {"question": "问题二", "answer": "答案二"}
  ]
}
```

### 2.2 读取与监督

- **样本组织**：遵循 δ-Mem 的组织形式，完整正文配上一条 QA 作为独立样本，这意味着一条包含若干不同问题的数据可以构造出多条训练/评估样本。
- **监督目标**：正文与问题作为输入，在 outer loop 计算答案 CE。
- **长度下限**：所有划分的样本均要求正文至少为 `pending_size + recent_size + 1` tokens，低于下限时整条过滤。
- **正文上限**：仅 train 可通过 `max_train_data_tokens` 限制原始 `text` 的 token 数，按 `tokenizer.encode(text, add_special_tokens=False)` 计算，不含问题、答案、系统提示与 chat template；dev/test 不使用该上限。
- **模型窗口**：所有划分的正文、问题、答案及模板总 token 数均不得超过模型窗口。任一长度条件不满足时整条过滤，不截断。
- **数量上限**：QA 展开并完成长度过滤后，train/dev 可分别限制独立 QA 样本数；按 `seed`、划分名和稳定样本 ID 确定随机子集，上限为 `null` 或不小于可用数时保留全部。划分隔离检查仍使用完整的原始文档集合。
- **固定子集**：每次运行选定一次，跨 epoch 不重抽；仅 train 每个 epoch 打乱顺序，dev 各次验证共用同一子集。test 与 benchmark 不受 train/dev 数量上限影响。

### 2.3 当前数据：δ-Mem QASPER

- **来源与转换**：[δ-Mem 发布集](https://huggingface.co/datasets/huaXiaKyrie/delta-mem-qasper-data) 的原始文件保存在 `data/raw/delta-mem-qasper/`。去除对话编号及固定提问包装，按原顺序拼接正文，并将同一论文的全部 QA 聚合为一行。
- **固定划分**：`data/delta-mem-qasper/` 保存 `train.jsonl`、`dev.jsonl`、`test.jsonl`，按原始论文划分，保存时不做长度过滤；来源 ID 与转换规则记录在 `preparation.json`。
- **实验预设**：train 正文上限 12288 tokens；dev 计算答案 CE，test 留作独立评估。dev/test 均为本实验留出集。
- **构造与使用**：见[数据说明](../../../data/delta-mem-qasper/README.md)与[构造入口](../data_preparation/delta_mem_qasper/README.md)。

## 三、配置与执行

- [experiment.json](../../../configs/v4/attention_matching/experiment.json) 是默认配置及格式示例，可通过 `--config` 指定其他文件。
- 预设 JSON 顶层使用 `seed`、`model` 和 `training`。运行时公共 `seed` 与 `model` 保存到 `plan/<method-dir>/config.json`，seed 创建后固定；`train.json` 的 `training` 保存训练配置，`execution` 保存 `train_only`、自动 test 数据目录 `test_data` 与样本上限 `max_test_samples`。`run.json` 和 checkpoint 保存完整实际配置。
- 本地路径支持 `~`；相对数据、输出路径以运行目录为基准，`run.sh` 统一从项目根目录执行。
- 下表为仓库预设；CLI 参数、覆盖规则和运行目录见[运行指南](scripts/README.md)。

**记忆与局部更新（`model`）**

| 参数 | 预设与含义 |
|---|---|
| `pending_size`／`recent_size` | 256／128；首次写入发生在处理完 384 tokens 后 |
| `num_slots`／`memory_dim` | 每层 32 slots，每个 128 维 |
| `query_dim`／`value_dim`／`num_probes` | 64／128／16 |
| `query_mode`／`query_normalization` | `conditioned`／`none`，选择见 1.1 节 |
| `read_query_norm`／`compression_query_norm` | `fixed_norm` 模式的固定范数；省略或 `null` 时为 `sqrt(query_dim)`，此处为 8 |
| `inner_steps`／`inner_lr` | 1／0.1；训练与推理采用相同更新规则 |
| `correction_scale` | Q/O 修正尺度 0.1 |

**外层训练与执行**

| 参数 | 预设与含义 |
|---|---|
| `model.model_name_or_path` | `~/models/Qwen3-4B-Instruct-2507` |
| `training.dataset_dir` | `data/delta-mem-qasper`；训练读取 `train.jsonl` 和 `dev.jsonl`，配套 test 读取 `test.jsonl` |
| `model.backbone_dtype` | GPU 预设 `bfloat16`；类默认 `float32`，新增模块与 slots 始终 FP32 |
| `model.attention_implementation` | 主干 `sdpa`，可选 `eager`；局部 reader 使用显式计算 |
| `model.loss_chunk_size` | 256；有效答案位置的 LM head／CE 分块重计算 |
| `training.epochs`／`learning_rate` | 1／0.0001 |
| `training.micro_batch_size_per_gpu`／`gradient_accumulation_steps` | 4／2；双卡全局 batch 为 16 |
| `training.max_train_data_tokens` | train 原始正文的 token 上限 12288；省略或 `null` 关闭 |
| `training.max_train_samples`／`max_dev_samples` | 过滤后的 train／dev 独立 QA 样本数上限；正整数或 `null`，预设 `null`（全部） |
| `seed` | 42；控制新增记忆模块初始化、train/dev 子集选择及训练样本打乱，test／benchmark 始终读取该公共值 |
| `training.dev_every` | 每 50 个 optimizer steps 验证；epoch 结束时也验证 |
| `training.save_every`／`save_total_limit` | 每 50 steps 保存、保留最近 2 个；支持 `after_each_epoch`，epoch 结束时始终保存 |

- 全局 batch = GPU 数 × 每卡 microbatch × 梯度累积步数。预设每卡累积 2 次 microbatch 后更新优化器；尾批按实际有效答案 token 数归一化。
- 使用 verl Engine、optimizer、DDP、collator、StatefulDataLoader 和 checkpoint 管理；恢复保留配置、加载器游标、各 rank RNG 和原运行身份。
- 同一 run 恢复要求配置、分词后数据、设备类型及 GPU 数一致。更改方法设置时使用新 run。
- 训练、恢复和评估均从配置中的本地基座模型目录加载 tokenizer；checkpoint 保存新增记忆模块与训练状态。
- 恢复与评估使用运行记录和 checkpoint 绑定的模型配置，不受公共 `config.json` 后续修改影响。
- 主干尚未采用带可变 KV／hook 状态的激活重计算；推理缓存有界不代表完整 meta-training 显存有界。

从项目根目录启动：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --gpus 0,1 --run-id am-query_20261013
```

## 四、评估与记录

### 4.1 评估方式

| 场景 | 计算内容 |
|---|---|
| 训练中 dev | 答案 token CE，以及局部写入前／后的 AM loss |
| QASPER test 评估 | 从 checkpoint 计算 NLL、greedy 生成的 EM／F1，保存逐题结果及汇总 |
| Benchmark 评估 | RULER v1 的 13 项任务与 LoCoMo 类别 1–4；独立生成，沿用官方评分 |

- 统一入口在训练或续训完成全部预算后自动执行最终 checkpoint 的 QASPER test；等待训练进程退出，再使用所选 GPU 中的第一张卡，默认生成上限为 128 tokens。新训练加 `--train-only` 时保留训练中 dev、跳过最终 test，该选择与最终 test 数据目录保存到 `train.json` 的 `execution`，恢复时沿用。训练中断或失败时不执行 test。
- NLL 按答案 token 加权，EM／F1 按 QA 平均；EM／F1 使用小写化、去标点及英文冠词、合并空白后的短答评分。这是当前单参考答案协议，不是官方 QASPER 多参考评测。
- NLL 与生成各自初始化记忆；生成仅接收 prompt，参考答案不进入生成路径。
- test 沿用固定正文下限及模型窗口上限；生成预算超过剩余窗口时缩至可用长度，记录实际预算，不改变样本筛选。
- 独立评估统一使用 `--evaluate-only`，通过 `--checkpoint <目录1> <目录2> ...` 指定同一 run 的一个或多个 checkpoint，或通过 `--eval-run` 指定 `artifacts/v4/<run-id>/train/<method-dir>/`，扫描其中全部 `checkpoints/global_step_*/state.pt`。两者互斥；不接受重复、跨 run 或缺少 `state.pt` 的未完成 checkpoint。多 GPU 时每卡一个进程并行评估不同 checkpoint，全部计算结束后由父进程统一发布。
- 独立评估由 `--eval-type` 选择 `test`（默认）或 `benchmark`。benchmark 模式须通过 `--benchmark` 指定 `ruler` 或 `locomo`，该参数默认 `null`，不用于自动切换评估类型。批量评估中某个 checkpoint 失败后继续其余 checkpoint，最终汇总结果并报错。
- 新训练用 `--data` 指定完整数据集目录，训练、验证与自动 test 分别读取其中的 `train.jsonl`、`dev.jsonl` 和 `test.jsonl`；`--test-data` 可单独覆盖自动 test 数据目录。
- 独立 test 默认沿用保存的评估配置中的数据目录和生成上限；没有该配置时，必须传入 `--test-data`，生成上限默认 128 tokens。`--test-data` 默认 `null`，显式传入时覆盖本次评估数据目录并读取其中的 `test.jsonl`，不修改训练配置。benchmark 沿用已保存的同类评估配置；没有该配置时，必须传入 `--benchmark-data` 指定数据目录。
- 评估类型、benchmark 名称及 benchmark 专用参数仅用于 `--evaluate-only`；不控制自动 test 或训练中 dev。test 不接受 benchmark 参数；benchmark 不接受 `--test-data`、`--max-test-samples` 或 `--max-new-tokens`，`--lengths` 仅用于 RULER。
- 评估结果由 checkpoint 推导到原实验的 `eval/<method-dir>/<test|ruler|locomo>/global_step_N/`，单个、多个 checkpoint 与 `--eval-run` 使用同一目录规则。
- 同一方法、同类评估共用 `plan/<method-dir>/<test|ruler|locomo>.json` 计算协议，记录实际数据内容、模型／分词、筛选、生成与评分设置。评估始终读取公共 `config.json` 的 seed，共用评估配置不重复保存；checkpoint 评估快照与结果记录实际 seed，补评估不一致时报错。独立评估不接受 `--seed`；已有协议后须保持一致。相同 checkpoint、相同协议且已有完整 `summary.json` 时直接复用，不重新加载模型；缺少汇总的未完成评估在原目录从头计算，配置冲突则报错。
- 本地汇总先于在线发布保存，发布失败可重跑原命令补传。tracking、设备分配与报告展示不改变计算协议，可先离线计算，再复用结果在线发布。
- Benchmark 使用官方原始格式，每题独立初始化记忆，输入只应用一次 chat template。严格检查输入与任务生成预算的总长度，不截断或套用训练筛选；数据位置、手动准备指令和评估协议见 [benchmark 说明](benchmarks/README.md)。
- 汇总包含模型各层 slots、维度、总 latent 元素及字节数，以及评估耗时、局部写入耗时和峰值显存。计量范围随结果保存，不将 latent 状态大小视为整个模型显存。

### 4.2 SwanLab 与本地产物

- 默认在线记录到 `latent-working-memory-v4`，使用与 v3 相同的 `SWANLAB_API_KEY` 配置；`--tracking disabled` 仅保存本地。相关 runs 通过 `group` 关联，`job_type=train`，固定 tags 为 `scope:main`、`method:attention-matching`，数据和研究标签通过配置补充。
- `--run-id` 标识实验系列；方法目录按 `<query_mode>-query-m<num_slots>` 自动生成，例如 `conditioned-query-m32`。SwanLab 名称为 `<method-dir>_<run-id>`。
- train/dev 在同一 run 中按真实 optimizer step 追加原生曲线。各分区的 `write_loss` 图合并写入前后两条曲线，以浅色／深色区分；底层分别保存 `write_loss_before` 和 `write_loss`，均先按层、再按写入事件平均。
- QASPER test 沿用原训练的在线设置和身份。单 checkpoint 的 NLL、EM／F1 合并为 `evaluation/quality` 双轴柱图：NLL 左轴从 0 自适应，EM／F1 右轴固定 0–1；写入前后 AM loss 独立展示。在线发布要求训练已完成、云端原 run 为 `FINISHED`。辅助计数与逐题结果保存在本地。
- Benchmark 结果同样追加到原 run，分别用 `evaluation/ruler` 汇总任务／长度、`evaluation/locomo` 汇总 QA 类别及总体分数。
- QASPER 与每个 benchmark 分别维护完整报告：合并此前已发布结果与本次选中结果，按真实 optimizer step 排序后重建。单 checkpoint 使用汇总图，多 checkpoint 的曲线以真实 step 为横轴；媒体滑块表示独立递增的报告发布版本，补评估更早的 checkpoint 后仍按 step 升序展示。
- 各评估目录的 `swanlab.json` 只保存最新成功报告的 run 身份、成员 `summary.json` 路径、报告签名、发布版本和面板信息。上传并核验成功后原子更新，失败时保留此前成功记录；内容相同时仅检查、修复布局。其他离线结果须明确选择后才发布，复用历史汇总无需保留对应 checkpoint 权重。

| 位置 | 主要产物 |
|---|---|
| `plan/<method-dir>/` | `config.json` 公共模型配置与 seed、`train.json` 流程设置与训练配置 |
| `plan/<method-dir>/<test\|ruler\|locomo>.json` | 同类评估共用配置 |
| `train/<method-dir>/` | `train.log`、执行状态 `status.json`、运行信息与数据统计 `run.json`、指标 `metrics.jsonl`、在线运行身份 `swanlab.json`、checkpoint 与训练摘要 `summary.json`（完成情况、进度统计、最终 checkpoint 路径） |
| `eval/<method-dir>/<test\|ruler\|locomo>/` | 任务级 `status.json` 保存本轮任务信息、选择、进度及各 checkpoint 最近执行状态；`swanlab.json` 保存最新成功发布的完整报告记录 |
| `eval/<method-dir>/<test\|ruler\|locomo>/global_step_N/` | 日志 `eval.log`、协议快照 `eval.json`、指标汇总与完成标记 `summary.json`、逐样本详细记录 `details.jsonl`；test 的完整数据统计保存在 `summary.json` 的 `data_statistics` 字段 |

共同父目录为 `artifacts/v4/<run-id>/`。详细文件和命令见[运行指南](scripts/README.md)。当前没有独立跨运行比较入口。

训练 `status.json` 保存命令、GPU、起止时间、返回码和失败原因等执行状态；训练结果统一从同目录的 `summary.json` 读取。

评估任务级 `status.json` 的 `selected_checkpoints` 记录本轮选择，整体状态与 `progress` 仅统计这些 checkpoint；`checkpoints` 以 `global_step_N` 为键，保留各 checkpoint 最近一次的命令、GPU、起止时间、返回码及计算／发布失败详情。补评估保留未选中的旧条目，不累计每次尝试历史。结果复用与完成判定以 `summary.json` 为准，成功发布记录以 `swanlab.json` 为准。

## 五、代码职责

| 职责 | 文件 |
|---|---|
| 配置、JSON 持久化与启动 | [config.py](config.py)、[io.py](io.py)、[gpu_job.py](gpu_job.py) |
| 数据读取与样本表示 | [data.py](data.py)、[data_types.py](data_types.py) |
| 逐层记忆与流式主干 | [memory.py](memory.py)、[model.py](model.py) |
| 外层训练与批处理 | [train.py](train.py)、[engine.py](engine.py)、[training_data.py](training_data.py) |
| 保存恢复与实验记录 | [checkpoint.py](checkpoint.py)、[tracking.py](tracking.py) |
| 评估协议、结果复用、模型加载与计时 | [evaluation_records.py](evaluation_records.py)、[evaluation_runtime.py](evaluation_runtime.py) |
| QASPER 评估 | [evaluate.py](evaluate.py) |
| Benchmark 原始读取、官方评分与评估 | [benchmarks/data.py](benchmarks/data.py)、[scoring.py](benchmarks/scoring.py)、[evaluate.py](benchmarks/evaluate.py) |

```bash
uv run --frozen python -m compileall -q src/latent_working_memory/v4
uv run --frozen ruff check src/latent_working_memory/v4
```
