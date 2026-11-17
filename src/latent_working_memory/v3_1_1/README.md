# 20261102_单次压缩预训练与自适应边界后训练

创建时间：20261102 16:06:43 Asia/Shanghai

最后修订时间：20261117 20:32:01 Asia/Shanghai

v3.1.1 先通过文本重建（AE）和续写（LM）训练压缩器（Writer），再用问答（QA）与强化学习联合训练记忆内容和分段边界。每次提交将一段正文压缩为固定 K 个记忆槽位（memory slots），已提交的记忆块保持不变；读取器（Reader）根据这些记忆回答问题。

本文说明方法、配置和执行协议；完整 CLI 参数与各任务命令见[运行指南](scripts/README.md)。

[方法概览](#方法概览) · [数据准备与课程](#数据准备与课程) · [配置与运行](#配置与运行) · [验证、独立评估与记录](#验证独立评估与记录) · [实现细节](#实现细节) · [代码与产物](#代码与产物)

## 方法概览

### 记忆与边界

| 符号 | 配置 | 含义 |
|---|---|---|
| K | `model.structure.memory_slots` | 每次提交生成的记忆槽位数 |
| U | `model.structure.unit_tokens` | 后训练与评估的正文处理单元长度，决定边界判断粒度 |
| G | GRPO 的 `training.rl.group_size`；SCST 的 `training.rl.num_samples` | 每篇文档的随机分段路径数 |

后训练按 U 将正文分成处理单元。在线状态由历史 memory、待提交原文 a 和新到达单元 b 组成。边界模块读取这些已到达的内容，选择继续积累 ab，或将 a 压缩为 K 个 slots，再以 b 开始下一块。正文末尾的剩余内容强制提交，这次写入不计为策略动作。

问题和任务指令交给 Reader，正文分段独立于 QA。Reader 在正文压缩完成后回答，生成期间保持 memory 不变。Writer 与 Reader 的模块及训练状态独立，共享冻结的基座参数；LoRA 仅作用于 Writer。

### 单次压缩预训练

每条完整 `text` 是一个样本，按 seed 和文本内容固定选择 AE 或 LM，跨 epoch 保持一致。

| 目标 | 压缩输入 | 预测目标 |
|---|---|---|
| AE | 完整文本 | 重建完整文本 |
| LM | 去掉末尾目标 tokens 后的前缀 | 文本末尾的 `lm_target_tokens` 个 tokens |

`training.objective.lm_ratio` 默认 0.5，`lm_target_tokens` 默认 512。LM 前缀长度不大于 K 时改用 AE；两个目标均追加 EOS 结束标记。每个样本按目标 tokens 平均负对数似然（NLL），再对样本等权平均。

预训练只更新 Writer LoRA 和可学习 memory embeddings。冻结的 Reader 保留损失到 memory 的梯度，边界模块在后训练阶段训练。

### QA 与策略联合训练

一篇完整文档及其固定 QA 集合是一个训练单位。每条随机路径处理整篇正文，各题共享该路径的记忆。QA NLL 先在每题的答案 tokens 内平均，再依次对题目、随机路径和文档等权平均。

每条路径 g 的奖励及联合目标为：

```text
rho_g = K × blocks_g / input_tokens
memory_cost_g = rho_g ** memory_cost_power
reward_g = -stopgrad(QA_NLL_g) - memory_cost_weight × memory_cost_g
loss = mean(QA_NLL_g) + policy_weight × policy_loss
```

`input_tokens` 是实际送入压缩器的 `input_ids` 长度，含上下文提示前缀；QA、padding 和追加的 memory tokens 不计入。`rho_g` 表示记忆保留比例。`memory_cost_power`（p）为正数，默认 1.0；`memory_cost_weight`（α）为非负数，默认 0.1。p=1 时，同一文档每增加一块的惩罚为 `α × K / input_tokens`。

`training.rl.strategy` 决定优势的计算方式：

| 策略 | 随机路径数 | 优势 |
|---|---|---|
| `grpo` | `group_size >= 2` | `(reward_g - mean(reward)) / (std(reward) + advantage_epsilon)`，使用总体标准差 |
| `scst` | `num_samples >= 1` | `reward_g - greedy_reward`，直接使用奖励差 |

SCST 每篇额外执行一次 greedy 评分，与随机路径共用参数、输入和奖励公式。greedy 路径只提供基线，随机路径承担 QA 与策略反传；`num_samples` 只统计随机路径。greedy 评分保持边界采样随机流不变，同篇相同动作共用 QA 评分。SCST 优势保留原始奖励差，`advantage_epsilon` 仅用于 GRPO。

两种策略均使用 clipped policy loss。对路径中的决策求平均，再对路径和文档平均；奖励、优势及旧动作概率均停止梯度。QA 损失独立按正权平均，策略损失由 `policy_weight` 加权。每次 optimizer update 重新采样，全部 microbatch 完成后统一更新参数。

默认 α=0.1 来自 `policy-post_k256_20261113` 的 step 69 checkpoint：对 118 篇 dev 文档的固定 1／2／4／8 块候选，按参考答案逐文档选择目标最小项。该冻结模型校准保留了无惩罚候选最优 NLL 收益的 90.72%，相对单块的额外保留比例减少 48.52%，平均块数从 2.551 降至 1.822，其中 55 篇选择多块。记录位于 `artifacts/v3-1/policy-post_k256_20261113/eval/boundary-cost_20261116/alpha-calibration.json`。

### 温度退火

`training.rl.policy_transition` 控制随机策略向确定性策略的过渡，GRPO 和 SCST 均可使用。`null` 表示固定 T=1；线性退火配置如下：

```json
{
  "method": "temperature_annealing",
  "schedule": "linear",
  "start_temperature": 1.0,
  "end_temperature": 0.2,
  "start_progress": 0.3,
  "end_progress": 0.9
}
```

当前支持 `temperature_annealing` 和 `linear`。温度满足 `start_temperature > end_temperature > 0`，进度满足 `0 <= start_progress < end_progress <= 1`，总更新数至少为 2。

设实际总 optimizer steps 为 S，即将执行的更新编号为 s（从 1 开始），进度为 `(s - 1) / (S - 1)`。开始进度前保持初始温度，区间内线性下降，结束进度后保持终止温度。上述配置在前 30% 保持 T=1，中间 60% 降至 T=0.2，最后 10% 保持 T=0.2。

训练按 `sigmoid(logit / T)` 采样，old log-prob、回放 log-prob 和策略熵使用同一温度；同次更新的所有 microbatch 与路径共用 T。greedy、dev 和独立评估按 `logit >= 0` 决定动作，训练内 dev 的熵使用该步温度。

总步数由数据、普通训练或课程预算及 `max_steps` 解析。进度贯穿整个 run；恢复训练由保存的 optimizer step 和原总预算继续计算温度。

## 数据准备与课程

### 数据格式与计长

预训练与后训练均读取准备后的 `train.jsonl`、`dev.jsonl`，每行分别为：

```json
{"text":"完整预训练文本"}
```

```json
{"text":"完整正文","qas":[{"question":"问题","answer":"答案"}]}
```

后训练每题必填 `question`、`answer`，可成对附带来源字段 `segment_id`、`segment_char_span`，例如 `"segment_id":"seg0","segment_char_span":[0,4]`。区间是相对当前正文的左闭右开字符下标，用于来源与位置分析，并随逐题结果展示。QASPER 的 question/answer 数据同样可读。

数据准备和 train/dev 长度筛选均使用 tokenizer 对完整正文计算的 token 数，不添加 special tokens、系统提示或 chat 前缀。训练筛选要求正文长度严格大于 K；`training.data.max_train_data_tokens` 只限制 train，超限样本整条排除。筛选与限量在内存中完成，独立 test 读取完整正文。

### 准备预训练与后训练数据

两个准备脚本共用以下入口约定：

| 参数 | 用途 |
|---|---|
| `--prepare-data` | 本次读取的来源 JSONL 文件 |
| `--model-path` | 必填，使用该模型的 tokenizer 计长 |
| `--target-samples` | 正整数指定样本数；`null` 使用全部合条件数据，指定数量须在可用范围内 |
| `--seed` | 随机种子，默认 42 |
| `--output-file` | 新建输出 JSONL；同目录保存构造参数与统计，如 `train.jsonl` 对应 `preparation-train.json` |

**预训练 `prepare_pre.sh`** 将 FineWeb Multisegment 的 `text + continuation` 合并，按目标 token 长度随机裁剪连续原文，再对最终 `text` 重新分词计长。

- `--min-tokens`、`--max-tokens` 接受正整数或 `null`；`null` 采用对应来源边界，请求范围与来源长度范围取交集。
- `--length-bins` 默认 10，整数长度不足时减少档数。
- `--length-distribution uniform` 尽量均衡分配各档样本，`random` 逐样本等概率选择档位；档内均随机选择整数目标长度，实际分布由来源容量决定。

**后训练 `prepare_post.sh`** 保留 FineWeb FactQA 的完整正文，按正文长度筛选，从 `task` 题池选择 QA。`--qas-per-sample` 默认最多 10 题，题数较少时全保留，`null` 使用全部 task 题。按保留题数 N 等分全文，依次选择证据最靠近各份中心的题，每题最多一次，最终按证据位置排序。

以下示例准备最多 4096 tokens 的预训练 train/dev，train 取 5000 条：

```bash
for split in train dev; do
  if [ "$split" = train ]; then target_samples=5000; else target_samples=null; fi
  bash src/latent_working_memory/v3_1_1/scripts/prepare_pre.sh \
    --prepare-data "data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/${split}.jsonl" \
    --output-file "data/fineweb_train5k_20261112/${split}.jsonl" \
    --model-path /root/rivermind-data/models/Qwen3-4B-Instruct-2507 \
    --seed 42 \
    --target-samples "$target_samples" \
    --max-tokens 4096 \
    --length-bins 10 \
    --length-distribution uniform
done
```

后训练构造命令见[后训练数据准备](scripts/README.md#22-后训练数据)。

### 长度分档与课程

`prepare_length.sh` 读取已准备目录中的指定 `<split>.jsonl`，按完整正文 token 数划分等宽档位，默认 4 档。各条正文及 QA 保持完整，分档结果写入 `<split>/bin_00.jsonl` 等文件，其余 split 原样复制；`preparation-<split>.json` 记录边界、样本数和长度统计。命令见[长度分档](scripts/README.md#23-长度分档)。

`training.data.length_curriculum=true` 或启动参数 `--length-curriculum` 开启课程。每轮按以下过程执行：

1. 从短到长选择当前档位，完整使用本档样本。
2. 从此前各档尽量均衡地随机无放回抽取历史样本，数量为本档样本数乘回放比例，四舍五入；历史不足时取全部。
3. 合并、打乱后切 batch，完成本轮训练。每轮重新抽取历史样本。

`training.loop.curriculum_stage_epochs` 控制每档轮数，默认 1；`curriculum_replay_ratio` 控制回放比例，默认 0.2，范围 `[0,1]`。例如本档 250 条时补充 50 条历史样本。对应 CLI 参数为 `--curriculum-stage-epochs` 和 `--curriculum-replay-ratio`。

课程沿用训练的长度筛选与限量，跳过空档，全部档位完成后结束。关闭课程时，各档合并后按普通 `epochs` 全量随机训练。课程只改变样本调度，G、QA 数、全局 batch 和 dev 协议保持原设置；日志记录档位与阶段内轮数，恢复时继续原样本顺序和游标。

## 配置与运行

### 配置组织

配置顶层为 `stage`、`seed`、`model_name_or_path`、`model`、`compute`、`training`。`stage` 取 `pretrain` 或 `post`；`model_name_or_path` 同时指定基座与 tokenizer，可由 `--model-path` 覆盖。

| 分组 | 职责与主要参数 |
|---|---|
| `model.structure` | K、U 和 `boundary_hidden_size`；预训练的后两项为 `null`，后训练为正整数 |
| `model.lora` | `lora_rank`、`lora_alpha`、`lora_target_modules` |
| `compute` | 加载精度、attention、混合精度、层内重算、CPU 暂存 |
| `training.data` | 数据目录、长度和数量筛选、课程开关 |
| `training.loop` | 轮数或更新数、课程参数、batch、验证与保存频率、checkpoint 保留数 |
| `training.objective` | 预训练专用：AE／LM 比例和 LM 目标长度 |
| `training.rl` | 后训练专用：`strategy`、GRPO 的 `group_size` 或 SCST 的 `num_samples`、`policy_transition`、成本与损失参数；`advantage_epsilon` 仅用于 GRPO |
| `training.backward` | 后训练专用：跨段反传长度 `bptt_steps` |
| `training.optimizer` | 压缩器学习率、AdamW 参数、梯度范数裁剪；后训练另含边界学习率 |
| `training.scheduler` | 学习率 warmup 与余弦衰减比例 |
| `training.run` | `output_root`、`run_id`；后训练另含 `init_checkpoint` |
| `training.swanlab` | `tracking`、`project`、`group`、`tags` |

模型结构、目标、RL、优化器及调度参数通过 JSON 设置。新训练的显式 CLI 参数覆盖 JSON，省略时沿用配置。覆盖后的配置保存到计划；解析训练预算后的实际配置写入训练目录、checkpoint 和启用时的 SwanLab config。两阶段均保存完整 `model`，`training` 只保存本阶段适用的字段。源 JSON 保持原样，实际 `training.run.output_dir` 由启动流程生成。

### 配置预设

| 配置 | 用途 |
|---|---|
| [pretrain_default.json](../../../configs/v3_1_1/pretrain_default.json) | 通用 K256 AE／LM 预训练 |
| [post_default.json](../../../configs/v3_1_1/post_default.json) | 通用 K256 GRPO 后训练，4 路采样、固定 T=1 |
| [pretrain_k256.json](../../../configs/v3_1_1/pretrain_k256.json) | FineWeb 5k 预训练与服务器计算设置 |
| [post_k256.json](../../../configs/v3_1_1/post_k256.json) | FactQA 长度课程与截断反传，GRPO 4 路采样、固定 T=1 |
| [post_k256_temp.json](../../../configs/v3_1_1/post_k256_temp.json) | 同套后训练数据与执行设置，GRPO 4 路采样、线性温度退火 |
| [post_k256_scst_temp.json](../../../configs/v3_1_1/post_k256_scst_temp.json) | 同套后训练数据与执行设置，SCST 单随机路径加 greedy 基线、线性温度退火 |

两份 `*_default.json` 的主要设置如下：

| 设置 | 预训练 | 后训练 |
|---|---|---|
| K／U／边界隐藏维度 | `256 / null / null` | `256 / 256 / 128` |
| Writer LoRA | rank 16，alpha 32，目标 `q_proj/k_proj/v_proj/o_proj` | 相同 |
| 计算 | FP32、SDPA；混合精度、层内重算、CPU 暂存均关闭 | 相同 |
| 目标 | `lm_ratio=0.5`，`lm_target_tokens=512` | `strategy=grpo`，`group_size=4`，`policy_transition=null` |
| 策略参数 | — | 成本权重 0.1，成本幂 1.0，策略权重 1.0，`clip_epsilon=0.2`，`advantage_epsilon=1e-6` |
| 学习率 | 压缩器 `1e-4` | 压缩器 `2e-5`，边界 `1e-4` |
| AdamW／裁剪 | `weight_decay=0.01`，`beta1=0.9`，`beta2=0.999`，`adam_epsilon=1e-8`，梯度范数上限 1.0 | 相同 |
| 调度 | warmup 50 steps，`min_lr_ratio=0.1` | 相同 |
| 跨段反传 | — | `bptt_steps=null`，完整反传 |

K256 实验预设使用 `/root/rivermind-data/models/Qwen3-4B-Instruct-2507`，开启 FA2、BF16 autocast 和层内重算。它们的执行设置为：

| 设置 | K256 预训练 | K256 后训练（GRPO／SCST） |
|---|---|---|
| 数据目录 | `data/fineweb_train5k_20261112` | `data/fineweb-factqa_q10_train1k_length3_20261112` |
| 预算 | 普通训练 1 轮 | 3 个长度档位，每档 1 轮，历史回放 20% |
| 每卡 microbatch × 梯度累积 | `16 × 2` | `2 × 8` |
| train 长度上限 | `null` | 12288 tokens |
| warmup／最低学习率比例 | `20 / 0.1` | `5 / 1.0` |
| dev／checkpoint 间隔 | 每 40 steps | 每 20 steps |
| checkpoint 保留上限 | 5 | 6 |
| 跨段反传 | — | `bptt_steps=2` |

两份 `*_temp.json` 均启用线性温度退火：前 30% 进度保持 T=1.0，30%～90% 从 1.0 降至 0.2，最后 10% 保持 T=0.2。进度按整个训练的实际 optimizer steps 计算，跨长度课程档位连续推进。GRPO 使用 `group_size=4`；SCST 使用 `num_samples=1`，每篇另计算一条 greedy 路径作为奖励基线，仅随机路径参与反向传播。

### 计算精度与显存

模型要求 full-attention 和固定 default RoPE。所有可训练参数、梯度和优化器状态保持 FP32；冻结基座的加载精度由 `compute.dtype` 控制。

| `compute` 参数 | 行为 |
|---|---|
| `dtype` | `float32` 或 `bfloat16`，统一控制 Writer／Reader 基座加载精度 |
| `attention_implementation` | 默认 `sdpa`，由 PyTorch 选择内核；`flash_attention_2` 指定 FA2 |
| `mixed_precision` | 默认 `false`；开启时要求 `dtype=float32`，用 BF16 autocast 执行采样、前向与 loss 计算 |
| `gradient_checkpointing` | 默认 `false`；开启 Transformer 层内重算以减少保存的激活 |
| `save_on_cpu` | 默认 `false`；带梯度前向通过 `save_on_cpu(pin_memory=True)` 将反向所需张量暂存至 CPU |

BF16 autocast 的 backward 与 optimizer 更新在 autocast 上下文外执行，使用 FP32 参数更新；梯度检查点恢复原前向精度。CPU 暂存支持两阶段，可与混合精度组合；模型参数、返回的 memory 和采样 KV 保留在 GPU，暂存只用于带梯度前向。

FA2 需要 `flash-attn`，并使用 BF16 基座或 BF16 autocast。项目默认的 Linux x86_64、PyTorch 2.6、CUDA 12.4 环境通过 `uv sync --frozen --group fa2` 安装该可选依赖。保留 FP32 参数时，配置为：

```json
{
  "compute": {
    "dtype": "float32",
    "attention_implementation": "flash_attention_2",
    "mixed_precision": true
  }
}
```

新训练可通过 `--dtype`、`--attention-implementation`、`--gradient-checkpointing`、`--mixed-precision`、`--save-on-cpu` 覆盖对应值。后三项仅传参数名表示开启，也接受小写 `true`／`false`，例如 `--mixed-precision false`。恢复训练和独立评估沿用已保存的计算设置。

### 启动预训练与后训练

从项目根目录运行，使用根目录 `.venv/`。训练与评估脚本默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，外部已设置时沿用。`CUDA_VISIBLE_DEVICES` 显式指定使用的 GPU，每卡一个进程。

新训练同时指定 `--stage` 与 `--config`，两者阶段一致。下面先启动 K256 预训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --stage pretrain \
  --config configs/v3_1_1/pretrain_k256.json \
  --run-id pretrain_k256_20261112 \
  --epochs 1 \
  --micro-batch-size-per-gpu 16 \
  --gradient-accumulation-steps 2
```

以筛选后 5000 条样本计算，该设置训练 1 轮共 157 个 optimizer steps，在第 40、80、120、157 步验证并保存。最终 checkpoint 记录在 `train/summary.json`。

后训练使用新的 run，通过 `--init-checkpoint` 指定预训练来源：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --stage post \
  --config configs/v3_1_1/post_k256.json \
  --run-id policy-post_k256_20261112 \
  --init-checkpoint artifacts/v3-1/pretrain_k256_20261112/train/checkpoints/global_step_157 \
  --micro-batch-size-per-gpu 2 \
  --gradient-accumulation-steps 8
```

预训练来源与后训练须使用相同基座、K 和 LoRA 设置（rank、alpha、目标模块）。启用 GRPO 温度退火时，将 `--config` 改为 `configs/v3_1_1/post_k256_temp.json`；使用 SCST 温度退火时，改为 `configs/v3_1_1/post_k256_scst_temp.json`。每次新训练分别设置 `--run-id`，并通过 `--init-checkpoint` 指定预训练来源。

运行根目录为 `<output_root>/<run_id>/`。显式 `run_id` 原样使用；为 `null` 或省略时，生成 `pretrain_k{K}_{YYYYMMDD-HHMM}` 或 `policy-post_k{K}_{YYYYMMDD-HHMM}`，时间采用 `Asia/Shanghai`，精确到分钟。

### 训练预算与恢复

全局 batch = GPU 进程数 × 每卡 microbatch × 梯度累积次数，预训练按样本、后训练按文档计数。上例两阶段分别为 32 和 16；随机路径数 G 不计入文档 batch。每轮尾批按实际样本数归约，保留全部样本且在本轮完成更新。

- 普通训练由 `training.loop.epochs` 控制完整轮数。
- 长度课程由 `curriculum_stage_epochs` 控制每档轮数，完成全部档位后结束。
- `training.loop.max_steps` 可进一步限制总更新数；解析后的实际步数写入训练配置和 checkpoint。
- `dev_every`、`save_every` 可设步数间隔或 `after_each_epoch`；每轮结束和最终步也执行 dev 与保存。dev 使用固定验证集。
- `warmup_steps` 和 `min_lr_ratio` 控制线性预热与余弦衰减，各参数组共用调度倍率；`min_lr_ratio=1.0` 时，预热后保持恒定学习率。

恢复同阶段训练使用原 run 最新完整 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --resume \
  --checkpoint 'artifacts/v3-1/policy-post_k256_20261112/train/checkpoints/global_step_<N>'
```

恢复沿用原计划、输出目录、基座、计算设置、数据、GPU 数量、样本游标和随机状态；其他已知 CLI 实验设置忽略并提示。后训练恢复直接使用自身 checkpoint，预算完成时返回训练摘要。`--resume` 与 `--overwrite` 互斥。

## 验证、独立评估与记录

### SwanLab 与训练内 dev

`training.swanlab.tracking` 默认 `true`。`--tracking online` 开启 SwanLab，`--tracking offline` 仅保存本地产物。project、group、tags 通过 JSON 设置；默认 project 为 `latent-working-memory-v3.1`，group 缺省使用 `run_id`，`job_type=train`。

train 与 dev 共用一个 run，恢复沿用原身份。两阶段自动追加 `stage:*`，后训练再添加 `method:adaptive-boundary-grpo` 或 `method:adaptive-boundary-scst`；具体超参数记录在 config。

dev 保持模型参数不变，后训练使用确定性边界。train/dev 曲线横轴为真实 optimizer step，只在实际产生结果时追加标量。首次记录某个指标时注册原生图表并配置标题、坐标轴、配色和合并关系；后续复用面板及历史数据。资源统计放在 `resources/*`。

### 指标口径与面板

预训练保存 AE／LM 的实际选样数量，并记录 `loss`、`lm_fraction`、分项 NLL 和样本数。后训练 train 面板如下，均使用单 Y 轴；表中指标省略 `train/` 前缀。

| 图表 | 指标与展示 |
|---|---|
| 联合、QA、策略损失 | `loss`、`qa_nll`、`policy_loss` 分别展示 |
| 奖励 `reward` | 随机路径 `reward` 与多样本 `reward_std`；SCST 加入 `baseline_reward`，图例为 `greedy_reward` |
| 零优势比例 | `zero_advantage_rate`：实际优势全部为零的文档比例 |
| 记忆用量 `memory_usage` | 平均块数 `blocks`，图例 `memory_blocks`，轴名 `memory_blocks = memory_slots / K` |
| 压缩程度 `compression` | `compression_ratio`，图例 `compression_rate`，轴名 `compression_rate = segment_tokens / K` |
| 策略熵 | `policy_entropy` |
| 回放概率偏差 | `replay_log_probability_drift`：各随机路径内最大 log-prob 差的平均值 |
| 采样温度 | `policy_temperature`，启用退火时记录 |
| 梯度范数 | `grad_norm`、`compressor_grad_norm`、`boundary_grad_norm` 分别展示，均为统一裁剪前的范数 |
| 学习率 `learning_rate` | `compressor_learning_rate`、`boundary_learning_rate` 共用纵轴 |

后训练 dev 显示四张图：QA NLL、记忆用量、压缩程度、策略熵。记忆与压缩图沿用 train 的图例、轴名和口径。

压缩倍率按全局输入 token 总数除以 memory token 总数计算，等于聚合后的 `segment_tokens / K`。例如 K256、平均段长 1024 对应 4× 压缩。train 的 QA、块数、压缩率和奖励只统计随机路径，greedy 奖励单独记录。

本地 `metrics.jsonl` 还保留辅助统计：train 的 `commit_rate`、`memory_slots`、`segment_tokens`，dev 的 `loss`、`samples`、`commit_rate`、`memory_slots`、`segment_tokens`。`dev/loss` 等于 `dev/qa_nll`，云端展示后者。SCST 单样本的 `reward_std` 为零，仅写本地；后训练不单独记录 `train/samples`。

### 独立评估任务

使用任务脚本独立评估后训练 checkpoint。各任务先将完整正文压缩为 memory，再由 Reader 回答。Test 同文档、LoCoMo 同对话的全部 QA 共用一次正文压缩得到的记忆，各题独立生成；RULER 按题目处理。评估计划、日志、明细和汇总保存在本地。

| 任务与脚本 | 数据入口 | 范围与评分 |
|---|---|---|
| Test：`eval_test.sh` | 数据目录下 `test.jsonl` | 完整正文 QA；NLL、EM、F1 |
| LoCoMo：`eval_locomo.sh` | 数据目录下 `locomo10.json` | 类别 1–4，分离对话与问题，复用公共模块的官方评分逻辑 |
| RULER：`eval_ruler.sh` | `<长度>/<子任务>/test.jsonl` | 全部 13 项任务，按模板拆分正文、指令和查询，复用公共模块的官方评分逻辑 |

Test 的 `qa_nll` 先按每题目标 tokens 平均，再按题目和文档等权平均；`token_nll` 按全部目标 tokens 加权，目标包含模板结束符。EM／F1 按题平均。`--max-test-samples` 限制文档数，`--max-new-tokens` 控制每题生成上限。

### 启动与续评

从 `train/summary.json` 取得最终 checkpoint 名称，填入下面的 `global_step_<N>`：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/eval_test.sh \
  --eval-run artifacts/v3-1/policy-post_k256_20261112 \
  --eval-id standard_20261112 \
  --eval-data data/delta-mem-qasper \
  --max-new-tokens 128 \
  --checkpoint 'global_step_<N>' \
  --batch-size-per-gpu 8
```

新建评估必填 `--eval-run` 和 `--eval-data`；前者指向运行根目录。`--eval-id` 缺省生成 `YYYYMMDD-HHMMSS`。`--checkpoint` 接受一个或多个 `global_step_<N>` 名称，省略时选择当时全部完整 checkpoint；多个 checkpoint 按真实 step 串行执行。

每卡批量上限由 `--batch-size-per-gpu` 控制，默认 1；Test 按文档、LoCoMo 按对话、RULER 按题目组批。LoCoMo 同一对话的未完成题目分配给同一 GPU，逐题保存与计数；续评时重新压缩尚有未完成题目的对话，只生成缺失答案，官方评分仍按题目汇总，记忆用量按对话平均。

RULER 通过 `--lengths` 选择长度，默认 `8192 16384 32768`；各长度单独组批，可用 `--batch-size-by-length 8192=8 16384=4 32768=2` 覆盖指定长度的批量，其余沿用通用值。GPU 数与批量属于本次执行设置，可在续评时调整。

续评使用原任务脚本，并传入 `--resume`、原 `--eval-run` 和 `--eval-id`；原任务重定向过输出目录时，同时传入原 `--output-root`。

| 续评内容 | 行为 |
|---|---|
| checkpoint 范围 | 省略 `--checkpoint` 时使用计划累计列表；传入时只调度指定项，并将新项加入累计列表 |
| 数据与协议 | 沿用原数据、预算和 RULER 长度；对应 CLI 设置忽略并提示 |
| 执行进度 | 复用已提交记录，将未完成样本按当前 GPU 数与批量重新分配 |
| 修改协议 | 使用新的 eval-id，或以 `--overwrite` 重建当前任务的计划与全部 checkpoint 结果 |

### 终端进度

主进程按 checkpoint 统一显示总体完成数（历史缓存 + 本轮完成）、百分比、耗时和各 GPU 的进度。每卡显示本轮分片的完成数、阶段、当前批量，RULER 另显示长度；速率和剩余时间按本轮已提交样本估算。Test 在一篇文档的全部 QA 完成后计数，LoCoMo 与 RULER 按题目计数。

交互终端每 5 秒原地刷新，重定向输出时每 30 秒记录一次快照，阶段变化和失败立即显示。逐样本明细与 worker 的完整输出保存在 `eval.log`，带 checkpoint step、GPU 和 rank 前缀。

### 本地看板

```bash
bash src/latent_working_memory/v3_1_1/scripts/dashboard.sh \
  --dashboard-run-dir artifacts/v3-1/policy-post_k256_20261112 \
  --dashboard-eval-id standard_20261112
```

看板展示独立评估，按 Test、RULER、LoCoMo 分页。页面上方是“多 checkpoint 比较”：展示全部已完成 checkpoint 的指标表、按实际 optimizer step 排列的变化曲线，以及折叠的资源测量与输出诊断。下方是“单 checkpoint 详情”：“明细 checkpoint”默认选择最新完成的一项，选择器下方依次展示折叠的评估配置、RULER 单步明细、压缩分析与问答。只有一个 checkpoint 时，上方展示其评估概览。Benchmark 支持接入本地基线，按实际输入、参考答案、基座和评估协议对齐。训练与 dev 指标在 SwanLab 查看。

压缩区按篇和记忆块分别展示平均、最低、最高压缩倍率，另列每篇平均记忆块数及单块完成整篇压缩的比例。输入长度 N 取实际 Writer 输入 token 数（含模板前缀），B 为记忆块数，K 为每块槽位数：

| 统计单位 | 压缩倍率 | 平均方式 |
|---|---|---|
| 整篇文本 | `N / (B × K)` | 每篇等权；LoCoMo 每段对话只计一次 |
| 单个记忆块 | `block_length / K` | 每块等权，包含末尾强制提交块 |

整篇倍率分布、单块倍率分布和“输入长度—记忆块数”散点图使用当前 checkpoint；RULER 可按长度和子任务筛选。逐文本列表支持排序，默认选中首行，表格高亮与下方的压缩分段条、块信息及 QA 始终对应同一篇文本。分段条内置范围手柄，支持缩放、拖动平移和悬停查看块信息。图表按需生成，不保存图片；首次读取当前 checkpoint 时仅缓存压缩统计和记录位置，完整回答按所选文本读取。

批次耗时按参与样本分摊，decode 耗时按当轮活跃样本分摊，汇总只累计一次批次 GPU 时间；峰值显存为 worker／batch 共享观测。

## 实现细节

### 批量计算与梯度归约

预训练在一次更新的本卡样本内按 Writer 输入长度降序排列，再切 microbatch 并按 AE／LM 分组，减少 padding。各组损失求和反传，全部 microbatch 完成后按实际全局样本数归约。该排序只作用于当前更新的样本集合。

后训练先固定参数、无梯度地采样和评分，再逐轮批量回放各文档的随机路径。Writer 按活跃文档批量执行边界判断与压缩；Reader 每轮从各文档取一道 QA，将梯度反传至独立 memory 叶子后释放 decoder 图。全部题目的 memory 梯度按题数与 G 归一，再与策略项一起回传 Writer，随后释放本轮路径图。

Writer 与 Reader 均使用左侧 padding、attention mask 和各样本真实位置，损失只统计有效 tokens。分布式归约使用实际全局样本数；尾批中的空 rank 也参与 collective。多卡 CUDA 进程组由公共 runtime 中的 verl 初始化，训练循环、采样回放和 optimizer 更新由本模块执行。

### KV 复用与重算

采样、dev 和独立评估复用原文增量 KV，并保留已提交 memory 的前缀 KV。提交时只压缩 a，排除用于判断的新单元 b；随后续算新 memory 与 b。

固定动作回放中，同一提交段内的判断点共享因果前向，分别读取对应单元末位置。压缩按提交段执行 activation checkpoint，较长可见前缀的边界判断按段重算，答案输出头按块重算。

同一参数下的采样与回放概率比理论值为 1；两种计算路径的浮点差异通过 `replay_log_probability_drift` 记录。

### 跨段反传

`training.backward.bptt_steps` 控制后训练回放的梯度跨度：

| 值 | 梯度传播 |
|---|---|
| `null` | 完整跨段反传 |
| `1` | 每次写入后截断供后续写入与判断使用的历史梯度 |
| 正整数 H | 每 H 次写入为一组，组内完整反传，组间将历史作为常量 |

计数按每条路径的实际写入次数进行，包含末尾强制提交。例如 `2` 将第 1～2 次写入分为一组，第 3～4 次分为下一组。

截断保留历史 memory 的前向数值及 Reader 到每个原始记忆块的直接 QA 梯度，去掉跨组的后续压缩、边界判断到早期 memory 的间接梯度。该设置只用于后训练的带梯度回放；梯度范数裁剪由 `training.optimizer.gradient_clip` 在更新前执行。

## 代码与产物

### 模块分工

执行组织沿用 v4.1 的配置、计划、训练恢复与独立评估方式，方法计算位于本模块：

```text
v3_1_1/
  config.py                 模型、计算与训练配置
  cli.py / io.py            共用参数转换与 JSON 原子写入
  execution.py              运行占用锁、GPU 选择与子进程管理
  preparation/
    pretrain.py / post.py   预训练裁剪与后训练 QA 选择
    length.py               正文长度分档与统计
  data.py                   AE／LM 样本与完整文档 QA
  modeling/                 压缩器、边界模块、流式状态与组批
  rollout.py                因果分段、采样与回放
  training/
    cli.py / plans.py       训练参数与计划
    run.py / train.py       启动调度与训练执行
    engine.py               预算、游标、梯度更新与 dev
    rl.py                   QA／GRPO／SCST 联合目标与反传
    checkpoint.py           checkpoint 保存与恢复
    tracking.py             SwanLab 指标与面板
  evaluation/
    test.py / locomo.py / ruler.py   任务数据适配、协议、组批与评分
    plans.py                评估计划与累计 checkpoint 范围
    run.py / evaluate.py    checkpoint 调度与单 checkpoint 执行
    worker.py               单卡进程入口
    inference.py            共用正文压缩、Reader 生成与 Test NLL
    records.py / shards.py  结果身份、进度、样本分工与提交
    diagnostics.py          输出诊断
  dashboard.py              本地看板
  dashboard_compression.py  压缩统计、记录索引与分段图
  scripts/                  数据准备、训练、评估和看板入口
```

### 运行目录

下列路径相对于 `<output_root>/<run_id>/`：

| 路径 | 内容 |
|---|---|
| `plan/config.json` | `stage`、`seed`、基座、模型与计算设置 |
| `plan/train.json` | 本阶段训练配置和执行安排；预训练含 `objective`，后训练含 `rl` |
| `train/config.json` | 实际训练配置，含解析后的总步数 |
| `train/data.json` | 选样统计及数据、tokenizer、模型和预训练来源绑定信息 |
| `train/metrics.jsonl` | 逐 optimizer step 的训练、dev 与资源指标 |
| `train/summary.json` | 已完成步数与最终 checkpoint 路径 |
| `train/checkpoints/global_step_<N>/state.pt` | 可恢复的训练状态 |
| `plan/eval-<eval-id>/<task>.json` | 独立评估协议及累计 `checkpoints` 列表 |
| `eval/<eval-id>/<task>/global_step_<N>/` | 对应 checkpoint 的评估进度、明细和汇总 |

Checkpoint 固定包含 `config`、`model`、`optimizer`、`scheduler`、`cursor`、`rng`、`boundary_rng`、`binding`。其中 `model` 包含 `memory_embeddings`、`adapter`、`boundary`，预训练的 `boundary` 为 `null`；`cursor.step` 表示已完成的 optimizer steps，随机状态按 rank 保存。恢复核对原配置、数据身份与游标。

评估计划绑定基座、模型结构（含 U）、LoRA、计算设置和 seed，并保存固定数据与协议。任务 `status.json` 的 `selected_checkpoints` 记录本次执行范围，与计划的累计列表分别保存。`progress/worker_<rank>.jsonl` 持续保存已完成样本，flush 后可续评；同目录的 `worker_<rank>.json` 是阶段、批量和长度的展示快照，每次续评重建。完整 details 提交后生成最终 summary。
