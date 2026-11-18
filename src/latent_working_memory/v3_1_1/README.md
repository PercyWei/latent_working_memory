# 20261102_压缩预训练与 QA 后训练

创建时间：20261102 16:06:43 Asia/Shanghai

最后修订时间：20261118 21:47:19 Asia/Shanghai

v3.1.1 的 `stage=pretrain` 使用文本重建／续写（AE／LM）；`stage=post` 使用问答（QA），由 `training.update_mode` 选择仅更新压缩器（`compressor_only`）、仅更新边界（`boundary_only`）或联合更新（`joint`）。每次写入将一段正文压缩为固定 K 个记忆槽位（memory slots），已提交的记忆块保持不变；读取器（Reader）根据这些记忆及未压缩的 pending 和 recent 尾段回答问题。

本文说明方法、配置和执行协议；完整 CLI 参数与各任务命令见[运行指南](scripts/README.md)。

[方法概览](#方法概览) · [数据准备与课程](#数据准备与课程) · [配置与运行](#配置与运行) · [验证、独立评估与记录](#验证独立评估与记录) · [实现细节](#实现细节) · [代码与产物](#代码与产物)

## 方法概览

### 记忆与边界

| 符号 | 配置 | 含义 |
|---|---|---|
| K | `model.structure.memory_slots` | 每次提交生成的记忆槽位数 |
| L | `model.structure.pending_stride` | 每次读入的 token 步长，决定边界判断间隔 |
| H | `model.structure.pending_size` | 允许压缩的最短 pending 长度，至少为 K |
| R | `model.structure.recent_size` | 方法默认保留的尾部 token 数 |
| G | GRPO 的 `training.objective.group_size`；SCST 的 `training.objective.num_samples` | 每篇文档的随机分段路径数 |

`joint` 和 `boundary_only` 使用 `compression.mode=policy`。先预留正文末尾固定 recent，其余前缀按 L 分成处理单元。在线状态由历史 memory、待提交原文 a 和新到达单元 b 组成。边界模块读取这些已到达的内容，选择继续积累 ab，或将 a 压缩为 K 个 slots，再以 b 开始下一块。a 小于 H 时强制继续；达到 H 后允许策略判断，不限制最大段长。最后一个不足 L 的单元也按实际长度读入，固定 recent 不参与判断。

前缀读取完毕后，若剩余 pending 长度至少为 H，则一次压缩为 K 个 slots；不足 H 则保留原文。收尾由规则决定，不计策略动作。Reader 输入顺序为历史 memory、未压缩 pending、固定 recent、问题提示。`recent_size=0` 只表示不预留 recent，最终仍可能保留短 pending。

预训练的默认 recent 长度为 0，后训练取 `model.structure.recent_size`。multi／policy 的 `training.compression.recent_size` 只覆盖训练，`null` 沿用默认长度，0 不预留，正整数指定训练长度。训练中的 dev 和独立评估均使用默认长度。single 的 `compression` 仅含 `mode`，在训练和验证中都强制 recent 为 0。

问题和任务指令交给 Reader，正文分段独立于 QA。Reader 在正文压缩完成后回答，生成期间保持 memory 不变。Writer 与 Reader 的模块及训练状态独立，共享冻结的基座参数；LoRA 仅作用于 Writer。

### 单次压缩预训练

`stage=pretrain` 使用 `training.update_mode=compressor_only`，目标配置为 `training.objective={"type":"ae_lm","strategy":"sft","lm_ratio":0.5,"lm_target_tokens":512}`。本节的单块预设使用 `training.compression={"mode":"single"}`，将完整压缩输入一次写入 K 个 slots，不保留 recent；预训练也支持后文所述的 multi 分段规则。

每条完整 `text` 是一个样本，按 seed 和文本内容固定选择 AE 或 LM，跨 epoch 保持一致。

| 目标 | 压缩输入 | 预测目标 |
|---|---|---|
| AE | 完整文本 | 重建完整文本 |
| LM | 去掉末尾目标 tokens 后的前缀 | 文本末尾的 `lm_target_tokens` 个 tokens |

预设中 `training.objective.lm_ratio=0.5`、`lm_target_tokens=512`。LM 前缀长度不大于 K 时改用 AE；两个目标均追加 EOS 结束标记。每个样本按目标 tokens 平均负对数似然（NLL），再对样本等权平均。

预训练只更新 Writer LoRA 和可学习 memory embeddings。冻结的 Reader 保留损失到 memory 的梯度，边界模块在后训练阶段训练。

### 仅压缩器 QA 后训练

`stage=post`、`training.update_mode=compressor_only` 使用完整正文及 QA，只更新 Writer LoRA 和 memory embeddings。目标配置为 `training.objective={"type":"qa","strategy":"sft"}`，不配置 `backward`，`model.structure.pending_stride`、`pending_size` 与 `boundary_hidden_size` 均为 `null`。多块压缩配置为：

```json
{"mode":"multi","ratios":[2,4,8,16],"recent_size":null}
```

预设的 `model.structure.recent_size=256`，训练覆盖值为 `null`。先固定保留正文末尾实际生效的 recent tokens，其余前缀全部压缩；生效值为 0 时不保留原文。最小压缩段长由 `min(ratios) × K` 确定，示例中为 2K。每一步从 `ratios` 中筛选能放入当前剩余前缀、且压缩后剩余长度为 0 或至少最小段长的倍率。合法候选非空时等概率采样；只有候选为空，才将全部剩余前缀一次压成 K 个 slots 并结束。收尾倍率不加入常规采样集合。

例如还剩 5K 时，4× 会遗留不足 2K 的尾块，只有 2× 合法；先压缩 2K，再将余下 3K 一次收尾。每篇每次训练采样一条路径，各题共享全部记忆和固定 recent。

训练目标仅为原始 QA NLL，先对每题答案 tokens 平均，再对题目、文档平均；不含记忆成本。Reader 冻结并保留到 memory 的梯度，不创建或训练边界模块。`compression.mode=single` 时整篇写入一个记忆块，不保留 recent。

配置的 `init_checkpoint=null` 表示从基座初始化 Writer LoRA 和 memory embeddings；也可加载 AE／LM 预训练或 `compressor_only` checkpoint 中的压缩器。阶段 A 按 `epochs=N` 训练压缩器后，可用该 checkpoint 新建阶段 B 的 `boundary_only` run。阶段切换只加载 Writer 和 memory，新建边界模块、optimizer 和 scheduler；只有 `--resume` 恢复完整运行状态。两段分别设置轮数，不自动判断收敛或交替切换。

### QA 与边界后训练

本节适用于 `joint` 和 `boundary_only`，二者均使用边界策略；`training.objective.type=qa`，`strategy` 为 `grpo` 或 `scst`，其余奖励与策略损失参数放在同一 `objective` 分组中。

一篇完整文档及其固定 QA 集合是一个训练单位。每条随机路径处理整篇正文，各题共享该路径的记忆。QA NLL 先在每题的答案 tokens 内平均，再依次对题目、随机路径和文档等权平均。

每条路径 g 的奖励及联合目标为：

```text
prefix_tokens = input_tokens - recent_tokens
prefix_retention_ratio_g = (K × blocks_g + pending_tokens_g) / prefix_tokens
memory_cost_g = prefix_retention_ratio_g ** memory_cost_power
reward_g = -stopgrad(QA_NLL_g) - memory_cost_weight × memory_cost_g
joint:         loss = mean(QA_NLL_g) + policy_weight × policy_loss
boundary_only: loss = policy_weight × policy_loss
```

`input_tokens` 是预留 recent 之前完整 `input_ids` 的长度，含上下文提示前缀；QA、padding 和追加的 memory tokens 不计入。令 D 为该长度，R 为实际保留的 recent 长度，则记忆成本只衡量 D−R 个前缀 tokens 的保留比例：`c=((K×B+P)/(D−R))^p`，其中 B 为记忆块数，P 为未压缩 pending 长度。固定 recent 不参与成本的分子和分母，但仍供 Reader 回答。`memory_cost_power`（p）为正数，默认 1.0；`memory_cost_weight`（α）为非负数，默认 0.1。p=1 且 pending 长度相同时，同一文档每增加一块的惩罚为 `α × K / (D−R)`。

完整保留比例 `retention_ratio=(K×B+P+R)/D` 单独统计，不用于记忆成本。独立评估若正文全部属于 recent，即 D=R，则没有待压缩前缀，`prefix_retention_ratio` 和 `memory_cost` 均记为 0，完整保留比例为 1；训练长度筛选保证 D−R 为正。

`training.update_mode=joint` 更新 Writer LoRA、memory embeddings 和边界模块；`boundary_only` 冻结 Writer、memory embeddings 和 Reader，优化器仅包含边界模块。冻结后的 QA NLL 只参与奖励结算，不执行 QA 反传。

`training.objective.reward_mode=terminal` 在终局结算上述奖励；`retention_delta` 在完整轨迹结束后，为初次读入、每次读入与判断、强制收尾分别结算占用差分：

```text
prefix_tokens = D - R
m_t = 已生成的 memory slots 数 + 当前 pending tokens 数
r_t = -α × [(m_t/prefix_tokens)^p - (m_{t-1}/prefix_tokens)^p]
最后一个事件额外加入 -QA_NLL
```

从 `m_0=0` 开始，首次只计入新读入的 pending，终局 `m_T=memory slots + 未压缩 pending tokens`，各事件均不计 recent。继续动作增加待保留原文，获得负成本奖励；压缩判断同时读入新单元，只有净占用减少时才获得返还。强制收尾也结算差分，并通过后续回报影响先前自主动作，但自身没有 log-prob 或策略损失。各事件奖励相加严格等于 `-QA_NLL-α×(m_T/(D−R))^p`，不再重复扣一次成本。D−R 只在轨迹完成后用于奖励归一化，策略观察和动作约束不读取该长度。

`training.objective.advantage_mode=trajectory` 为路径 g 的所有自主决策分配同一优势。`reward_to_go` 为每个判断位置 t 计算优势，其中 `U[g,t]` 是路径 g 从当前位置的事件到强制收尾及 QA 评分的奖励总和，不使用折扣。

| 策略 | 每篇随机路径数 | `trajectory` 优势 | `reward_to_go` 优势 |
|---|---|---|---|
| `grpo` | `group_size >= 2` | `(reward_g - mean(reward)) / (std(reward) + advantage_epsilon)`，使用总体标准差 | `U[g,t]` 减去同篇其余随机路径在 t 的平均回报 |
| `scst` | `num_samples >= 1` | `reward_g - greedy_reward` | `U[g,t] - U[greedy,t]` |

SCST 每篇额外从文首运行一条 greedy 路径，供所有随机路径、所有判断位置共用。它与随机路径共用参数、输入和奖励公式，只用于计算基线。greedy 评分独立于边界采样的随机数，同篇动作序列相同的路径共用 QA 评分。`advantage_epsilon` 仅用于 GRPO 的 `trajectory` 优势。

`reward_to_go` 按同一正文读取位置对齐各路径，允许各自的 memory 和 pending 不同。每条路径都保留强制等待事件的回报，因此即使 greedy 在 t 强制等待，也能为该位置提供基线。只有随机路径的自主决策进入策略损失。SCST 配合 `terminal` 时，各位置优势与 `trajectory` 相同。配合 `retention_delta` 时，记忆成本只计入从 t 开始的净变化，各位置优势可以不同。

两种优势模式均使用 clipped policy loss，默认裁剪范围 0.2。`trajectory` 对路径中的自主决策平均，`reward_to_go` 对自主决策求和，再对路径和文档平均。因此即使各位置的优势相同，单条路径的损失尺度也可能不同。奖励、优势及旧动作概率均停止梯度。每次 optimizer update 重新采样，全部 microbatch 完成后统一更新参数。

默认 α=0.1 来自 `policy-post_k256_20261113` 的 step 69 checkpoint：对 118 篇 dev 文档的固定 1／2／4／8 块候选，按参考答案逐文档选择目标最小项。该冻结模型校准保留了无惩罚候选最优 NLL 收益的 90.72%，相对单块的额外保留比例减少 48.52%，平均块数从 2.551 降至 1.822，其中 55 篇选择多块。记录位于 `artifacts/v3-1/policy-post_k256_20261113/eval/boundary-cost_20261116/alpha-calibration.json`。

### 温度退火

`training.objective.policy_transition` 控制随机策略向确定性策略的过渡，GRPO 和 SCST 均可使用。`null` 表示固定 T=1；线性退火配置如下：

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

两阶段均读取准备后的 `train.jsonl`、`dev.jsonl`。AE／LM 预训练和 QA 训练的数据行分别为：

```json
{"text":"完整预训练文本"}
```

```json
{"text":"完整正文","qas":[{"question":"问题","answer":"答案"}]}
```

QA 后训练每题必填 `question`、`answer`，可成对附带来源字段 `segment_id`、`segment_char_span`，例如 `"segment_id":"seg0","segment_char_span":[0,4]`。区间是相对当前正文的左闭右开字符下标，用于来源与位置分析，并随逐题结果展示。QASPER 的 question/answer 数据同样可读。

数据准备和 train/dev 长度筛选均使用 tokenizer 对完整正文计算的 token 数，不添加 special tokens、系统提示或 chat 前缀。AE／LM 与 `compressor_only` 使用固定分段规则：single 要求正文长度严格大于 K；multi 要求至少 `recent_size + min(ratios) × K`，保证固定保留 recent 后仍有一块合法压缩输入。AE／LM 的 LM 前缀也须满足对应下限，不足时改用 AE。policy 要求至少 `recent_size + pending_size`。train 下限采用训练实际 recent，dev 下限采用模型默认 recent。

`training.data.max_train_data_tokens` 只限制 train，超限样本整条排除。筛选与限量在内存中完成，独立 test 读取完整正文。分段和成本的 D 则采用实际编码后的完整正文长度，包含既有上下文前缀。

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
| `model.structure` | 预训练仅 `memory_slots` 为正整数，其余四项均为 `null`。后训练的 `recent_size` 为非负整数。`pending_stride`、`pending_size` 和 `boundary_hidden_size` 在 policy 模式下为正整数，其余模式均为 `null`。 |
| `model.lora` | `lora_rank`、`lora_alpha`、`lora_target_modules` |
| `compute` | 加载精度、attention、混合精度、层内重算、CPU 暂存 |
| `training.data` | 数据目录、长度和数量筛选、课程开关 |
| `training.loop` | 轮数或更新数、课程参数、batch、验证与保存频率、checkpoint 保留数 |
| `training.compression` | single 仅含 `mode`；multi 含 `mode`、`ratios`、训练 recent 覆盖值；policy 含 `mode`、训练 recent 覆盖值 |
| `training.objective` | 所有模式均含 `type`（`ae_lm/qa`）、`strategy`（`sft/grpo/scst`），以及该目标适用的参数 |
| `training.update_mode` | 参数更新范围：`compressor_only/boundary_only/joint`；预训练固定为 `compressor_only` |
| `training.dev` | 固定验证倍率 `fixed_ratios`，QA 生成上限 `max_new_tokens` |
| `training.backward` | 边界策略专用：跨段反传长度 `bptt_steps` |
| `training.optimizer` | 压缩器学习率、AdamW 参数、梯度范数裁剪；边界策略模式另含边界学习率 |
| `training.scheduler` | 学习率 warmup 与余弦衰减比例 |
| `training.run` | `output_root`、`run_id`；后训练另含 `init_checkpoint` |
| `training.swanlab` | `tracking`、`project`、`group`、`tags` |

multi／policy 的训练 recent 覆盖值为 `null` 时沿用模型默认值，0 或正整数只改变训练；dev 和独立评估始终采用模型默认值。single 强制 recent 为 0。

支持以下组合，`type` 指定监督任务，`strategy` 指定训练方法，`update_mode` 指定可更新的参数：

| `stage` | `objective.type` | `objective.strategy` | `update_mode` |
|---|---|---|---|
| `pretrain` | `ae_lm` | `sft` | `compressor_only` |
| `post` | `qa` | `sft` | `compressor_only` |
| `post` | `qa` | `grpo` 或 `scst` | `joint` 或 `boundary_only` |

AE／LM 的 `objective` 还包含 `lm_ratio` 和 `lm_target_tokens`；QA SFT 只需 `type` 和 `strategy`。策略训练的 `objective` 另含路径数、`reward_mode`、`advantage_mode`、`policy_transition`、成本与损失参数，`advantage_epsilon` 仅用于 GRPO 的 trajectory 优势。`joint` 在策略损失之外保留 QA NLL 的直接反传；`boundary_only` 仅通过奖励使用 QA NLL。

模型结构、目标、优化器及调度参数通过 JSON 设置。新训练的显式 CLI 参数覆盖 JSON，省略时沿用配置。覆盖后的配置保存到计划；解析训练预算后的实际配置写入训练目录、checkpoint 和启用时的 SwanLab config。两阶段均保存完整 `model`，`training` 只保存本阶段适用的字段。源 JSON 保持原样，实际 `training.run.output_dir` 由启动流程生成。

### 配置预设

| 配置 | 用途 |
|---|---|
| [pretrain_default.json](../../../configs/v3_1_1/pretrain_default.json) | 通用 K256 AE／LM 预训练 |
| [post_default.json](../../../configs/v3_1_1/post_default.json) | 通用 K256 GRPO 后训练，4 路采样、固定 T=1 |
| [pretrain_k256.json](../../../configs/v3_1_1/pretrain_k256.json) | FineWeb 5k 预训练与服务器计算设置 |
| [post_k256.json](../../../configs/v3_1_1/post_k256.json) | FactQA 长度课程与截断反传，GRPO 4 路采样、固定 T=1 |
| [post_k256_temp.json](../../../configs/v3_1_1/post_k256_temp.json) | 同套后训练数据与执行设置，GRPO 4 路采样、线性温度退火 |
| [post_k256_scst_temp.json](../../../configs/v3_1_1/post_k256_scst_temp.json) | 同套后训练数据与执行设置，SCST 单随机路径加 greedy 基线、线性温度退火 |
| [post_k256_compressor_only.json](../../../configs/v3_1_1/post_k256_compressor_only.json) | FactQA 仅压缩器后训练，从合法倍率采样，固定保留 256 tokens recent |
| [post_k256_boundary_only.json](../../../configs/v3_1_1/post_k256_boundary_only.json) | 冻结压缩器、仅训练边界，4 路采样、占用差分奖励和 reward-to-go |

两份 `*_default.json` 的主要设置如下：

| 设置 | 预训练 | 后训练 |
|---|---|---|
| K／pending 步长／最短 pending／recent／边界维度 | `256 / null / null / 0 / null` | `256 / 256 / 256 / 0 / 128` |
| Writer LoRA | rank 16，alpha 32，目标 `q_proj/k_proj/v_proj/o_proj` | 相同 |
| 计算 | FP32、SDPA；混合精度、层内重算、CPU 暂存均关闭 | 相同 |
| 目标 | `type=ae_lm`、`strategy=sft`，`lm_ratio=0.5`、`lm_target_tokens=512` | `type=qa`、`strategy=grpo`，`group_size=4`、`policy_transition=null` |
| 策略参数 | — | 成本权重 0.1，成本幂 1.0，策略权重 1.0，`clip_epsilon=0.2`，`advantage_epsilon=1e-6` |
| 学习率 | 压缩器 `1e-4` | 压缩器 `2e-5`，边界 `1e-4` |
| AdamW／裁剪 | `weight_decay=0.01`，`beta1=0.9`，`beta2=0.999`，`adam_epsilon=1e-8`，梯度范数上限 1.0 | 相同 |
| 调度 | warmup 50 steps，`min_lr_ratio=0.1` | 相同 |
| 跨段反传 | — | `bptt_steps=null`，完整反传 |

K256 实验预设使用 `/root/rivermind-data/models/Qwen3-4B-Instruct-2507`，开启 FA2、BF16 autocast 和层内重算。它们的执行设置为：

| 设置 | K256 单块预训练 | K256 后训练（GRPO／SCST） |
|---|---|---|
| 数据目录 | `data/fineweb_train5k_20261112` | `data/fineweb-factqa_q10_train1k_length3_20261112` |
| 预算 | 普通训练 1 轮 | 3 个长度档位，每档 1 轮，历史回放 20% |
| 每卡 microbatch × 梯度累积 | `16 × 2` | `2 × 8` |
| train 长度上限 | `null` | 12288 tokens |
| warmup／最低学习率比例 | `20 / 0.1` | `5 / 1.0` |
| dev／checkpoint 间隔 | 每 40 steps | 每 20 steps |
| checkpoint 保留上限 | 5 | 6 |
| 跨段反传 | — | `bptt_steps=2` |

仅压缩器 QA 后训练配置 `post_k256_compressor_only.json` 使用 `compressor_learning_rate=1e-4`、`warmup_steps=4`、`min_lr_ratio=0.1`。每卡 `micro_batch_size=4`、`gradient_accumulation_steps=8`，每次更新每卡累计 32 篇文档，全局 batch 为 GPU 进程数乘 32；同篇全部 QA 不额外增加文档 batch。该配置的模型默认 `recent_size=256`，训练覆盖为 `null`，默认训练 1 轮，关闭长度课程。

当前数据含 train 1002 篇／10020 道 QA、dev 118 篇／1180 道 QA，现有长度筛选保留全部样本。按下面的单卡命令运行 1 轮，共 `ceil(1002 / 32)=32` 个 optimizer steps。`dev_every=8`、`save_every=8`：在 step 0、8、16、24、32 验证，在 step 8、16、24、32 保存 checkpoint；每轮末与最终步也执行验证和保存。4 步 warmup 与 8 步间隔分别约占总步数的 1/8、1/4，与单块 157 步训练的 20 步 warmup、40 步间隔比例相近。改变 GPU 数或训练轮数时，应重新换算这些固定步数。

边界策略预设均设置 `pending_stride=256`、`pending_size=256`。联合更新预设的模型默认 recent 为 0，`boundary_only` 预设为 256，训练覆盖均为 `null`。

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

新训练指定 `--config`，训练阶段直接读取 JSON 中的 `stage`。下面先启动 K256 AE／LM 预训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --config configs/v3_1_1/pretrain_k256.json \
  --run-id pretrain_k256_20261112 \
  --epochs 1 \
  --micro-batch-size-per-gpu 16 \
  --gradient-accumulation-steps 2
```

以筛选后 5000 条样本计算，该设置训练 1 轮共 157 个 optimizer steps，在第 40、80、120、157 步验证并保存。最终 checkpoint 记录在 `train/summary.json`。

联合后训练使用新的 run，通过 `--init-checkpoint` 指定压缩器来源：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --config configs/v3_1_1/post_k256.json \
  --run-id policy-post_k256_20261112 \
  --init-checkpoint artifacts/v3-1/pretrain_k256_20261112/train/checkpoints/global_step_157 \
  --micro-batch-size-per-gpu 2 \
  --gradient-accumulation-steps 8
```

初始化来源与目标 run 须使用相同基座、K 和 LoRA 设置（rank、alpha、目标模块）。`joint/boundary_only` 均可从 AE／LM 预训练或 `compressor_only` checkpoint 加载压缩器。启用 GRPO 温度退火时，将 `--config` 改为 `configs/v3_1_1/post_k256_temp.json`；使用 SCST 温度退火时，改为 `configs/v3_1_1/post_k256_scst_temp.json`。每次新训练分别设置 `--run-id`，并通过 `--init-checkpoint` 指定初始化来源。

阶段 A（`compressor_only`）→ 阶段 B（`boundary_only`）使用以下两个后训练配置，分别通过 `--epochs` 设置 N、M 轮；默认均为 1 轮，关闭长度课程。阶段 A 在开始、每 8 步及每轮末／最终步验证，保存间隔为 8 步；阶段 B 在开始及每轮结束验证。阶段 A 默认 `init_checkpoint=null`，从基座初始化；也可通过 `--init-checkpoint` 指定 AE／LM 或 `compressor_only` checkpoint。阶段切换新建 run，只载入压缩器权重，不沿用 optimizer 或 scheduler：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --config configs/v3_1_1/post_k256_compressor_only.json \
  --run-id compressor-post_k256_20261118 --epochs 1

CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --config configs/v3_1_1/post_k256_boundary_only.json \
  --run-id boundary-only_k256_20261118 --epochs 1 \
  --init-checkpoint 'artifacts/v3-1/compressor-post_k256_20261118/train/checkpoints/global_step_<N>'
```

将 `<N>` 替换为阶段 A 的 `train/summary.json` 中的真实 checkpoint step。边界策略允许一直等待到末尾，因此训练前检查去掉固定 recent 后的完整前缀能否放入 Writer 窗口；固定分段模式在实际 Writer／Reader 前向检查长度。

运行根目录为 `<output_root>/<run_id>/`。显式 `run_id` 原样使用；为 `null` 或省略时，AE／LM 预训练生成 `pretrain_k{K}_{YYYYMMDD-HHMM}`，`compressor_only` 生成 `compressor-post_k{K}_{YYYYMMDD-HHMM}`，`joint/boundary_only` 生成 `policy-post_k{K}_{YYYYMMDD-HHMM}`。时间采用 `Asia/Shanghai`，精确到分钟。

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

train 与 dev 共用一个 run，恢复沿用原身份。自动记录阶段、压缩／目标模式及后训练更新方式等 tags；具体超参数记录在 config。

dev 保持模型参数不变，边界策略使用确定性判断。train/dev 曲线横轴为真实 optimizer step，只在实际产生结果时追加标量。首次记录某个指标时注册原生图表并配置标题、坐标轴、配色和合并关系；后续复用面板及历史数据。资源统计放在 `resources/*`。

### 固定倍率与策略验证

`training.dev.fixed_ratios=[2,4,8,16]` 使用同一批 dev；`max_new_tokens=128` 控制 greedy 生成上限。固定倍率验证按压缩模式采用对应尾段协议：

- AE／LM 或 `compressor_only` 的 multi：四个倍率使用相同的 recent 长度，预训练取 0，后训练取 `model.structure.recent_size`。对剩余前缀按当前固定倍率写入；若剩余长度不足一个固定倍率块，或写入一块会遗留不足 `min(ratios) × K` 的非空尾块，就将当前剩余前缀整体收尾。例如前缀 5K、固定倍率 4× 时，一次压缩 5K；最后一块实际倍率可略大于主倍率。single 若指定固定倍率对照，不保留 recent。
- policy 的固定倍率基线：同样保留模型默认 recent；按固定倍率处理其余前缀，写入后若会遗留不足 `pending_size` 的非空尾块，就将当前剩余前缀整体收尾。各固定倍率块长至少为 `pending_size`。基线将 recent 之外的前缀全部压缩。

各条件均记录实际段长和倍率。`fixed_ratios=[]` 不关闭 dev：AE／LM 和 `compressor_only` 的 single 验证单块压缩，multi 为每篇固定种子采样一条路径；`joint/boundary_only` 只验证 greedy 边界策略：模型默认 recent 为 0 时仅评 NLL，正数时额外生成答案并计算 EM／F1。

阶段 A 分别验证四个固定倍率；阶段 B 额外验证 greedy 边界策略。策略与固定倍率基线使用相同 recent，策略收尾可额外保留不足 `pending_size` 的 pending。进入条件验证的 QA 设置均报告 NLL、EM／F1、块数、未压缩 pending、recent、前缀保留比例及完整保留比例；边界策略模式另报告记忆成本及 `QA NLL + α × cost`，其中成本采用前缀保留比例的 p 次幂。NLL 按题目再文档平均，EM／F1 按题目平均。AE／LM 条件只评价对应重建／续写 NLL，不生成 QA 答案。

这些条件在 step 0、配置的验证间隔、每轮末和最终步计算。`boundary_only` 的固定倍率基线在本次进程内复用，联合更新时每次重算；恢复运行后首次验证重新建立基线缓存。所有验证保留训练 RNG，不改变后续采样。逐篇分段、逐题生成和汇总写入 `train/dev/global_step_<step>/rank_<rank>.jsonl` 与 `summary.json`。

SwanLab 使用 `dev/ratio_2/*`、`dev/ratio_4/*`、`dev/ratio_8/*`、`dev/ratio_16/*` 和 `dev/policy/*`，按同一指标合并条件并固定配色。前缀保留比例与完整保留比例分别展示，`memory_cost` 作为本地辅助统计，包含该成本的 `objective` 单独展示。

### 指标口径与面板

AE／LM 预训练保存实际选样数量、`loss`、`lm_fraction` 和分项 NLL；`compressor_only` 记录 `loss`、`qa_nll`、块数和保留比例。边界策略模式的 train 面板如下，均使用单 Y 轴；表中指标省略 `train/` 前缀。

| 图表 | 指标与展示 |
|---|---|
| 训练、QA、策略损失 | `loss`、`qa_nll`、`policy_loss` 分别展示；仅边界模式的 `loss` 不含 QA 项 |
| 奖励 `reward` | 随机路径 `reward` 与多样本 `reward_std`。SCST 的 `baseline_reward` 记录 greedy 路径的整轨总奖励，图例为 `greedy_reward` |
| 零优势比例 | `zero_advantage_rate`：实际优势全部为零的文档比例 |
| 记忆用量 `memory_usage` | 平均块数 `blocks`，图例 `memory_blocks`，轴名 `memory_blocks = memory_slots / K` |
| 压缩程度 `compression` | `compression_ratio`，图例 `compression_rate`，轴名 `compression_rate = segment_tokens / K` |
| 策略熵 | `policy_entropy` |
| 回放概率偏差 | `replay_log_probability_drift`：各随机路径内最大 log-prob 差的平均值 |
| 采样温度 | `policy_temperature`，启用退火时记录 |
| 梯度范数 | `grad_norm`、`compressor_grad_norm`、`boundary_grad_norm` 分别展示，均为统一裁剪前的范数 |
| 学习率 `learning_rate` | `compressor_learning_rate`、`boundary_learning_rate` 共用纵轴 |

policy 未配置固定倍率且模型默认 recent 为 0 时，dev 只执行 greedy 路径的 NLL 评分，记录 QA NLL、记忆用量、压缩程度、策略熵、两种保留比例、记忆成本和 `objective`，不生成答案。其余 policy 设置进入条件验证，按上一节组织 `dev/policy/*` 和固定倍率指标。

`compression_ratio=segment_tokens/K` 表示已压缩段的整体压缩倍率，未压缩 pending 和 recent 不计入分子或分母。`prefix_retention_ratio=(memory slots+pending)/(D−R)` 衡量待压缩前缀的保留比例，`memory_cost=prefix_retention_ratio^p`；`retention_ratio=(memory slots+pending+recent)/D` 独立记录整篇最终保留比例。没有未压缩 pending 时，压缩倍率与前缀保留比例在单篇上互为倒数。train 的 QA、块数、压缩率和奖励只统计随机路径，greedy 奖励单独记录。同时记录 pending 原文峰值和自主动作的平均绝对优势。

边界策略模式的本地 `metrics.jsonl` 还保留辅助统计：train 的 `commit_rate`、`memory_slots`、`segment_tokens`，dev 的 `loss`、`samples`、`commit_rate`、`memory_slots`、`segment_tokens`。`dev/loss` 等于 `dev/qa_nll`，云端展示后者。SCST 单样本的 `reward_std` 为零，仅写本地；边界策略模式不单独记录 `train/samples`。

### 独立评估任务

任务脚本只接受具有边界策略的 `joint/boundary_only` checkpoint；`compressor_only` 当前使用训练内 dev 验证。各任务按 checkpoint 中 `model.structure` 的边界参数和默认 recent 处理完整正文，忽略训练 recent 覆盖值，再由 Reader 根据 memory、未压缩 pending 和 recent 回答。Test 同文档、LoCoMo 同对话的全部 QA 共用一次压缩结果，各题独立生成；RULER 按题目处理。评估计划、日志、明细和汇总保存在本地。

逐篇记录和汇总分别保留 `prefix_retention_ratio`、`memory_cost` 和完整 `retention_ratio`，均按文档等权平均；LoCoMo 每段对话只计一次。同文档各题共用压缩成本，不按 QA 数重复累计。

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

压缩区按篇和记忆块分别展示平均、最低、最高压缩倍率，另列每篇平均记忆块数及只生成一个记忆块的比例。输入长度 N 取实际编码后的完整正文 token 数（含模板前缀），B 为记忆块数，K 为每块槽位数，P 为未压缩 pending 长度，R 为固定 recent 长度：

| 统计单位 | 压缩倍率 | 平均方式 |
|---|---|---|
| 整篇文本 | `N / (B × K + P + R)` | 每篇等权；LoCoMo 每段对话只计一次 |
| 单个记忆块 | `block_length / K` | 每块等权，包含末尾强制提交块 |

整篇倍率分布、单块倍率分布和“输入长度—记忆块数”散点图使用当前 checkpoint；RULER 可按长度和子任务筛选。逐文本列表支持排序，默认选中首行，表格高亮与下方的压缩分段条、块信息及 QA 始终对应同一篇文本。分段条内置范围手柄，支持缩放、拖动平移和悬停查看块信息。图表按需生成，不保存图片；首次读取当前 checkpoint 时仅缓存压缩统计和记录位置，完整回答按所选文本读取。

批次耗时按参与样本分摊，decode 耗时按当轮活跃样本分摊，汇总只累计一次批次 GPU 时间；峰值显存为 worker／batch 共享观测。

## 实现细节

### 批量计算与梯度归约

AE／LM 和 `compressor_only` 在一次更新的本卡样本内按 Writer 输入长度降序排列，再切 microbatch。单块 AE／LM 按目标分组前向；多块模式按写入轮次批量压缩，Reader 先向独立 memory 叶子累积梯度，再回传完整 Writer 图；QA 对同篇各题平均。全部 microbatch 完成后按实际全局样本数归约。该排序只作用于当前更新的样本集合。

联合后训练先固定参数、无梯度地采样和评分，再逐轮批量回放各文档的随机路径。Writer 按活跃文档批量执行边界判断与压缩；Reader 每轮从各文档取一道 QA，将梯度反传至独立 memory 叶子后释放 decoder 图。全部题目的 memory 梯度按题数与 G 归一，再与策略项一起回传 Writer，随后释放本轮路径图。

仅边界模式使用冻结 Writer 的状态回放边界 log-prob，跳过 Reader 反传和 memory 梯度回注；固定 Writer 保持 eval 状态，边界模块保持 train 状态。

Writer 与 Reader 均使用左侧 padding、attention mask 和各样本真实位置，损失只统计有效 tokens。分布式归约使用实际全局样本数；尾批中的空 rank 也参与 collective。多卡 CUDA 进程组由公共 runtime 中的 verl 初始化，训练循环、采样回放和 optimizer 更新由本模块执行。

### KV 复用与重算

采样、dev 和独立评估复用原文增量 KV，并保留已提交 memory 的前缀 KV。提交时只压缩 a，排除用于判断的新单元 b；随后续算新 memory 与 b。

固定动作回放中，同一提交段内的判断点共享因果前向，分别读取对应单元末位置。压缩按提交段执行 activation checkpoint，较长可见前缀的边界判断按段重算，答案输出头按块重算。

同一参数下的采样与回放概率比理论值为 1；两种计算路径的浮点差异通过 `replay_log_probability_drift` 记录。

### 跨段反传

`training.backward.bptt_steps` 控制边界策略回放的梯度跨度：

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
  compression.py            单块、多倍率随机与固定倍率分段
  modeling/                 压缩器、边界模块、流式状态与组批
  rollout.py                因果分段、采样与回放
  training/
    cli.py / plans.py       训练参数与计划
    run.py / train.py       启动调度与训练执行
    engine.py               预算、游标、梯度更新与 dev
    dev.py                  固定倍率／策略验证、逐题生成与冻结基线缓存
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
| `plan/train.json` | 本阶段训练配置和执行安排，均含 `objective`、`update_mode` 和 `compression` |
| `train/config.json` | 实际训练配置，含解析后的总步数 |
| `train/data.json` | 选样统计及数据、tokenizer、模型和初始化来源绑定信息；`compression_windows.train_recent_size/dev_recent_size` 记录实际生效的 recent 长度 |
| `train/metrics.jsonl` | 逐 optimizer step 的训练、dev 与资源指标 |
| `train/summary.json` | 已完成步数与最终 checkpoint 路径 |
| `train/checkpoints/global_step_<N>/state.pt` | 可恢复的训练状态 |
| `plan/eval-<eval-id>/<task>.json` | 独立评估协议及累计 `checkpoints` 列表 |
| `eval/<eval-id>/<task>/global_step_<N>/` | 对应 checkpoint 的评估进度、明细和汇总 |

Checkpoint 固定包含 `config`、`model`、`optimizer`、`scheduler`、`cursor`、`rng`、`boundary_rng`、`binding`。其中 `model` 包含 `memory_embeddings`、`adapter`、`boundary`，AE／LM 和 `compressor_only` 的 `boundary` 为 `null`；`cursor.step` 表示已完成的 optimizer steps，随机状态按 rank 保存。恢复核对原配置、数据身份与游标。

评估计划绑定基座、模型结构（含 pending 步长、最短长度和默认 recent）、LoRA、计算设置和 seed，并保存固定数据与协议。任务 `status.json` 的 `selected_checkpoints` 记录本次执行范围，与计划的累计列表分别保存。`progress/worker_<rank>.jsonl` 持续保存已完成样本，flush 后可续评；同目录的 `worker_<rank>.json` 是阶段、批量和长度的展示快照，每次续评重建。完整 details 提交后生成最终 summary。
