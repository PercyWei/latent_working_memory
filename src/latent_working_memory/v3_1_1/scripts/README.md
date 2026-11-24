# 20261102_v3.1.1 运行指南

创建时间：20261102 17:04:01 Asia/Shanghai

最后修订时间：20261124 09:38:12 Asia/Shanghai

计算协议、数据格式与模块分工见[方法说明](../README.md)，命令行参数集中见[第 5 节](#5-调整参数)。以下命令使用项目根目录 `.venv`；正式实验先检查[K256 预训练配置](../../../../configs/v3_1_1/pretrain_k256.json)和[K256 后训练配置](../../../../configs/v3_1_1/post_k256_grpo.json)中的基座与数据路径。

## 1. 环境与入口

- 数据准备：`prepare_pre.sh` 和 `prepare_post.sh` 分别负责准备预训练数据和后训练数据，`prepare_length.sh` 对数据集进行长度分档，为长度课程学习准备数据。
- 训练：`train.sh` 负责训练流程。
- 评估：`eval_test.sh`、`eval_locomo.sh` 和 `eval_ruler.sh` 分别执行 Test、LoCoMo 和 RULER 评估。

本文示例约定：

| 内容 | 路径 |
|---|---|
| 基座与 tokenizer | `~/models/Qwen3-4B-Instruct-2507` |
| AE / LM 数据来源 | `data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/{train,dev,test}.jsonl` |
| QA 数据来源 | `data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008/{train,dev,test}.jsonl` |
| 产物父目录 | `artifacts/v3-1` |

运行目录与各阶段产物见 [第 7 节](#7-产物目录与记录)。

## 2. 离线数据准备

`prepare_pre.sh` 与 `prepare_post.sh` 均遵守如下规则：

- `--model-path` 必填，使用与训练相同的 tokenizer 计算实际正文 token 数，不额外添加 special tokens、系统提示与 chat 前缀。
- `--target-samples` 默认 `null`，使用全部合条件数据；指定正整数时随机选取对应数量，超过可用数量时报错。
- `--seed` 默认 42，相同输入和参数可复现生成结果。
- `--output-file` 作为数据输出文件一般为 `<split>.jsonl`，构造参数与统计保存为同目录的元信息文件 `preparation-<split>.json`。

### 2.1 预训练数据

```bash
bash src/latent_working_memory/v3_1_1/scripts/prepare_pre.sh \
  --prepare-data data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/train.jsonl \
  --output-file data/fineweb_train5k_20261112/train.jsonl \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --target-samples 5000 \
  --min-tokens null \
  --max-tokens 4096 \
  --length-bins 10 \
  --length-distribution uniform \
  --seed 42
```

- **数据格式**：JSONL 每行仅包含 `{"text":"完整文本"}`。
- **正文与长度**：对 Multisegment 的 `text + continuation` 整体分词，统计实际 token 容量。`--min-tokens`、`--max-tokens` 默认 `null`（采用对应来源边界），也可指定正整数；请求范围与来源长度范围取交集，再按目标 token 长度随机裁出一条连续原文，对最终 `text` 重新分词并统计实际长度。
- **长度分布**：`--length-bins` 默认划分 10 个等宽区间，整数长度不足时减少区间数。`--length-distribution` 的 `uniform`（默认）尽量均衡分配，`random` 逐条等概率选择区间；均在区间内随机选长度，实际分布受来源容量约束。

### 2.2 后训练数据

```bash
bash src/latent_working_memory/v3_1_1/scripts/prepare_post.sh \
  --prepare-data data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008/train.jsonl \
  --output-file data/fineweb-factqa_q10_train1k_20261112/train.jsonl \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --target-samples null \
  --min-tokens null \
  --max-tokens 10000 \
  --qas-per-sample 10 \
  --seed 42
```

- **数据格式**：JSONL 每行仅包含 `{"text":"完整正文","qas":[{"question":"问题","answer":"答案","segment_id":"seg0","segment_char_span":[0,4]}]}`。`segment_char_span` 为来源段在正文中的字符区间，左闭右开。
- **正文筛选**：完整保留 FactQA 正文；`--min-tokens`、`--max-tokens` 按实际正文 token 数过滤整条文本，默认 `null` 取消对应界限。
- **QA 选择**：仅使用 `task` 题，`--qas-per-sample` 默认最多 10 题，题数不足时全保留，`null` 保留全部。按保留题数等分全文，依次选择答案证据最靠近各份中心的题，每题最多一次，最后按证据在正文中的顺序输出。

### 2.3 长度分档

两阶段均可对已准备数据集中的指定 split 执行长度分档。以下以后训练的 train split 为例：

```bash
bash src/latent_working_memory/v3_1_1/scripts/prepare_length.sh \
  --prepare-data data/fineweb-factqa_q10_train1k_20261112/ \
  --output-dir data/fineweb-factqa_q10_train1k_length3_20261112 \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --split train \
  --length-bins 3
```

- **数据格式**：读取来源目录中的 `<split>.jsonl`，根据数据正文长度进行分别保存，保留原数据格式；其余 split 的 JSONL 文件原样复制到输出目录。
- **长度分档**：使用与训练相同的 tokenizer，按完整正文的实际 token 数划分等宽区间，不含系统提示和 chat 前缀。`--length-bins` 默认 4。
- **输出与统计**：输出目录须不存在。分档写入 `<split>/bin_XX.jsonl`，终端与 `preparation-<split>.json` 展示各档边界、样本数及实际长度的最小值、最大值和均值，方便在训练前查看并调整数据。

```text
fineweb-factqa_q10_train1k_length3_20261112/
├── train/
│   ├── bin_00.jsonl
│   ├── bin_01.jsonl
│   └── bin_02.jsonl
├── dev.jsonl
└── preparation-train.json
```

## 3. 训练

新训练指定 `--config`，直接使用 JSON 中的 `stage`。各阶段统一配置 `training.objective.type`、`training.objective.strategy` 和 `training.update_mode`：

| `stage` | `objective.type` | `objective.strategy` | `update_mode` |
|---|---|---|---|
| `pretrain` | `ae_lm` | `sft` | `compressor_only` |
| `post` | `qa` | `sft` | `compressor_only` |
| `post` | `qa` | `grpo` 或 `scst` | `joint` 或 `boundary_only` |
| `post` | `qa` | `ppo` | `boundary_only` |

`type` 指定监督任务，`strategy` 指定训练方法，`update_mode` 指定更新范围。`joint` 同时保留 QA NLL 的直接反传和策略梯度；`boundary_only` 冻结压缩器，QA NLL 只用于奖励。

`training.optimizer` 的学习率按更新范围配置：`compressor_only` 仅含 `compressor_learning_rate`，`boundary_only` 仅含 `boundary_learning_rate`，`joint` 同时包含两者。PPO 的 critic 沿用边界学习率数值。仅 `joint` 策略训练包含 `training.backward`，通过 `bptt_steps` 设置跨段反传；其他模式省略该分组。`training.dev.max_new_tokens` 仅用于 QA 生成验证，AE／LM 不配置。

所有 v3.1.1 预设统一使用 `training.scheduler.warmup_ratio=0.1`，允许范围为 `[0,1)`。实际预热步数为 `ceil(warmup_ratio × 总 optimizer steps)`，必须小于总步数；比例为 0 时不预热。PPO 的总预算包含 `ppo_epochs`，actor／critic 各自按实际更新次数推进同一预热长度的 scheduler。启动日志同时报告预热比例和换算后的步数。

各模式共用以下运行规则：

- **配置覆盖**：显式 CLI 参数覆盖 JSON，具体说明见[第 5 节](#5-调整参数)。
- **计算设置**：FlashAttention2 需 BF16 基座或开启混合精度。恢复与独立评估沿用已保存设置，详细计算方式见[方法说明](../README.md#配置与运行)。
- **训练批量**：`--micro-batch-size-per-gpu` 控制每卡实际批量计算的样本数，各 microbatch 之间累积梯度；全局 batch = GPU 数 × 每卡 microbatch × 梯度累积次数。预训练按样本、后训练按文档计数；下列单卡示例中，AE／LM 与 `compressor_only` 均为 32，联合更新为 16。
- **长度课程**：`post_k256_grpo.json` 及其温度退火变体已开启课程，每档训练 1 轮，并从此前各档尽量均衡地随机无放回补充本档数量的 20%。按档位从短到长推进，每轮完整遍历本档，历史补充数量四舍五入、不足时取全部；每轮重新抽取，混合打乱后统一切 batch，全部档位完成后结束，不额外执行 `--epochs` 全量训练。其他配置可通过 `--length-curriculum` 开启，以 `--curriculum-stage-epochs` 和 `--curriculum-replay-ratio` 调整轮数与比例。关闭课程时，各档合并后按 `--epochs` 全量随机训练；dev 和 test 保持原协议。
- **执行流程**：先保存训练计划，随后直接开始训练；dev 仅在配置的验证间隔、每轮末和最终步执行，checkpoint 按保存设置写入。
- **覆盖训练**：追加 `--overwrite` 会清理本次 `run-id` 对应的整个 run，包括计划、训练和评估结果，再保存新计划重新训练；云端历史保留。

### 3.1 单次压缩预训练

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --config configs/v3_1_1/pretrain_k256.json \
  --run-id pretrain_k256_20261112 \
  --epochs 1 \
  --micro-batch-size-per-gpu 16 \
  --gradient-accumulation-steps 2
```

- **运行目录**：`run_id` 默认为 `pretrain_k{memory_slots}_{YYYYMMDD-HHMM}`，其中 `memory_slots` 为记忆槽位数。
- **当前数据**：K256 配置读取 `data/fineweb_train5k_20261112`，5000 条训练样本按上述批量训练 1 轮，共 157 个 optimizer steps。每 40 步及训练结束时验证和保存，在第 40、80、120、157 步生成 checkpoint，最多保留 5 个。
- **输入与目标**：读取准备后的完整 `text`，先按完整样本的实际 token 数执行长度筛选，再依据 seed 与文本内容标识固定选择 AE 或 LM，跨 epoch 不变。AE 压缩并重建完整样本；LM 以前缀为压缩输入，默认预测样本末尾的 512 tokens，若剩余前缀长度不大于 K 则改为 AE。默认 LM 概率 0.5，目标均追加 EOS。
- **批量计算**：每次更新的本卡样本先按输入长度从长到短排序，再切 microbatch 并分 AE／LM，组内批量压缩和读取；每条样本先对目标 tokens 平均损失，累计后按实际全局样本数归约，padding 不计入目标。
- **可训练参数**：压缩器 LoRA 和 memory embeddings；基座与读取器冻结，读取器保留损失到 memory 的梯度。预训练不更新边界模块。

预训练配置使用 `training.objective={"type":"ae_lm","strategy":"sft","lm_ratio":0.5,"lm_target_tokens":512}` 和 `training.update_mode=compressor_only`。`model.structure` 中仅 `memory_slots` 填正整数，`pending_stride`、`pending_size`、`recent_size` 和 `boundary_hidden_size` 均填 `null`。

`training.compression={"mode":"single"}` 将压缩输入全部写入一块，训练和 dev 的实际 recent 长度均为 0。采用 `mode=multi` 时，可在 `training.compression.recent_size` 指定训练专用的长度，`null` 沿用默认长度 0，dev 也使用 0。

### 3.2 QA 后训练（仅更新压缩器）

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --config configs/v3_1_1/post_k256_conly.json \
  --run-id compressor-post_k256_20261120
```

[post_k256_conly.json](../../../../configs/v3_1_1/post_k256_conly.json) 使用 `stage=post`、`update_mode=compressor_only`，目标配置为 `training.objective={"type":"qa","strategy":"sft"}`，不配置 `backward`，`pending_stride`、`pending_size` 与 `boundary_hidden_size` 均为 `null`。该预设的压缩器学习率为 `1e-4`，`warmup_ratio=0.1`，`min_lr_ratio=0.1`；每卡 `micro_batch_size=4`、`gradient_accumulation_steps=8`，每次更新累计 `4 × 8 = 32` 篇文档，全局 batch 再乘 GPU 进程数，同篇 QA 数不计入文档 batch。默认运行名为 `compressor-post_k{memory_slots}_{YYYYMMDD-HHMM}`。

该模式只更新 Writer LoRA 和 memory embeddings，Reader 冻结并保留输入梯度。预设的 `init_checkpoint` 为 `artifacts/v3-1/pretrain_k256_20261113_copy2/train/checkpoints/global_step_157`，保留 AE／LM 预训练最后第 157 步的压缩器权重，元数据按当前 checkpoint 格式保存。也可通过 `--init-checkpoint` 加载其他 AE／LM 或 `compressor_only` checkpoint，或在配置中设 `init_checkpoint=null` 从基座初始化压缩器；新 run 使用新的 optimizer 和 scheduler。

多块 QA 数据包含 train 1002 篇／10020 道 QA、dev 118 篇／1180 道 QA，当前筛选保留全量；默认训练 5 轮。全局 batch 为 32 时，每轮 32 steps，共 160 steps，预热为 `ceil(0.1 × 160)=16` 步。`dev_every` 和 `save_every` 均为 `after_each_epoch`：在每轮末的 step 32、64、96、128、160 执行 dev 并保存 checkpoint，保留全部 5 份。改变 GPU 数或轮数后，预热随实际总 optimizer steps 自动换算，验证和保存仍在每轮末执行。

多块预设的模型默认 `recent_size=256`，训练使用 `{"mode":"multi","ratios":[2,4,8,16],"recent_size":null}`：先固定保留末尾 256 tokens，再从不会遗留过短尾块的合法倍率中逐段采样，仅在候选为空时将全部剩余前缀一次压缩。完整分段和固定倍率 dev 协议见[仅压缩器 QA 后训练](../README.md#仅压缩器-qa-后训练)与[固定倍率验证](../README.md#固定倍率与策略验证)。

### 3.3 QA 后训练（联合更新或仅边界更新）

`training.objective` 包含 `type=qa`、`strategy=scst/grpo/ppo`，以及 `group_size`、奖励与策略损失参数。SCST／GRPO 的 `training.update_mode` 取 `joint` 或 `boundary_only`，PPO 取 `boundary_only`。`compression={"mode":"policy","recent_size":null}` 沿用模型默认 recent；`pending_stride=256`、`pending_size=256` 位于 `model.structure`。预留固定 recent 后，每个步长检查一次已有 pending，达到最短长度才允许策略压缩；结束时达到阈值强制压缩，否则保留原文。

实际最终记忆成本为 `c=((K×B+P)/(D−R))^p`：D 为完整输入长度，R 为实际 recent 长度，B 为最终记忆块数，P 为最终未压缩 pending 长度。`terminal` 在末尾结算 `Z=-QA_NLL-αc`。

仅 `reward_mode=stepwise` 允许配置 `training.objective.stepwise_cost_basis`，支持 `observed` 和 `full`，省略时默认 `observed`；terminal 和 SFT 配置不能包含该字段。`post_k256_bonly_grpo.json` 使用 terminal 奖励；`post_k256_bonly_ppo.json` 使用 stepwise 奖励，并显式设置 `stepwise_cost_basis=observed`。stepwise 的逐事件奖励为 `-α×(c_t−c_{t-1})`，最终额外加入一次 `-QA_NLL`：

| 成本基准 | 过程成本 | 即时记忆奖励 |
|---|---|---|
| `observed` | `c_t=((K×B_t+P_t)/(D−R))^p`，初始成本 0 | 读取增加占用，压缩按净占用变化返还成本 |
| `full` | `c_t=((K×B_t+P_t+U_t)/(D−R))^p`，初始成本 1 | U 为未读前缀原文长度；两类等待均为 0，自主及强制压缩均按实际节省返还成本 |

recent 不收费，强制收尾不重复扣完整成本。D>R 时，未折扣原始总奖励在 `observed` 下为 Z，在 `full` 下为 Z+α；D=R 时两种基准均只结算 QA 奖励。两种基准共用实际最终成本及保留比例指标，dev／独立评估口径不变。

`group_size` 是每篇正文的随机轨迹数，SCST／PPO 至少为 1，GRPO 至少为 2；SCST 每篇额外计算一条 greedy 轨迹。`reward_mode=terminal` 对所有策略使用有效折扣率 1，覆盖配置中的 `gamma`；`stepwise` 使用配置中的 `gamma`，默认 1.0。优势计算由策略和奖励模式确定，完整公式见[方法说明](../README.md#qa-与边界后训练)。

SCST terminal 调用 verl 原生 ReMax 得到完整得分差，stepwise 使用随机路径与 greedy 路径在同一正文读取位置的折扣后续回报差。GRPO terminal 调用原生 GRPO 标准化最终得分，stepwise 先对同篇采样组的全部真实事件奖励统一标准化，再累计折扣回报；两种模式均使用样本标准差。真实强制事件参与回报和统计，padding 与初始成本占位不参与。只有允许自主选择压缩或继续积累的位置参与 actor 损失。

PPO 使用独立 value head 和 verl 原生 GAE，`gae_lambda` 必须显式设置，两种奖励模式都读取该值。原生 GAE 对完整 rollout 批次的真实事件优势做白化，critic 目标仍使用白化前的优势。预设 [post_k256_bonly_ppo.json](../../../../configs/v3_1_1/post_k256_bonly_ppo.json) 使用 `group_size=1`、`gae_lambda=0.95`、`ppo_epochs=4`、`value_loss_weight=0.5` 和 `critic_warmup=8`；0.95 仅为预设值。Writer 和 memory 冻结，每批 rollout 固定特征、旧概率、旧价值、优势及价值目标。actor／critic 使用独立 optimizer 和梯度裁剪，先完成 critic 的全部内轮，再完成 actor 的全部内轮；value head 与边界沿用同一学习率数值。

三种策略在 terminal 和 stepwise 下都使用 verl vanilla 策略损失：轨迹内对自主动作求和，再按完整 `group_size` 和实际全局文档数平均。零自主动作轨迹贡献 0，仍保留在轨迹和文档的平均分母中。策略项采用原生 dual-clip 系数 3 和 log 概率比 `[-20,20]` 截断；critic 保持每条轨迹的真实事件平均，再对轨迹和文档平均，并采用范围为 0.5 的原生 value clipping。求和归约会影响 actor 损失尺度，`clip_epsilon`、损失权重、学习率与其他超参数沿用配置值。

`training.loop.critic_warmup` 默认 0，直接使用原生 rollout 批次阈值；仅 PPO 配置可以包含该字段。批次从 1 开始，0／1 均从首批更新 actor，11 表示前 10 批只更新 critic。预热期间继续采样新轨迹，使用相同 GAE 目标；该参数与学习率的 `scheduler.warmup_ratio` 独立。

`post_k256_bonly_ppo.json` 的 `critic_warmup=8` 表示前 7 批只更新 critic，第 8 批开始更新 actor；这是该预设的配置，通用默认仍为 0。

PPO 的 `max_steps`、checkpoint 和 SwanLab 横轴使用 critic 实际更新数作为主步数；其他策略使用 actor 实际更新数。完整数据轮数对应的文档批次数乘 `ppo_epochs` 为 PPO 预算，critic 预热占用原预算。两个 optimizer 的学习率 scheduler 分别按自身实际更新次数推进，温度按训练主步数计算；actor 在预热或全局无自主动作时不更新。`rollout_step`、`actor_optimizer_step` 和 `critic_optimizer_step` 分别记录在本地辅助指标中，不能将原生预热的批次阈值当成 optimizer 步数。记录、dev 与保存位于完整 rollout 更新批次结束处，跨过整数间隔阈值时触发，恢复从下一批 rollout 继续。

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --config configs/v3_1_1/post_k256_grpo.json \
  --run-id policy-post_k256_20261112 \
  --init-checkpoint 'artifacts/v3-1/pretrain_k256_20261112/train/checkpoints/global_step_157' \
  --micro-batch-size-per-gpu 2 \
  --gradient-accumulation-steps 8
```

- **运行目录**：`run_id` 默认为 `policy-post_k{memory_slots}_{YYYYMMDD-HHMM}`，其中 `memory_slots` 为记忆槽位数。
- **当前数据**：K256 配置读取 `data/fineweb-factqa_q10_train1k_length3_20261112`，按 3 个长度档位依次训练，dev 读取该目录下的 `dev.jsonl`；课程已在配置中开启，无需额外传参。
- **初始化**：通过 `--init-checkpoint` 指定 AE／LM 预训练或 `compressor_only` checkpoint，只加载 Writer 和 memory embeddings；边界模块、optimizer、scheduler 独立初始化。来源与本次 run 的基座、记忆槽位数、LoRA 设置须一致。
- **训练单位**：每行一篇文档及其全部 QA。一篇文档采样 G 条完整硬分段，每条路径共享 memory 评价同一套 QA；G 不计入全局 batch。
- **一次联合更新**：每卡按 microbatch 组文档，批量采样、评价，再按路径轮次批量回放；每轮从各篇选一道 QA 组批反传并释放 Reader 图，按各篇题数及 G 汇总 memory 梯度后，与策略项一起回传压缩器。各篇优势与损失独立计算，累计全部 microbatch 后按实际全局文档数归约并更新一次。同一提交段的边界判断共享因果前向，压缩排除仅用于判断的下一单元。
- **反传跨度**：在 JSON 中设置 `training.backward.bptt_steps`，支持 `null`（保留完整跨段反传）或正整数。例如设为 `2`，每两次实际写入为一组，组内正常反传，组间截断后续压缩与边界判断使用的历史梯度。全部历史 memory 仍参与前向，所有块保留直接 QA 监督；AE／LM、`compressor_only`、无梯度采样和评估不受影响。
- **归约与尾批**：损失按实际文档数平均，尾批不补齐、不丢弃、不跨 epoch 拼接；空 rank 同样参加 collective。

仅边界更新使用 `post_k256_bonly_grpo.json` 或 `post_k256_bonly_ppo.json`，冻结 Writer、memory embeddings 和 Reader，GRPO 只训练边界，PPO 另训练 value head。两份预设的 `init_checkpoint` 均为 `artifacts/v3-1/compressor-post_k256_20261120/train/checkpoints/global_step_160`，每卡 `micro_batch_size=2`、`gradient_accumulation_steps=2`，训练 1 轮且关闭长度课程。边界学习率为 `1e-4`，`warmup_ratio=0.1`、`min_lr_ratio=1.0`，预热后保持恒定；PPO value head 使用相同学习率设置。以下按四卡运行，全局 batch 为 `4 × 2 × 2=16`：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --config configs/v3_1_1/post_k256_bonly_grpo.json \
  --run-id boundary-only-grpo_k256_20261121
```

使用 PPO 时将配置替换为 `configs/v3_1_1/post_k256_bonly_ppo.json`，并指定独立 `run_id`。初始化来源不同则用 `--init-checkpoint` 显式覆盖；新 run 的边界、value head、optimizer 和 scheduler 独立初始化。

按四卡、全局 batch 16 和当前 1002 篇训练文档计算，1 轮共 63 个 rollout 批次，末批保留 10 篇：GRPO 共 63 次主更新，学习率预热 `ceil(0.1 × 63)=7` 步；PPO 共 `63 × 4=252` 次 critic 主更新，actor／critic 各自学习率预热 `ceil(0.1 × 252)=26` 步。PPO 前 7 批共 28 次更新仅训练 critic，其余 56 批每批均存在自主动作时，actor 共更新 224 次。GRPO 的 `dev_every`／`save_every` 均为 20，在主步数 20、40、60、63 验证并保存；PPO 均为 80，对应主步数 80、160、240、252。两者均对应第 20、40、60、63 个 rollout 批次，最后一批由轮末规则触发一次。`save_total_limit=5` 是保留上限，实际生成的 4 份 checkpoint 全部保留。两份预设均设置 `dev.fixed_ratios=[]`，在当前 `recent_size=256` 下仅验证当前 greedy 边界策略，记录 QA NLL、EM／F1、最终记忆成本和保留比例。数据规模、全局 batch 或 PPO 内轮数改变时，需重新计算预算及验证、保存间隔。

`post_default.json` 的模型默认 recent 为 0，K256 实验预设为 256；训练覆盖均为 `null`。`recent_size=0` 不预留 recent，但仍可能留下不足 `pending_size` 的 pending。

当前阶段 A 预设使用固定倍率 dev；阶段 B 两份预设只使用 greedy 策略 dev，可通过非空 `dev.fixed_ratios` 额外加入固定倍率对照。两段分别启动新 run；也可直接从 AE／LM checkpoint 启动 `boundary_only`。

### 3.4 中断恢复

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --resume \
  --checkpoint 'artifacts/v3-1/pretrain_k256_20261112/train/checkpoints/global_step_<N>'
```

- **恢复起点**：仅接受当前 run 最新的完整 checkpoint。
- **恢复状态**：可训练参数、独立的 actor／critic optimizer 与 scheduler、rollout 批次计数、数据游标、课程阶段与阶段内轮数及各 rank 的随机状态。非 PPO 的 critic 状态为 `null`。
- **参数来源**：仅用 `--checkpoint` 定位原训练计划，恢复全部实验设置；其他已知 CLI 实验设置忽略，启动时集中提示被忽略的选项。
- **一致性要求**：原计划与 checkpoint 保存的配置须一致，GPU 数量和数据须保持一致。修改训练设置应启动新 run，恢复不依赖初始化来源文件。
- **执行计划**：沿用原训练计划；预算已完成时直接返回，不读取或启动评估计划。

## 4. 评估

当前独立评估仅接受具有边界策略的 `joint/boundary_only` checkpoint，模型与计算设置从 checkpoint 读取；`compressor_only` 当前使用训练内 dev。

### 4.1 启动评估

以下以 QASPER test 为例，评估原运行的 step 25、50：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/eval_test.sh \
  --eval-run 'artifacts/v3-1/policy-post_k256_20261104' \
  --eval-id standard_20261104 \
  --eval-data 'data/delta-mem-qasper' \
  --checkpoint global_step_25 global_step_50 \
  --batch-size-per-gpu 8
```

按任务选择脚本和数据目录，并按需设置专属参数：

| 脚本 | `--eval-data` | 专属参数 |
|---|---|---|
| `eval_test.sh` | `data/delta-mem-qasper` | `--max-test-samples` 限制文档数，`--max-new-tokens` 设置每题生成上限 |
| `eval_locomo.sh` | `data/benchmarks/locomo` | 无 |
| `eval_ruler.sh` | `data/benchmarks/ruler-v1/Qwen3-4B-Instruct-2507` | `--lengths` 选择长度，`--batch-size-by-length` 设置各长度的每卡批量上限 |

RULER 默认评估 `8192 16384 32768`，可为不同长度分配批量：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/eval_ruler.sh \
  --eval-run 'artifacts/v3-1/policy-post_k256_20261104' \
  --eval-id standard_20261104 \
  --eval-data 'data/benchmarks/ruler-v1/Qwen3-4B-Instruct-2507' \
  --lengths 8192 16384 32768 \
  --batch-size-by-length 8192=8 16384=4 32768=2
```

- **评估数据格式**：`--eval-data` 指定评估数据根目录：Test 读取 `test.jsonl`，LoCoMo 读取 `locomo10.json`，RULER 按 `<长度>/<子任务>/test.jsonl` 读取各长度档位和子任务的数据。Benchmark 数据须提前准备，数据布局与准备入口见[公共 benchmark 指南](../../benchmarks/README.md#2-数据准备)。
- 三个评估入口都只向 Writer 传入正文。Reader 先读取 system 消息原文与 user 角色开头，再读取记忆块、剩余原文和题目提示。LoCoMo、RULER 沿用官方评分逻辑。
- RULER 以每题的最终问句为界。此前的全部文本（含任务说明和已答示例）按原顺序进入 Writer；最终问句原样交给 Reader。
- **Test 协议**：固定读取数据目录的完整 `test.jsonl`，不套用训练长度筛选，默认每题最多生成 128 tokens；同文档全部 QA 共用记忆。
- **LoCoMo 协议**：同对话的全部题目共用一次正文压缩得到的记忆，各题独立生成并立即保存；官方指标按题目评分，记忆用量按对话平均。续评时只对尚有未完成题目的对话重新压缩，并补齐缺失答案。
- **运行与范围**：`--eval-run` 必填，指向运行根目录。`--checkpoint` 可选，只填一个或多个 `global_step_<N>` 名称，自动查找 `<eval-run>/train/checkpoints/<名称>/state.pt`；省略时选择当时保存的全部完整 checkpoint。
- **评估计划**：新建时须提供 `--eval-data`，默认先将选定范围与评估设置保存到 `<eval-run>/plan/eval-<eval-id>/<task>.json`，再执行评估，结果保存到 `<eval-run>/eval/<eval-id>/<task>/`。评估标识 `--eval-id` 默认 `YYYYMMDD-HHMMSS`，同一标识下各任务独立管理，目标任务计划已存在时直接报错。计划绑定基座、记忆架构、LoRA、计算设置和 seed。
- **并行与批量**：checkpoint 按 step 升序依次评估，可见 GPU 分摊样本，每卡一份模型。`--batch-size-per-gpu` 设置每卡批量上限（默认 1），Test 按文档、LoCoMo 按对话、RULER 按题目计数；LoCoMo 同一对话的剩余题目由同一 GPU 处理。RULER 各长度档位分别组批，可用 `--batch-size-by-length 长度=批量 ...` 覆盖指定档位，其余沿用通用值。
- **覆盖评估**：追加 `--overwrite`，按本次参数重建当前评估任务（`<eval-run>` `<eval-id>` `<task>`）的计划和全部 checkpoint 结果，其他任务、其他 eval-id、其他运行的评估结果保留。目标正在使用时直接报错。

### 4.2 中断恢复或补充

使用已有计划继续评估，也可选择本次范围、补评新 checkpoint。以下假设原 test 计划包含 step 25、50，训练又保存了 step 75，本次只评估 step 50、75：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/eval_test.sh \
  --resume \
  --eval-run 'artifacts/v3-1/policy-post_k256_20261104' \
  --eval-id standard_20261104 \
  --checkpoint global_step_50 global_step_75 \
  --batch-size-per-gpu 4
```

- **恢复定位**：使用原任务脚本，显式提供 `--eval-run` 和 `--eval-id`，对应任务计划必须存在。若原评估通过 `--output-root` 重定向，仍须传入原值以定位计划。
- **本次范围**：省略 `--checkpoint` 时评估计划中的全部 checkpoint；传入名称列表时只调度指定项。完整且身份一致的结果直接复用，未完成结果保留逐文档／题目进度，只补缺失样本；产物身份不一致时报错。
- **计划更新**：新 checkpoint 先校验归属与配置，再并入计划的 `checkpoints` 列表，按 step 排序去重，不缩减旧范围；仅有新增时原子更新计划。上述命令使累计范围变为 25、50、75，任务状态的 `selected_checkpoints` 仅记录本次的 50、75。
- **参数来源**：数据、预算和 RULER 长度沿用原计划；对应 CLI 选项忽略，启动时集中提示。
- **执行资源**：GPU 数量、批大小可调整；未完成样本按本次资源重新分片、组批，已提交记录保留。

### 4.3 评估结果与看板

- **终端进度**：主进程统一显示当前 checkpoint 的总体进度（历史缓存 + 本轮完成）及各卡本轮分片进度，包括阶段、批量、RULER 长度、已提交样本速率和预计剩余时间。Test 按完整文档计数，LoCoMo 与 RULER 按题目计数。交互终端每 5 秒原地刷新，重定向输出时每 30 秒记录快照，阶段变化和失败立即显示。逐样本明细和 worker 完整输出保存在 `eval.log`。

- **产物路径**：下表路径均相对于运行根目录。

| 评估产物 | 内容 |
|---|---|
| `plan/eval-<eval-id>/<task>.json` | 该任务累计的 checkpoint 列表、固定数据与评估设置 |
| `eval/<eval-id>/<task>/global_step_<N>/eval.json` | 当前 checkpoint 的评估设置快照 |
| `eval/<eval-id>/<task>/global_step_<N>/summary.json` | 实验身份、协议与指标汇总 |
| `eval/<eval-id>/<task>/global_step_<N>/details.jsonl` | 按原始样本顺序汇总的回答与评分 |
| `eval/<eval-id>/{test,locomo}/global_step_<N>/boundary.jsonl` | 每个正文的边界概率、pending 长度与实际压缩位置 |
| `eval/<eval-id>/ruler/global_step_<N>/lengths/<长度>/boundary.jsonl` | 按长度保存的 RULER 边界诊断 |
| `eval/<eval-id>/<task>/global_step_<N>/work.json` | 本次执行的待评估样本分工 |
| `eval/<eval-id>/<task>/global_step_<N>/progress/worker_<rank>.jsonl` | 各分片逐文档／题目保存的完成进度与资源记录 |
| `eval/<eval-id>/<task>/global_step_<N>/progress/worker_<rank>.json` | 阶段、批量与长度的展示快照，每次续评重建 |

- **结果管理**：

  - **结果提交**：各分片在一篇 Test 文档的全部 QA 或一道 benchmark 题目完成后保存进度，样本齐全并通过校验后才提交最终 details 与 summary。RULER 另在 `lengths/<长度>/` 保存计划中各长度的完成结果，续跑复用已有记录。
  - **运行关联**：Summary 的 `checkpoint` 指向绝对 `state.pt` 路径，`training_run.directory` 记录实际 `train/` 路径。

- **启动看板**：通过 Streamlit + ECharts 查看本地结果，`--dashboard-run-dir` 指向运行根目录；启动后访问 `http://127.0.0.1:8501`。

```bash
bash src/latent_working_memory/v3_1_1/scripts/dashboard.sh \
  --dashboard-run-dir artifacts/v3-1/policy-post_k256_20261104 \
  --dashboard-eval-id standard_20261104 \
  --dashboard-baselines \
    base-Qwen artifacts/baselines/base/Qwen3-4B-Instruct-2507
```

- **结果浏览**：

  - **边界诊断**：仅指定 `--dashboard-run-dir` 即可查看 train 批次与 dev 验证集的概率直方图和 pending–概率散点图。选择多个训练步可在同一张图中比较。独立评估页面按任务提供相同的图表。详见[边界诊断的统计口径](../README.md#本地看板)。
  - **评估选择**：`--dashboard-eval-id` 指定启动时选中的评估，省略时默认名称排序后的末项；页面仍可切换其他 eval-id。指定评估不存在或没有已完成结果时提示错误。
  - **任务切换**：按 Test、RULER、LoCoMo 切换任务；点击“刷新结果”重新加载本地结果。
  - **比较区**：位于页面上方，展示全部已完成 checkpoint 的指标汇总和按实际 optimizer step 升序排列的变化曲线；基线排在当前方法之前。“资源测量与执行设置”和“输出诊断”在曲线下方折叠展示，均汇总多个 checkpoint。只有一项结果时展示评估概览。
  - **详情区**：位于比较区下方，“明细 checkpoint”默认选择最新完成的一项。选择器下方依次展示折叠的“评估配置与统计”、RULER 单步明细、压缩分析和问答；切换详情不改变上方比较范围。

- **压缩与逐文本分析**：

  - **统计口径**：整篇倍率为实际编码后的完整正文 tokens 除以 memory slots、未压缩 pending 与 recent tokens 之和，单块倍率为块长除以 K；分别展示平均、最低、最高值，平均按篇或块等权。评估另记录前缀保留比例 `prefix_retention_ratio=(K×B+P)/(D−R)` 及其 p 次幂 `memory_cost`，完整 `retention_ratio=(K×B+P+R)/D` 独立保留。若正文全为 recent，前缀比例和成本记为 0，完整比例为 1。正文长度按 Writer 输入计数。LoCoMo 每段对话只计一次，末尾强制提交块计入块统计。
  - **概览与分布**：另列每篇平均记忆块数及单块文本比例，展示整篇倍率分布、单块倍率分布和“输入长度—记忆块数”散点图。
  - **文本选择**：按输入长度、压缩倍率或块数排序，默认选中并高亮首行，始终保持一篇文本被选中；下方分段条、块信息表和问答与选中行同步。分段条宽度表示块长、颜色表示倍率，黄色表示未压缩 pending，灰色表示固定 recent，支持缩放；按实际动作标注策略提交或末尾强制提交。
  - **加载方式**：首次扫描当前 checkpoint 的明细，缓存压缩字段和记录位置；只为所选文本读取完整回答并生成分段条，不产生图片或逐文本图表文件。

- **RULER 展示**：

  - **长度与任务**：用折线图比较各长度的 Mean，选择长度后用分组柱图展示 13 项任务；只显示已完成的长度，同一方法保持同色。
  - **Checkpoint 趋势**：多 checkpoint 可选择 Overall 或具体长度查看 step 曲线，再查看单个 checkpoint 的长度总览与任务明细。基线 Overall 按对应 checkpoint 的长度范围重算，缺任一长度时不显示该点，不补零。
  - **文本筛选**：当前 checkpoint 的压缩概览、三张分布图和文本列表共用长度与子任务筛选；选择文本后查看参考答案、模型响应与评分。
  - **完整题面**：展开“完整输入题面”，从原始 `test.jsonl` 读取全文；已有评估结果可直接使用，无需重新评估。

- **基线接入**：

  - 通过 `--dashboard-baselines <标签> <结果目录> ...` 添加多组基线。原始模型基线可以直接指定 `artifacts/baselines/base/<model_name>/`。也支持直接含 `summary.json` 的单任务或单 checkpoint 目录。标签用于图例，须唯一。
  - **异常提示**：缺少当前 benchmark 汇总时跳过该基线；路径、格式或比较设置不一致时在页面提示，不影响已有结果。

- **比较规则**：

  - **质量比较**：核对基座模型标识、实际输入与参考答案、输入／解码／生成预算及评分口径。基座按模型路径末级名称识别，父目录可以不同；算法、窗口、精度、设备和批量不作为质量比较限制。
  - **数据核对**：使用顺序无关的比较指纹，RULER 按长度分别核对。旧结果可沿用原数据指纹；缺少分组指纹且覆盖范围不同时，提示无法核实对应范围。
  - **资源与成本**：资源统计与质量指标分区展示。批量计算时间按参与样本分摊后累计，不等同于每条请求的独立等待时延；峰值显存按 worker／批次范围观测。比较成本须统一硬件、实际评估批量及测量范围，不自动计算加速比；重试墙钟时间仅覆盖本次执行。
  - **缓存校验**：跨方法展示允许实现与运行环境不同；结果复用、失败重试和训练恢复仍严格核对原有身份与设置。

- **访问方式**：

  - **访问配置**：地址和端口见 [`.streamlit/config.toml`](../../../../.streamlit/config.toml)，修改后须重启看板；临时更换端口可在启动命令前加 `STREAMLIT_SERVER_PORT=8502`。
  - **异地查看**：训练边界诊断同步 `<运行目录>/train/boundary/`，独立评估同步 `<运行目录>/eval/`，保留目录结构并将 `--dashboard-run-dir` 指向运行根目录。查看 RULER 完整题面还需对应原始数据，在页面“RULER 数据目录”填写本地路径，目录下应包含 `<长度>/<子任务>/test.jsonl`。

## 5. 调整参数

本节汇总训练、恢复、独立评估、离线准备与看板的公开参数。训练默认值以[预训练配置](../../../../configs/v3_1_1/pretrain_default.json)和[后训练配置](../../../../configs/v3_1_1/post_default.json)为准：新训练的显式 CLI 值在校验前覆盖 JSON，省略时沿用配置。模型结构、训练目标及优化器设置直接修改 JSON，见[配置与运行](../README.md#配置与运行)。

### 5.1 `prepare_pre.sh`、`prepare_post.sh`、`prepare_length.sh`

共同参数：

| 参数 | 用途／默认值 |
|---|---|
| `--model-path` | 必填，tokenizer 路径 |
| `--prepare-data` | 必填；`prepare_pre.sh`／`prepare_post.sh` 为来源 JSONL 文件，`prepare_length.sh` 为来源数据目录 |

专属参数：

| 参数 | 用途／默认值 |
|---|---|
| **prepare_pre.sh／prepare_post.sh** | |
| `--output-file` | 必填，完整输出文件路径 |
| `--seed` | 非负随机种子，默认 42 |
| `--target-samples` | 目标生成条数，支持正整数和 `null`（默认） |
| `--min-tokens` | 实际正文长度下限，支持正整数和 `null`（默认） |
| `--max-tokens` | 实际正文长度上限，支持正整数和 `null`（默认） |
| **prepare_pre.sh** | |
| `--length-bins` | 长度区间数，默认 10 |
| `--length-distribution` | 不同长度区间的样本数分布，支持 `uniform`（默认）和 `random` |
| **prepare_post.sh** | |
| `--qas-per-sample` | 每条数据 QA 数上限，支持正整数和 `null`，默认 10 |
| **prepare_length.sh** | |
| `--output-dir` | 必填，完整输出数据集目录 |
| `--split` | 必填，处理对应 split，支持 `train`、`dev` 和 `test` |
| `--length-bins` | 长度分档数，默认 4 |

### 5.2 `train.sh`

| 参数 | 用途／默认值 |
|---|---|
| **运行方式** | |
| `--resume` | 恢复训练，默认不开启 |
| `--overwrite` | 清理当前 run 并重新训练，默认不开启 |
| `--checkpoint` | 恢复训练必填，最新完整 checkpoint 的单个路径 |
| **配置与模型** | |
| `--config` | 新训练必填，JSON 配置文件；阶段取文件中的 `stage` |
| `--model-path` | 基座与 tokenizer 路径，省略时沿用配置 |
| `--seed` | 非负随机种子，默认 42 |
| `--init-checkpoint` | 覆盖配置中的压缩器来源；`joint/boundary_only` 须提供有效来源，`compressor_only` 可选；加载 AE／LM 或 `compressor_only` 压缩器，使用新的 optimizer／scheduler |
| **目录与运行标识** | |
| `--output-root` | 训练产物父目录，默认 `artifacts/v3-1` |
| `--run-id` | 训练运行目录名，默认见[目录命名](#7-产物目录与记录) |
| **数据集与筛选** | |
| `--train-data` | 训练必填，训练与 dev 验证的数据目录 |
| `--max-train-samples` | 训练样本数上限，支持正整数和 `null`（默认） |
| `--max-train-data-tokens` | 预训练完整样本／后训练正文的实际 token 数上限，支持正整数和 `null`（默认） |
| **训练时长** | |
| `--epochs` | 普通训练轮数，仅关闭长度课程时生效，默认 1 |
| `--length-curriculum` | 按长度档位从短到长分阶段训练，默认不开启 |
| `--curriculum-stage-epochs` | 长度课程每阶段完整遍历本档数据的轮数，默认 1 |
| `--curriculum-replay-ratio` | 长度课程的历史补充数量／本档样本数，支持 `[0, 1]`，默认 0.2 |
| `--warmup-ratio` | 学习率预热占实际总 optimizer steps 的比例，支持 `[0,1)`；省略时沿用配置，预设为 0.1 |
| **批处理** | |
| `--micro-batch-size-per-gpu` | 每卡 microbatch 的样本数，默认 1 |
| `--gradient-accumulation-steps` | 每次优化器更新前累积的 microbatch 数，默认 16 |
| **计算设置** | |
| `--dtype` | 基座参数精度，支持 `float32`（默认）和 `bfloat16` |
| `--attention-implementation` | attention 实现，支持 `sdpa`（默认）和 `flash_attention_2` |
| `--gradient-checkpointing` | Transformer 层内激活重算，默认关闭 |
| `--mixed-precision` | FP32 参数与标准 BF16 autocast，默认关闭 |
| `--save-on-cpu` | 反向所需的保存张量暂存到 CPU，默认关闭 |
| **验证、保存与记录** | |
| `--save-every` | checkpoint 保存的正整数 optimizer step 间隔或 `after_each_epoch`（默认） |
| `--save-total-limit` | 保留最近成功保存的 checkpoint 数，默认 2 |
| `--max-dev-samples` | dev 验证样本数上限，支持正整数和 `null`（默认） |
| `--dev-every` | dev 验证的正整数 optimizer step 间隔或 `after_each_epoch`（默认） |
| `--tracking` | SwanLab 记录，支持 `online`（默认）和 `offline` |

- **数据筛选**：按完整正文的实际 token 数筛选，不含系统提示、chat template 等。AE／LM 和 `compressor_only` 的 single 要求长度大于 K，multi 要求至少 `recent_size + min(ratios) × K`。policy 要求至少 `recent_size + pending_size`。train 使用训练实际 recent，dev 使用模型默认 recent；上限仅用于 train，超限样本整条排除，`null` 取消上限。LM 前缀不足对应下限时改用 AE，实际编码／解码仍须满足模型窗口限制。
- **验证与保存**：除了 `--save-every` 和 `--dev-every` 所指定的 step，每轮末和最终步也保存 checkpoint 及进行 dev 验证；长度课程中一轮指当前档位的一轮训练，`after_each_epoch` 同样适用。

### 5.3 `eval_test.sh`、`eval_locomo.sh`、`eval_ruler.sh`

共同参数：

| 参数 | 用途／默认值 |
|---|---|
| **运行方式** | |
| `--resume` | 续跑或补评当前任务，默认不开启 |
| `--overwrite` | 重建当前任务的计划和全部 checkpoint 结果，默认不开启，与 `--resume` 互斥 |
| **目录与运行标识** | |
| `--eval-run` | 评估必填，训练运行根目录，用于定位计划和 checkpoint |
| `--eval-id` | 评估标识，默认 `YYYYMMDD-HHMMSS` |
| `--output-root` | 评估产物父目录，省略时保存在 `--eval-run`；指定时使用 `<output-root>/<原run目录名>/` |
| **数据与评估范围** | |
| `--eval-data` | 新建或覆盖必填，当前任务的数据目录；续跑沿用原计划 |
| `--checkpoint` | 一个或多个 `global_step_<N>` 名称；新建或覆盖时省略选择当时全部，续跑时省略沿用累计列表，显式传入仅调度指定项并累计新增项 |
| **批处理** | |
| `--batch-size-per-gpu` | 每卡评估批量上限，默认 1，续跑可调整 |

专属参数：

| 参数 | 用途／默认值 |
|---|---|
| **eval_test.sh** | |
| `--max-test-samples` | Test 评估文档数上限，支持正整数和 `null`（默认） |
| `--max-new-tokens` | Test 每题生成 token 数上限，默认 128 |
| **eval_locomo.sh** | |
| - | - |
| **eval_ruler.sh** | |
| `--lengths` | RULER 评估长度档位，默认 `8192 16384 32768` |
| `--batch-size-by-length` | RULER 各长度档位的每卡批量上限，格式为一个或多个 `length=batch`，如 `8192=8 16384=4 32768=2`；未指定的档位沿用 `--batch-size-per-gpu` |

- **批量调整**：续跑可修改通用批量与逐长度映射；省略 `--batch-size-per-gpu` 时使用默认 1，不沿用上次批量。

### 5.4 `dashboard.sh`

| 参数 | 用途／默认值 |
|---|---|
| `--dashboard-run-dir` | 看板必填，指向运行根目录 |
| `--dashboard-eval-id` | 看板启动时选中的 eval-id；省略时默认列表首项，页面可切换 |
| `--dashboard-baselines` | 按“标签 结果目录”成对传入，可指定基线模型目录或直接含 `summary.json` 的目录 |

## 6. 配置变更与结果复用

| 变更 | 操作 |
|---|---|
| 预训练准备的来源文件、tokenizer、目标数量、完整样本长度范围、区间数或分配方式 | 修改准备参数，准备对应 split，再启动新训练 |
| 后训练准备的来源文件、tokenizer、目标数量、正文过滤范围或每条 QA 上限 | 调整 `prepare_post.sh` 参数，准备对应 split，再启动新训练 |
| 长度分档的正文、QA、tokenizer 或档位数量 | 重新执行 `prepare_length.sh`，检查分布后启动新训练 |
| 从 AE／LM 进入后训练，或从 `compressor_only` 进入边界策略训练 | 新 run，通过 `--init-checkpoint` 载入兼容压缩器，optimizer／scheduler 重新初始化 |
| 模型结构、计算、训练目标或优化器设置 | 修改基础 JSON，再启动新训练 |
| 数据、batch、轮数或保存等运行设置 | 新训练使用 CLI 覆盖或修改基础 JSON |
| 评估重试、调整评估批量或 GPU 数量 | 使用原任务脚本追加 `--resume`，以原 `--eval-run`、`--eval-id` 定位计划，只重新分片并组批原计划的未完成样本；RULER 可调整逐长度批量 |
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
    │   ├── config.json                       基本配置，包括 stage、seed、model_name_or_path（基座模型）、model（训练模块）与 compute
    │   ├── train.json                        训练设置与执行安排
    │   └── eval-<eval-id>/<task>.json        各任务的评估计划
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
            ├── progress/worker_<rank>.json     本轮展示快照
            ├── summary.json                  指标、数据、比较指纹与资源统计
            ├── details.jsonl                 逐文档／题目回答与评分
            └── lengths/<长度>/               RULER 各长度的评估结果
```

- **目录命名**：完整训练路径为 `<output_root>/<run_id>/train`。默认 `run_id`：AE／LM 使用 `pretrain_k{memory_slots}_{YYYYMMDD-HHMM}`，`compressor_only` 使用 `compressor-post_k{memory_slots}_{YYYYMMDD-HHMM}`，`joint/boundary_only` 使用 `policy-post_k{memory_slots}_{YYYYMMDD-HHMM}`。时间戳采用 `Asia/Shanghai` 时区、精确到分钟。显式指定的 `run_id` 原样作为目录名。新训练默认拒绝覆盖已有计划或产物，显式 `--overwrite` 才清理并重建；恢复训练沿用原目录。
- **计划与配置**：`plan/` 保存执行前确定的设置，最终解析生成的 `run_id` 写入计划配置；`train/config.json` 保存实际训练配置，评估结果目录的 `eval.json` 保存评估设置快照。训练只创建训练计划；独立评估默认新建不存在的 task 计划，固定数据、预算与长度，`checkpoints` 保存累计范围。续跑可选择本次 checkpoint 子集并加入新项，仅有新增时原子更新计划；`--overwrite` 重建当前 task。
- **覆盖与互斥**：校验通过后，持删除范围之外的稳定锁执行清理与重建；锁覆盖 worker 生命周期，占用时拒绝覆盖。锁文件不属于实验结果。
- **训练与恢复**：Rank 0 统一写入产物，train/dev 指标汇总全部 GPU 的实际样本。`state.pt` 保存压缩器 LoRA、memory embeddings、配置适用的边界与 value head 参数、按 actor／critic 分组的优化器与 scheduler、真实更新进度、rollout 批次和数据游标及各 rank 随机状态；恢复时严格核对训练计划与绑定状态，不依赖评估计划。完整格式见[方法说明](../README.md#代码与产物)。
- **评估结果**：各卡结果按原始样本顺序汇总；Test 的 `qa_nll` 先按题、再按文档平均，`token_nll` 按答案 token 加权，EM／F1 按题平均，benchmark 沿用官方汇总。评估续跑复用完整结果；未完成时保留逐文档／题目进度，可调整 GPU 数和评估批量，只重新分片、组批剩余样本。RULER 继续复用已完成长度。
- **比较指纹**：Benchmark summary 的 `comparison_data` 按原始输入、参考答案、生成预算与任务生成，忽略执行顺序，RULER 按长度分组；仅供跨方法展示，不参与 `eval.json`、缓存或恢复校验。
- **日志与状态**：训练日志同步写入文件和终端；评估 worker 的完整输出合并到 `eval.log`，带 step、GPU 和 rank 前缀，终端由主进程统一展示进度。任务 `status.json` 的 `eval_run` 指向运行根目录，`selected_checkpoints` 是本次调度范围，与计划 `checkpoints` 的累计范围区分。状态记录 checkpoint 与各分片的设备、命令、起止时间、返回码和错误，整体进度仅统计本轮选择。
- **展示方式**：SwanLab 默认开启，记录 train/dev、训练资源及配置。project、group 和 tags 通过配置文件设置；project 默认为 `latent-working-memory-v3.1`，group 未指定时默认 `run_id`。`--tracking offline` 关闭 SwanLab 记录，仅保留本地产物；恢复训练沿用 `swanlab.json` 中的 run。评估由独立 dashboard 展示，不读取 SwanLab 身份或凭据。
- **verl 接入**：边界训练直接运行原生 `PPOTrainer.step`，由其控制 critic／actor 更新与 critic 预热；RL 更新进入原生 `BaseEngine.train_batch` 生命周期，优势与损失尽量调用原生函数。分段采样、任务 autograd、复制式梯度同步、外层预算与单文件 checkpoint 保留项目适配，不启动 Ray worker 或 TransferQueue 服务。
- **后训练图表**：`compressor_only` 记录 QA NLL、记忆用量和固定倍率 dev；边界策略模式另记录奖励、零优势比例、策略熵与回放概率偏差。SCST 的 greedy 基线并入奖励图，单随机路径时 `reward_std` 仅保留本地；启用退火时增加 `policy_temperature` 图。train 的“compression_retention_ratio”“segment_compression_ratio”“memory_blocks”分别绘制 `compression_retention_ratio`、`segment_compression_ratio`、`memory_blocks`；Y 轴仅标单位，依次为 `%`、`×`、`blocks`；保留率底层记录原始比例，仅由原生图表表达式乘 100 展示百分比。dev 按相同指标合并各条件，通用 dev 路径只为实际产生的裸指标配置对应图。图表标题与指标显示名采用对应 key 的小写下划线名称；NLL、loss、reward、learning_rate、EM／F1 和其他无量纲指标的 Y 轴标题留空，资源耗时和显存仅保留 `s`、`GiB`。首次出现真实记录时注册并配置原生面板，运行中只追加真实 optimizer step 的标量，dev 在实际验证步记录。辅助统计保存在本地，完整指标口径与展示规则见[方法说明](../README.md#验证独立评估与记录)。
- **固定倍率基线**：启用非空 `dev.fixed_ratios` 时，`boundary_only` 在首次正常 dev 计算并缓存固定倍率结果，本次进程的后续验证复用；恢复后首次 dev 重建。压缩器可更新的训练每次重新计算。当前两份 `post_k256_bonly_*` 预设未启用固定倍率对照，policy 每次 dev 重新评价。
- **压缩统计**：压缩区保留率为 `(K×B+P)/(D−R)`，记忆块数为 B，二者按文档等权平均；分段压缩倍率为所有已压缩段的平均长度除以 K，按块加权，未压缩 pending 和 recent 不参与，没有压缩块时不记录倍率。dev 的三张条件图使用一致的条件顺序和颜色，图例依次为 `policy`、`2×`、`4×`、`8×`、`16×`，每图最多 5 个数据源；共享项目保留历史运行已有的同指标条件，超过平台单图 8 个数据源时需按实验条件拆分。完整 `retention_ratio` 仅保存在本地。
