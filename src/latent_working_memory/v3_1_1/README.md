# 20261102_单次压缩预训练与自适应边界后训练

创建时间：20261102 16:06:43 Asia/Shanghai

最后修订时间：20261108 15:52:29 Asia/Shanghai

[运行指南](scripts/README.md) · [预训练配置](../../../configs/v3_1_1/pretrain_default.json) · [后训练配置](../../../configs/v3_1_1/post_default.json) · [K256 预训练](../../../configs/v3_1_1/pretrain_k256.json) · [K256 后训练](../../../configs/v3_1_1/post_k256.json)

每次提交将待压缩正文写成固定 K 个 memory slots，已提交的块保持不变。先用 AE／LM 目标建立单次压缩能力，再用 QA 训练记忆内容、用 GRPO 训练因果边界策略。执行层沿用 v4.1 的配置、计划、训练、恢复与独立评估组织方式；方法计算由本模块实现。

## 方法与数据

train／dev 的长度筛选在预训练按完整样本 `text`、后训练按正文的实际 token 数进行，均须严格大于 `model.memory_slots`，不计系统提示与 chat 前缀。`training.max_train_data_tokens` 按同一口径仅限制 train，超限样本整条排除；独立 test 不套用训练筛选。

### 单次压缩预训练

`prepare-pre` 读取原 FineWeb Multisegment 窗口，将 `text + continuation` 合并成完整原文，再按目标长度随机裁剪一条完整 `text`。预训练只读取准备后的 `train.jsonl`、`dev.jsonl`；JSONL 每行仅为 `{"text":"完整预训练文本"}`。

每条完整 `text` 是一个训练样本，先按其实际 token 数执行长度筛选，再依据 seed 和文本内容标识按概率选择 AE 或 LM，跨 epoch 固定，每个样本只计算所选目标：

- AE 将完整 `text` 压缩为 K slots，并重建完整文本。
- LM 以完整样本最后 `objective.lm_target_tokens` 个 tokens 为目标（默认 512），仅将前缀压缩为 K slots；若剩余前缀长度不大于 K，则改为 AE。

默认 `objective.lm_ratio=0.5`，两个目标均追加 EOS。基座与读取器冻结，只训练压缩器 LoRA 和可学习 memory embeddings；冻结读取器仍保留损失到 memory 的梯度。训练时的筛选与限量在内存中完成，不保存派生数据副本。

### 后训练（QA／GRPO 联合训练）

`post` 使用每行一篇完整文档及其全部 QA 的数据：

```json
{"text":"完整正文","qas":[{"question":"问题","answer":"答案"}]}
```

每题必填 `question`、`answer`，可成对附带 `segment_id` 和 `segment_char_span`，例如：

```json
{"text":"完整正文","qas":[{"question":"问题","answer":"答案","segment_id":"seg0","segment_char_span":[0,4]}]}
```

`prepare-post` 从 FineWeb FactQA 的 task 题池准备该格式，完整 `text` 逐字保留；按要保留的题数 N 将全文等分为 N 份，依次选择答案证据最靠近各份中心的 QA，每题最多选择一次，最后按证据在正文中的顺序输出。每条默认最多 10 题，少于上限时全保留，`null` 保留全部 task 题。段区间来自原段，使用相对当前 `text` 的左闭右开字符下标；这些属性仅用于来源和位置分析，不参与边界判断，评估结果与看板逐题保留展示。QASPER 的 question/answer 数据同样可读。完整命令见[后训练数据准备](scripts/README.md#22-后训练数据)。

一篇文档及其固定 QA 集合是一个训练单位。同一文档采样 G 条完整硬分段；每条路径只压缩一次整篇正文，各题共享其记忆。各题先按答案 token 平均 NLL，再对题目、轨迹和文档等权平均；分布式归约按实际文档数计算。

在线状态包含历史 memory、待提交原文 a 和新到达单元 b。边界模块观察当前已到达的内容，选择继续积累 ab，或提交 a 并保留 b 作为下一块开头。训练与文档 QA test 中，问题、答案和未来正文不进入边界判断。末尾非空正文强制提交，不计为随机动作。公开 benchmark 按其完整题面协议编码，题面中的问题也参与压缩；生成答案期间不再写入 memory。

当前计算要求 full-attention 和固定 default RoPE。基座默认 BF16，所有可训练参数、梯度和优化器状态使用 FP32。预训练不训练边界模块；后训练阶段的目标为：

```text
reward_g = -stopgrad(QA_NLL_g) - slot_cost × K × blocks_g
loss = mean(QA_NLL_g) + policy_weight × GRPO_loss
```

奖励、组内优势和旧动作概率停止梯度；压缩器 QA 损失始终使用正权平均，不乘优势。每批固定参数无梯度采样并评价，再固定全部动作带梯度回放，累计整批梯度后联合更新一次，下一批刷新轨迹。采样、dev 和独立评估复用原文增量 KV；提交前排除仅用于判断的 b，追加 memory tokens 压缩 a，再以新 memory 上下文重建 b 的缓存。回放重算完整可见前缀，保留历史 memory 的梯度，并按轨迹执行 activation checkpoint 重算。

首版不多轮复用轨迹。数学上，同一组参数的新旧概率比为 1；BF16 分块采样与完整回放可能存在舍入差，`train/replay_log_probability_drift` 记录轨迹内最大 log probability 差的平均值。稳定性主要依赖保守更新与及时刷新反馈。

## 配置与运行

配置主体是 `stage`、`seed`、`model`、`objective`、`training`。`stage` 使用 `pretrain` 或 `post`，分别表示单次压缩预训练和后训练。模型与目标参数保存在各自配置类中；数据目录、初始化来源、训练预算、batch、优化器、保存和 SwanLab 设置集中在 `training`。显式 CLI 覆盖 JSON 后统一校验，最终设置保存到计划、实际配置、checkpoint 和启用时的云端 config，源 JSON 文件保持原样。

模型结构与计算、方法目标、优化器及学习率调度只通过 JSON 修改，`*_default.json` 的 K128 默认设置如下：

- `model`：`dtype="bfloat16"`、`attention_implementation="sdpa"`、`gradient_checkpointing=true`；`memory_slots=128`、`lora_rank=16`、`lora_alpha=32`、`lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`、`boundary_hidden_size=128`。
- `objective`：`lm_ratio=0.5`、`lm_target_tokens=512`；`unit_tokens=256`、`group_size=4`、`slot_cost=0.001`、`policy_weight=1.0`、`clip_epsilon=0.2`、`advantage_epsilon=1e-6`。
- `training` 中的优化设置：`compressor_learning_rate` 预训练为 `1e-4`、后训练为 `2e-5`；`boundary_learning_rate=1e-4`、`weight_decay=0.01`、`gradient_clip=1.0`、`warmup_steps=50`、`min_lr_ratio=0.1`。AdamW 的 `beta1=0.9`、`beta2=0.999`、`adam_epsilon=1e-8`。

本文训练示例采用 K256 配置，两阶段基座均为 `/root/rivermind-data/models/Qwen3-4B-Instruct-2507`；预训练读取 `data/fineweb_train5k_20261108`，后训练读取 `data/fineweb-factqa_q10_train1k_20261108`。预训练／后训练的 `warmup_steps` 分别为 20／5，均每 20 个 optimizer steps 验证和保存。预训练不设实际 token 数上限，后训练上限为 12288；准备阶段的 4096 是字符估算长度上限。

新训练必须同时指定阶段（`--stage pretrain` 或 `--stage post`）和配置文件（`--config <配置路径>`），配置中的 `stage` 必须与 `--stage` 一致。训练恢复从原计划读取全部实验设置，其他已知 CLI 实验设置忽略并集中提示；独立评估从 checkpoint 读取模型与方法设置。

运行根目录为 `<output_root>/<run_id>/`，显式 `run_id` 原样作为目录名。`training.run_id` 为 `null` 或省略时，按阶段生成 `pretrain_k{memory_slots}_{YYYYMMDD-HHMM}` 或 `policy-post_k{memory_slots}_{YYYYMMDD-HHMM}`；时间使用 `Asia/Shanghai`，精确到分钟。最终解析的 `run_id` 写入计划与配置，恢复训练沿用原目录。

使用项目根目录 `.venv/`。`CUDA_VISIBLE_DEVICES` 决定训练与评估使用的全部 GPU，每卡一个进程。复用项目公共 runtime，其中多卡 CUDA 进程组由 verl 初始化；采样回放、损失归约和 optimizer 更新是本项目的训练循环。

以下示例分别准备 train/dev，再使用 K256 配置执行预训练和后训练。`--prepare-data` 指定来源 JSONL 文件，每次只读取该文件，先合并每条原始记录的 `text + continuation`，按完整原文的 `floor(len / 4)` 统计估算长度与可用容量；请求范围夹取到该文件合并原文的最短、最长长度之间，`--min-tokens`、`--max-tokens` 接受正整数或 `null`，默认 `null` 时采用对应来源边界。本例使用来源最短长度为下界，上界最多 4096。训练阶段仍按 tokenizer 的实际 token 数执行长度筛选。

`--target-samples` 指定目标生成数据数量：正整数指定生成条数，超过可用数量时报错；`null`（默认）不设构造数量上限，使用满足长度条件的全部数据。默认分 10 个长度区间，整数长度不足时减少；`uniform` 为各区间设置等配额、尽量均衡，`random` 逐样本等概率选择区间，两者均在区间内随机选择整数长度。随机种子可通过 `--seed` 设置，默认 42，从合并原文随机裁出一条完整 `text`，准备长度范围约束整条输出文本；AE／LM 的输入与目标由训练阶段划分。数据写入 `--output-file`，已存在时报错；同目录的元信息按输出文件名命名，例如 `train.jsonl` 对应 `preparation-train.json`。完整流程见[离线数据准备](scripts/README.md#2-离线数据准备)。

```bash
for split in train dev; do
  if [ "$split" = train ]; then target_samples=5000; else target_samples=null; fi
  bash src/latent_working_memory/v3_1_1/scripts/run.sh prepare-pre \
    --prepare-data "data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/${split}.jsonl" \
    --output-file "data/fineweb_train5k_20261108/${split}.jsonl" \
    --seed 42 \
    --target-samples "$target_samples" \
    --max-tokens 4096 \
    --length-bins 10 \
    --length-distribution uniform
done
```

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --stage pretrain \
  --config configs/v3_1_1/pretrain_k256.json \
  --run-id pretrain_k256_20261104 \
  --epochs 1 \
  --micro-batch-size-per-gpu 2 \
  --gradient-accumulation-steps 4
```

预训练完成后，从 `train/summary.json` 取得真实 checkpoint；后训练阶段使用新的 run，并通过 `--init-checkpoint` 指定该来源：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --stage post \
  --config configs/v3_1_1/post_k256.json \
  --run-id policy-post_k256_20261104 \
  --init-checkpoint 'artifacts/v3-1/pretrain_k256_20261104/train/checkpoints/global_step_<N>' \
  --micro-batch-size-per-gpu 2 \
  --gradient-accumulation-steps 4
```

`<N>` 替换为真实 optimizer step。两阶段 `model` 配置须一致；后训练初始化只继承压缩器，重新创建边界模块、optimizer、scheduler、游标与边界随机流。恢复同阶段训练用 `--resume --checkpoint <最新完整checkpoint路径>`，沿用原计划、数据和 GPU 数量；其他已知实验设置忽略并提示。预算已完成时直接返回，不读取或启动评估计划。恢复后训练不依赖预训练来源文件。`--resume` 与 `--overwrite` 互斥，可搭配 `--evaluate-only` 续跑评估。

`training.epochs` 控制完整轮数，JSON 中可选 `training.max_steps` 进一步限制更新次数。全局 batch = 训练进程数 × 每进程 microbatch × 梯度累积次数。默认每卡 microbatch 为 2、梯度累积为 4；以上两阶段示例均使用双卡，全局 batch 为 `2 × 2 × 4 = 16`；epoch 尾批按实际样本数归约，不补齐、不丢弃、不跨 epoch 拼接，空 rank 同样参与 collective。`warmup_steps` 和 `min_lr_ratio` 直接位于 `training`，控制学习率线性预热及后续余弦衰减；各参数组使用相同倍率。更多参数和可复制命令见[运行指南](scripts/README.md)。

## 验证、独立评估与记录

训练内 dev 使用固定样本与确定性边界，不更新参数。train/dev 使用相同真实 optimizer step 横轴，没有执行的指标不补零。预训练保存实际 AE／LM 选样数量，并记录 `lm_fraction` 和分项 NLL；后训练阶段记录 QA、策略目标、记忆用量和边界统计。压缩器与边界模块分别记录实际学习率和统一裁剪前的梯度范数。

训练完成后退出。独立评估仅接受后训练 checkpoint，通过 `--eval-task test|locomo|ruler` 选择任务，新建时用 `--eval-data` 指定数据并保存固定协议与 checkpoint 范围。文档 QA test 固定读取数据目录下 `test.jsonl` 的完整正文，不应用训练长度筛选。评分先对每题目标 tokens 平均，再对题目及文档等权平均得到 `qa_nll`；`token_nll` 另按全部目标 tokens 加权，目标含模板结束符。EM／F1 按题平均。LoCoMo、RULER 使用项目公共协议和评分。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/run.sh \
  --evaluate-only \
  --eval-run artifacts/v3-1/policy-post_k256_20261104 \
  --eval-id standard_20261104 --eval-task test \
  --eval-data data/delta-mem-qasper --max-new-tokens 128 \
  --checkpoint 'global_step_<N>' \
  --eval-batch-size-per-gpu 8
```

每卡评估批量由 `--eval-batch-size-per-gpu` 控制，默认 8。Test 以文档为单位，每篇文档的全部 QA 共用其 memory，按 QA 轮次每篇选一题组批，全部 QA 完成后保存该文档进度；benchmark 每题完成后保存，RULER 按 length 分组。批量仅控制本次执行，不写入永久评估计划或请求身份，可在续评时调整。

`--eval-run` 必填并指向运行根目录；评估的 `--checkpoint` 只填 `global_step_<N>` 名称，省略时新建选择当时全部完整 checkpoint。已有任务计划须显式用 `--evaluate-only --resume` 续评，同时提供原 `--eval-run`、`--eval-id` 和 `--eval-task`。续评省略 checkpoint 时使用累计列表，传入时仅调度指定项并将新项加入计划；数据、预算与 RULER 长度从原计划恢复，对应 CLI 设置忽略并提示。GPU 数和每卡批量可调整，已提交记录复用；修改协议须新 eval-id 或 `--overwrite` 重建当前任务。多个 checkpoint 按真实 step 串行，评估日志、状态和结果均保存本地，不依赖 SwanLab 凭据。

默认 `training.tracking="disabled"`。启用 SwanLab 使用 `--tracking online --swanlab-project <已有项目>`，project 须明确配置。group 未指定时默认 `run_id`，可通过 `--group <实验系列>` 覆盖；可重复 `--tag` 追加标签并去重。训练与 dev 共用一个 run，恢复沿用原云端身份，自动追加方法与阶段标签。具体超参数保存至 config。

```bash
bash src/latent_working_memory/v3_1_1/scripts/dashboard.sh \
  --dashboard-run-dir artifacts/v3-1/policy-post_k256_20261104 \
  --dashboard-eval-id standard_20261104
```

本地看板保留训练曲线，独立评估按 test、RULER 和 LoCoMo 分页，展示质量、真实记忆用量、输出诊断及逐文档全部 QA。Benchmark 可接入本地基线；质量比较核对实际输入、参考答案、基座及公共协议，资源表保留各自真实精度、attention、实际批量及测量范围。批处理计算时间分摊给参与样本，decode 时间只归属当轮活跃样本；汇总不重复累计同一批 GPU 时间。峰值显存为 worker／batch 共享观测。

## 代码与产物

```text
v3_1_1/
  config.py / cli.py       typed 配置与显式 CLI 覆盖
  prepare_pre.py          长度分层的完整样本裁剪
  prepare_post.py         完整正文与固定位置 QA 选择
  data.py                 单次 AE／LM 与完整文档 QA
  modeling/               压缩器、边界模块和流式状态
  rollout.py / objective.py
  plans.py / execution.py  训练／评估计划和占用锁
  training/               调度、训练、梯度归约、checkpoint 与记录
  evaluation/             test／benchmark、分片、汇总和输出诊断
  dashboard.py / scripts/
```

训练计划分为 `<run>/plan/config.json` 与 `train.json`；实际选样和预算写入 `<run>/train/`。Checkpoint 位于 `train/checkpoints/global_step_<N>/state.pt`，固定包含 `config`、`model`、`optimizer`、`scheduler`、`cursor`、`rng`、`boundary_rng` 和 `binding`。`cursor.step` 为已完成 optimizer step，随机流按 rank 保存，恢复严格检查原配置、数据身份与游标。

独立评估计划为 `plan/eval-<eval-id>/<task>.json`，固定数据与协议并保存累计 `checkpoints`；结果为 `eval/<eval-id>/<task>/global_step_<N>/`，任务 `status.json` 的 `selected_checkpoints` 记录本次范围。逐文档／题目进度 flush 后可续评，完整 details 提交后才保存最终 summary。正式规模 GPU 训练、成本和方法效果需通过实际实验确认。
