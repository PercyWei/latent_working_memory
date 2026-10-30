# 20261027_v4.1 在线 TTT 记忆

创建时间：20261027 16:34:06 Asia/Shanghai（UTC+08:00）

最后修订时间：20261030 19:31:25 Asia/Shanghai（UTC+08:00）

每层维护独立的 latent tokens，生成共享 memory KV，通过少量局部 Attention Matching（AM）更新压缩历史。外层用 QA 与完整历史提供的 AM 参照，学习连续写入与读取。

[运行指南](scripts/README.md) · [默认配置](../../../configs/v4-1/online_ttt_memory.json) · [框架笔记](../../../notes/v4-1/20261022_online_ttt_memory_framework.md)

## 1. 计算与执行流程

| 环节 | 当前实现 |
|---|---|
| 输入处理 | Memory 保存已压缩历史；pending 保存待压缩原始 KV；recent 保存近期原始 KV 与 queries |
| 记忆写入 | Pending 达到阈值后，固定本轮 recent queries，执行多步 inner AM，将旧 memory 与 pending 合并 |
| 训练监督 | 每次写入后计算 outer AM；全文处理后固定 memory，输入问题和答案，计算答案 CE |
| 外层更新 | 通过完整写入轨迹学习初始化、KV／bias 解码器、正步长与 attention Q/K/V LoRA |
| 推理 | 固定慢参数，沿用在线写入；正文处理完成后固定 memory，追加问题 KV 并自回归生成 |
| 完整运行 | 先保存训练和 test 计划；训练与 dev 结束、全部训练进程退出后，在可见 GPU 间按样本分片测试最后一个 checkpoint |

首版支持 full-attention Qwen3、固定 default RoPE、每卡训练 microbatch 1，不启用随机 dropout。GPU 由 `CUDA_VISIBLE_DEVICES` 指定：离线准备按文档分配，训练每卡一个数据并行进程，评估按样本分片共同处理一个 checkpoint，多个 checkpoint 按 step 串行。Inner 使用 recent 自然读取；outer 使用固定文本提示与 teacher 自然响应。Latent query 生成器与 self-study 保留为研究候选。

### 1.1 记忆接口与在线边界

每层状态为 $M^{(\ell)}\in\mathbb R^{m\times d_m}$。逐 slot 的 SiLU MLP 解码出 memory keys、values 和 attention bias，与原始 KV 拼接后参与显式 attention。Memory 没有额外 query 行：真实 token 可以读取全部 memory，原始 KV 保持因果可见性。

- 每个独立 QA 样本从可学习的 $M_0^{(\ell)}$ 开始；首次写入前不挂载 memory KV。
- 累计压缩长度为 $T$ 时，memory slot $i$ 的 RoPE 位置为 $Ti/m$；真实 token 保留绝对位置。保留的原始 KV 不重算、不重新旋转。
- 一个 token 完成全部层前向后才移动窗口。Recent 溢出到 pending；pending 达到 $L_p$ 时写入，清空 pending，保留 recent。
- 第 $j$ 次写入发生在已处理 $jL_p+L_r$ tokens 时，压缩范围终点为 $jL_p$。计数包含固定输入模板前缀；默认首次边界为 1280，之后间隔 1024。
- 两个边界之间可分块 prefill，分块不跨越写入时点。正文尾部未满的 pending 继续保留；问题与响应阶段仅追加原始 KV，不再触发写入。

### 1.2 Inner 与 outer 的参照分工

| 项目 | Inner AM | Outer AM |
|---|---|---|
| Queries 来源 | Recent 正常前向的逐层 queries | 固定基座在完整已到达历史上的提示与响应 queries |
| Teacher 局部区域 | 旧 memory + pending；首次只有 pending | 从历史起点到本轮压缩终点的原始 KV |
| Student 局部区域 | 当前候选 memory | 本轮更新完成后的 memory |
| 计算时机 | 每轮写入，训练与推理相同 | 离线固定文本、位置池与尺度；训练时重算选中 teacher 参照并计算 student |

Outer teacher 输入包含该边界已到达的完整原始历史，**包括 recent**；但局部压缩目标排除 recent、提示与响应 KV。Teacher 是独立冻结的基座实例，不使用 student 的 LoRA。Inner 只使用在线仍可取得的旧 memory 与 pending。

对已包含位置编码的 query/key，局部 attention 为：

$$
s_i=\frac{q^\top k_i}{\sqrt{d_h}}+\beta_i,\qquad
o=\sum_i\operatorname{softmax}(s)_i v_i,\qquad
z=\operatorname{logsumexp}_i(s_i).
$$

AM 等权相加局部输出 $o$ 与 log-mass $z$ 的标准化 MSE；$z$ 是归一化前总权重的对数。Teacher 尺度 $\sigma_o,\sigma_z$ 按层和 query head 计算并固定，标准化只进入损失；memory bias 同时用于正式 attention。

Inner 每轮使用全部 recent 位置，固定 queries 与 teacher 的数值，执行 $S_{\mathrm{in}}$ 步无动量 SGD。每层步长 $\eta_\ell=\operatorname{softplus}(a_\ell)$ 由 outer 学习；跨轮仅延续 memory，无额外 inner optimizer 状态。固定本轮参照不表示截断它对此前写入的元梯度。

Outer 对单样本计算：

$$
\mathcal L=\operatorname{mean}_{\text{答案 tokens}}\mathrm{CE}
+\lambda_{\mathrm{AM}}\operatorname{mean}_{\text{压缩轮次}}\mathcal L_{\mathrm{AM}}.
$$

AM 内部先对查询／通道、再对 heads 与层取平均；训练 batch 对完整样本损失取平均。每轮包括最后一轮均有 inner 与 outer AM。无压缩的样本只有 QA 损失。

### 1.3 精度与梯度

基座原有权重冻结。Meta-training 保留 inner 二阶梯度、跨写入轮次依赖及 memory 对后续表示的影响；非重入激活重计算复用本次确定的状态与采样。推理仍需为局部 TTT 启用 autograd，不能用 `torch.inference_mode()` 包住整个过程。

基座默认 BF16；memory、解码器、训练／推理中的局部 AM、可训练参数、梯度累积及 AdamW 状态使用 FP32。Student attention 的 scores 与 softmax 使用 FP32，权重与挂载的 memory KV 转为主干精度后聚合。完整跨轮训练图的开销随轨迹增长，固定在线缓存不意味着训练显存固定。

Train/dev 每卡保存完整的 teacher 与 student，单样本按边界顺序执行。Teacher 前向使用 SDPA，不建立梯度图，局部 AM 参照以 FP32 计算；其权重及完整原文 prefix KV 与 student 的元训练图共同占用显存。独立 test／benchmark 每卡只加载一份完整 student，每次处理 1 个独立样本；多卡通过样本分片并行，不另设评估 batch size。

多卡共享全局样本游标，将同一 batch 分配到各卡，不重复补齐尾批。各卡完成本地 outer backward 后，打包归约慢参数梯度，按全局实际样本数平均，再统一裁剪并更新。尾批没有样本的进程仍参与归约。Dev 同样分片执行，按全局样本数加权汇总。

## 2. 数据与 teacher 缓存

### 2.1 QA 数据

数据目录保存 `train.jsonl`、`dev.jsonl`、`test.jsonl`。每行一篇正文，字段固定如下；同篇文档的所有 QA 放在一行：

```json
{"text":"完整正文","qas":[{"question":"问题","answer":"答案"}]}
```

一篇文档的每条 QA 展开为独立样本。正文、问题和答案按固定模板分别分词后拼接；正文部分包含系统提示与聊天前缀，答案包含 assistant 结束符。文档标识由正文确定，因此同一文档的 QA 共用 teacher 缓存。

长度筛选保留完整正文，不截断。Train/dev 的原始正文至少为 $L_p+L_r$ tokens，独立 test 不加该下限；`max_train_data_tokens` 只限制 train 原始正文的 token 数，不含模板；模型窗口限制覆盖全部输入。Train/dev 数量上限按 seed 与稳定 QA 标识选取一次，train 每个 epoch 打乱，dev 固定。原始文档集合用于检查划分之间的重叠。

公共 benchmark 沿用已有任务的完整 prompt 与官方评分：完整 prompt（含问题）作为在线输入，prefill 结束后固定 memory 再生成。不从 prompt 文本猜测正文与问题边界，这与上述显式 QA 分段协议分别记录。

### 2.2 离线参考池

固定提示库使用 [Latent Compaction 官方的 480 条组合提示](https://github.com/THUNLP/latent-compaction/blob/81cfd4b3f30f820fb5f76451cb0f6385e0ac9510/latent_compaction/prompts.py)，来源与许可见 [teacher/prompts.py](teacher/prompts.py)。每个文档、每个压缩边界独立准备：

1. 以完整已到达原文为共同前缀，按提示原顺序分批贪心生成，默认每卡 `prepare_batch_size=64`。每条最多 24 个新 token，遇 EOS 独立结束；最后一个 token 仍执行前向以取得其 query，不继续生成新答案。
2. 提示／响应前向使用 FlashInfer，分别计算压缩范围内的原始历史、recent、提示／响应三段 attention，并按各段 log-mass 合并为完整输出。前两段 KV 在批内共享，提示／响应 KV 按序列保存。
3. 直接复用压缩历史段的局部输出与 log-mass，缩为每个 query head 的三项统计：输出通道均值、通道方差及 log-mass。每批结束后，按原提示编号与序列内位置执行 reservoir sampling，均匀、无放回保留 4096 个位置及其统计。相同响应与 seed 得到相同位置池；候选不足即准备失败。
4. 从最终完整候选池的统计量计算 $\sigma_o,\sigma_z$，应用尺度下限，以 FP32 保存。完整 query、局部输出向量与临时统计缓冲均不持久化。

GPU 保存一份共享原文 KV，以及当前提示批的私有 KV 和三项统计，中间向量及时释放。CPU 保留固定大小的候选池：以 4096 位置、36 层、32 query heads 为例，三项 FP64 统计约占 **108 MiB／进程**；这不包含模型权重、KV 和当前批工作区。

离线准备在每张可见 GPU 上启动一个进程，各加载一份冻结 teacher。文档按预计计算量分配，同篇文档的全部边界始终由一个进程处理；原文 prefill 使用 Hugging Face SDPA，每次处理一篇，并跨边界延续 prefix KV。各进程写入同一缓存目录中互不重叠的文档文件，中断后可复用已完成边界。

Prepare 与 train 从 `<output_root>/teacher/p{pending_size}-r{recent_size}` 读写缓存，不同 run 共享同一窗口配置的缓存；修改窗口后自动切换目录。自定义 `--output-root` 时，两者使用同一父目录。实际配置记录解析后的缓存路径，完整缓存身份由 `identity.json` 校验。

每个窗口目录分别包含 `identity.json` 与 `documents/`。`identity.json` 保存身份、生成终止标识 `eos_token_ids` 与全局 `prompt_token_ids`；后者为 480 条完整提示片段的 token ID 列表，跨边界共用。每个边界保存为 `documents/<document_id>/write_<j>.safetensors`，$j$ 从 1 开始：

| 张量 | 类型与形状；$L$ 为层数，$H_q$ 为 query heads |
|---|---|
| `response_token_ids` | INT64，所有提示的响应 token IDs 顺序拼接，保留生成的 EOS |
| `response_offsets` | INT64，$[N_{\mathrm{prompt}}+1]$，各响应在拼接序列中的起止边界 |
| `positions` | INT64，$[N_{\mathrm{pool}},2]$，提示编号与序列内位置 |
| `scales_o`、`scales_z` | FP32，$[L,H_q]$，完整候选池上的固定尺度 |

缓存只保存响应、位置与尺度。以 36 层、32 query heads、2571 个边界为例，即使每条响应均达到 24 tokens，以上主要张量总量也约为 439 MB；另有少量元数据与全局提示。

### 2.3 运行时生成 AM 参照

每次 outer AM 从完整 4096 位置池均匀、无放回抽取 128 个位置，各层与 heads 共用索引。Teacher 读取缓存的固定响应，只回放涉及的提示分支，并在各分支最远的选中位置结束，重算这些位置的 $q,o^T,z^T$；无需重新生成响应。Student 使用这些停止梯度的参照与离线固定尺度计算 AM。

下次使用该边界时仍从完整池重新抽样，本次反向与激活重计算复用已确定的参照。尺度始终来自离线完整池，不用本次 128 位置重估。

Teacher 的完整原文 prefix KV 在一个 QA episode 内随压缩边界递增，结束后释放；同文档的其他 QA 或下个 epoch 重新计算。运行时 teacher forcing 使用 `batch_size`，默认每卡 4；离线生成与统计使用 `prepare_batch_size`，默认每卡 64，二者分别设置。

缓存绑定 teacher、tokenizer、输入模板、提示库、窗口边界、候选池、尺度规则与 seed。GPU 数量及两种 teacher 批量只控制执行，不绑定缓存身份；调整时可以复用缓存。不同 attention 内核、GPU 或批量的数值舍入可能影响响应及尺度，不保证重新生成结果逐位一致。同格式、同配置的已完成边界可直接复用。独立 test／benchmark 不加载 teacher 或读取这些缓存。

## 3. 配置

实验输入 JSON 的结构为 `seed`、`model`、`teacher`、`training`，未知字段报错。缺省字段采用 [config.py](config.py) 中的默认值，下表以[正式配置](../../../configs/v4-1/online_ttt_memory.json)为准。路径支持 `~`，相对路径以运行目录为基准，脚本统一切到项目根目录。

Prepare/train 默认读取 `configs/v4-1/online_ttt_memory.json`，分别通过 `--prepare-data`、`--train-data` 覆盖 `training.dataset_dir`；显式 CLI 值在校验前覆盖配置。两者的数据、模型、seed、筛选设置与产物父目录须保持一致。

`run.sh` 先保存计划，再按计划执行。默认在训练后评估最后一个 checkpoint；`--train-only` 只创建训练计划并执行训练和 dev，`--evaluate-only` 读取或创建独立评估计划。首次独立评估须明确指定数据目录，已有计划沿用原设置。具体选项见[运行指南](scripts/README.md)。

### 3.1 Memory 与 inner：`model`

| 字段 | 符号／默认值 | 含义 |
|---|---|---|
| `model_name_or_path` | `~/models/Qwen3-4B-Instruct-2507` | 本地 Qwen3 基座与 tokenizer |
| `backbone_dtype` | `bfloat16` | 离线准备采用 BF16；训练／评估也可用 `float32` |
| `num_slots`／`memory_dim` | $m=256$／$d_m=512$ | 每层 memory 形状 |
| `pending_size`／`recent_size` | $L_p=1024$／$L_r=256$ | 在线窗口容量 |
| `decoder_hidden_dim` | $d_{\mathrm{dec}}=1024$ | SiLU 解码器中间维度 |
| `init_std` | $\sigma_{\mathrm{init}}=0.02$ | $M_0$ 的正态初始化标准差 |
| `inner_steps`／`inner_lr_init` | $S_{\mathrm{in}}=3$／$\eta_{\ell,0}=1$ | 每轮更新步数与初始正步长 |
| `am_std_floor` | $\sigma_{\min}=10^{-4}$ | Inner／outer 固定尺度下限 |
| `lora_rank`／`lora_alpha`／`lora_dropout` | $r=16$／$\alpha=32$／$p=0$ | Q/K/V LoRA；首版固定 dropout 为 0 |
| `gradient_checkpointing` | `true` | 非重入激活重计算 |
| `prefill_chunk_size`／`loss_chunk_size` | 256／256 | 正文前向与答案 CE 的工程分块 |

解码器权重使用 Xavier uniform（gain 1），偏置为 0；LoRA A 使用 Kaiming uniform，B 为 0。层数、KV heads、query heads 与 head 维度沿用基座，不另设超参数。

### 3.2 Outer 参考：`teacher`

| 字段 | 符号／默认值 | 含义 |
|---|---|---|
| `num_prompts` | $N_{\mathrm{prompt}}=480$ | 24 类文档请求 × 20 条响应指令 |
| `response_max_tokens` | $L_{\mathrm{resp}}=24$ | 每条 teacher 响应的新 token 上限 |
| `pool_size` | $N_{\mathrm{pool}}=4096$ | 每边界须填满的固定候选池 |
| `am_sample_size` | $N_{\mathrm{AM}}=128$ | 每次均匀、无放回采样；下次重抽 |
| `prepare_batch_size` | $B_{\mathrm{prepare}}=64$ | 每卡离线生成与统计的批量 |
| `batch_size` | $B_{\mathrm{teacher}}=4$ | 每卡运行时 teacher forcing 批量 |

`prepare --prepare-batch-size-per-gpu N` 覆盖 `teacher.prepare_batch_size`，`--teacher-batch-size-per-gpu N` 覆盖 `teacher.batch_size`；省略时沿用配置。两者不计入训练 batch，也不改变候选池或 AM 采样数量。

### 3.3 Outer 训练与运行：`training`

| 字段 | 默认值 | 含义 |
|---|---|---|
| `dataset_dir` | `data/delta-mem-qasper` | QA 数据目录 |
| `output_root` | `artifacts/v4-1` | 产物父目录；teacher 共享，每个 run 下划分 train 与 eval |
| `run_id` | `null` | 运行目录的末尾标识；`null` 时自动生成 `YYYYMMDD-HHMMSS` |
| `train_only` | `false` | 仅训练与 dev 验证，跳过最终 test |
| `test_data` | `null` | 最终 test 数据目录；`null` 沿用 `dataset_dir` |
| `max_test_samples` | `null` | 最终 test 的 QA 数量上限，默认全量 |
| `max_new_tokens` | 128 | 最终 test 的响应生成上限 |
| `eval_id` | `null` | 最终 test 的计划标识；默认 `auto`，独立评估未指定时使用时间戳 |
| `epochs`／`max_steps` | 4／`null` | 训练轮数／可选 optimizer step 上限；实际配置保存解析后的总步数 |
| `micro_batch_size`／`gradient_accumulation_steps` | 1／8 | 每卡 microbatch／累积次数；全局 batch 还需乘 GPU 数量 |
| `learning_rate`／`lora_learning_rate` | $10^{-4}$／$2\times10^{-5}$ | 新增参数／LoRA 峰值学习率 |
| `weight_decay` | 0.01 | 仅解码器权重矩阵；其他慢参数为 0 |
| `beta1`／`beta2`／`adam_epsilon` | 0.9／0.999／$10^{-8}$ | Outer AdamW |
| `warmup_steps`／`min_lr_ratio` | 50／0.1 | 线性 warmup 后 cosine 降到峰值的 0.1 |
| `gradient_clip`／`lambda_am` | 1／1 | Outer 全局梯度范数上限／outer AM 权重 |
| `dev_every` | 50 | 每 50 步验证；epoch 末和最终步也验证 |
| `save_every`／`save_total_limit` | `after_each_epoch`／4 | 每个 epoch 保存，保留最近 4 个；也可指定正整数步距，最终步必保存 |
| `max_train_data_tokens` | 12288 | Train 原始正文长度上限；`null` 关闭 |
| `max_train_samples`／`max_dev_samples` | `null`／`null` | 过滤后的独立 QA 数量上限 |
| `tracking`／`swanlab_project` | `swanlab`／`latent-working-memory-v4.1` | 默认上传 train/dev；设为 `disabled` 可关闭 |
| `group`／`tags` | `null`／`[]` | Group 默认使用运行目录名；自定义 tags 使用命名空间 |

`teacher_cache_dir` 记录共享缓存路径，`output_dir` 记录 `<output_root>/<方法设置>_<run_id>/train/` 的实际路径；恢复时原计划、checkpoint 和训练目录须一致。

训练参数 `--micro-batch-size-per-gpu` 对应 `training.micro_batch_size`（当前仅支持 1），`--gradient-accumulation-steps` 对应每卡累积次数。有效全局 batch 为“GPU 数量 × 每卡 microbatch × 累积次数”：默认单卡 8、双卡 16；双卡设累积次数为 4 可保持全局 batch 8。Epoch 尾批按实际样本数平均；teacher 批量不计入训练 batch。

实际选样数为 $N$、全局 batch 为 $B$ 时，总步数按 `epochs × ceil(N/B)` 推导，用于 warmup/cosine 并写入实际配置。验证与保存均在 optimizer step 完成后进行，epoch 末及最终步也执行。

命令行覆盖值先保存到计划，执行时记录在实际配置、checkpoint 与 SwanLab config；新建 SwanLab run 另记录 `world_size` 与 `effective_batch_size`，恢复训练不重写原云端配置。CLI `--tracking online` 对应配置中的 `swanlab`，`--tag` 与已有 tags 取并集。公共 `seed=42` 控制初始化、选样与训练顺序，各 rank 独立采样 outer 查询并保存随机状态。

### 3.4 计划与执行记录

| 计划 | 内容 |
|---|---|
| `<运行目录>/plan/config.json` | 公共 seed、模型与 teacher 设置 |
| `<运行目录>/plan/train.json` | 训练设置与后续执行安排 |
| `<运行目录>/plan/eval-<eval-id>/<task>.json` | 任务的数据、生成预算与评估规则 |

完整运行在训练启动前创建自动 test 计划，默认位置为 `plan/eval-auto/test.json`；显式 `eval_id` 替换其中的 `auto`。自动 test 未指定数据时采用训练数据目录，并将最终选择写入计划。计划不绑定本次选中的 checkpoint，补评时可以沿用同一任务计划。

恢复使用 `--resume --checkpoint`，先核对原计划与 checkpoint，再恢复参数、优化器和进度，不接受 CLI 覆盖；GPU 数量和绑定数据须保持一致。Checkpoint 保留完整恢复状态和实际设置快照，计划中的请求预算不随运行改写。训练已完成时仅继续或复用最终 test；训练失败时不启动 test。

独立评估首次创建计划须指定数据目录；已有计划省略参数即沿用，固定设置不一致时报错。RULER 可扩展目标长度，已有结果只补缺失长度。`--evaluate-only` 与 `--train-only`、`--resume` 互斥，不接受训练配置覆盖，评估卡数不受训练卡数限制。

训练保存 `train.log` 与 `status.json`；评估按 step 逐个处理 checkpoint，各卡分片其未完成样本。`eval.log` 汇总带 GPU／分片前缀的输出，任务级 `status.json` 记录 checkpoint 和分片状态。单项失败后继续后续 checkpoint；中断保留逐题完成进度，重试可改变 GPU 数并重新分片剩余样本。SwanLab 仅记录 train/dev，评估不读取云端身份或凭据；dashboard 只需本地 `eval/` 结果。

## 4. 代码与结果

```text
v4_1/
  config.py              # 唯一配置结构与输入校验
  cli.py                 # 公共 CLI 参数与显式覆盖
  data.py                # QA 编码、样本选择、压缩边界
  modeling/              # Memory 解码、局部 AM、Qwen3 接入与在线状态
  teacher/               # 固定提示、离线 token 缓存、运行时 teacher 回放
  training/              # 完整运行调度、元训练、验证、checkpoint
  evaluation/            # 评估计划、样本分片、结果汇总与输出诊断
  dashboard.py           # 从本地评估结果构建独立看板
  scripts/               # 环境与设备入口、可复制命令
```

`training/run.py` 按运行方式调度训练或独立评估；完整运行等待 `training/train.py` 的全部训练进程退出后，再复用独立评估入口计算最终 test。自动评估只选择最后一个 checkpoint，结果续记到原 SwanLab run。

训练的 `qa_loss` 是样本内答案 token 平均 CE，`outer_am` 是完整历史参照下的写入质量，`inner_before`／`inner_after` 是局部更新诊断。Inner loss 下降不能直接说明 QA 有效。正式实验应结合生成得分、逐题回答及资源成本分析。

### 4.1 持久化约定

JSON／JSONL 均使用 UTF-8。训练产物与 SwanLab 记录统一由 rank 0 写入；训练指标每个 optimizer step 一行，逐题评估每个样本一行。模型恢复、实验记录与展示文件按下表区分。

| 产物 | 保存内容与用途 |
|---|---|
| Teacher `identity.json` 与边界 safetensors | 固定参考的权威来源；边界只含 2.2 节列出的五个张量，运行时 $q/o/z$ 按保存的 tokens 重算 |
| `checkpoints/global_step_<N>/state.pt` | 模型参数与完整训练恢复状态；恢复和评估以其中的配置、参数及绑定身份为准 |
| 训练 `config.json`、`data.json` | 实际配置、选样统计与绑定身份的可读快照，不替代 checkpoint |
| 训练 `metrics.jsonl` | 每步 train/dev 指标与资源记录；未执行 dev 的 step 不补值 |
| 训练 `summary.json` | 仅含 `step`、`checkpoint`、`output_dir`，用于定位完成步数与 checkpoint，不保存指标汇总 |
| 训练 `swanlab.json` | SwanLab run 的 `id/url`；启用上传时，恢复及评估续记还需要此文件 |
| 评估任务 `plan.json` | 同一 eval-id 下该任务固定的数据与生成／评分设置，供再次执行或扩展 RULER 长度复用 |
| 评估 `eval.json` | 当前 checkpoint 的评估设置快照 |
| 评估 `summary.json`、`details.jsonl` | 各卡结果按原始样本顺序汇总；所有样本齐全后提交最终结果 |
| 评估 `work.json`、`progress/worker_<rank>.jsonl` | 本轮样本分工与已完成逐题进度；不作为评估完成标记 |

`state.pt` 固定八个顶层字段：`config`、`model`、`optimizer`、`scheduler`、`cursor`、`rng`、`query_rng`、`binding`。`model` 只保存慢参数，不重复保存基座权重或样本临时 memory；`cursor.step` 是已完成的 optimizer step，样本游标为全局游标。`rng` 与 `query_rng` 均为按 rank 排列的列表，单卡长度为 1；恢复时列表长度须与 GPU 数量一致，各 rank 恢复自己的随机状态。

训练与评估 summary 的 `checkpoint` 均为 `state.pt` 的绝对路径。评估另保存 `training_run: {directory, swanlab}`，`directory` 指向实际 `train/` 路径，`swanlab` 为 `null` 或 `{project, id, url}`；续记仍读取该目录中的 `swanlab.json`。

训练与评估分别位于同一运行目录的 `train/` 与 `eval/`，完整布局见[运行指南](scripts/README.md#1-环境与入口)。评估结果为 `<运行目录>/eval/<eval-id>/<task>/global_step_<N>/`，`eval-id` 默认 `YYYYMMDD-HHMMSS`，续跑或复用同一计划时显式指定已有标识；显式指定其他 `output_root` 时，仍保留原运行目录名。RULER 按长度保留子结果，扩展或重试时复用已完成样本与长度；同计划采用调度锁，匹配的完整结果直接复用。所有 checkpoint 完成后，调度器串行向原 SwanLab run 发布结果，不保存终端日志。

Teacher 边界与 checkpoint 先写临时文件，再原子替换正式文件。恢复训练时，`metrics.jsonl` 原子重写为不超过 checkpoint step 的完整记录，仅丢弃未完成末行；其他格式错误直接报错。独立看板读取上述 JSON／JSONL 展示指标与逐题记录，不改写原始答案或评分；启动方式见[运行指南](scripts/README.md#43-评估结果与看板)。

### 4.2 坐标与单位

- **Teacher 位置：**`write_index` 从 1 开始；`positions` 的提示编号和序列内位置从 0 开始，后者相对“提示 + 响应”。`response_offsets`、`processed_end`、`compressed_end` 均使用右端不包含的边界；正文计数包含固定模板。尺度按 `[层数, query_heads]` 保存，head 轴不是 KV heads。
- **训练进度：**`step` 按已完成 optimizer updates 计数，`epoch` 从 0 开始。写入轮次与每轮 inner 步数不充当训练横轴。
- **评估指标：**EM／F1／score 范围为 0–1。Test NLL 按答案 token 加权，EM／F1 按样本平均；benchmark 用完整逐题结果执行官方汇总。
- **评估计时：**时间单位为秒。`resources.seconds` 累计各卡已完成样本的计算时间，不含模型加载；`resources.wall_seconds` 记录本次分工至汇总的墙钟耗时，包含模型加载，重试时仅反映本次执行。`generation_seconds`、`nll_seconds` 按样本求和，前者包含 prefill、inner 写入及 decode。
- **资源统计：**`peak_memory_bytes` 取各卡样本峰值的最大值；训练 `peak_memory_gib` 使用 $2^{30}$ 字节，`step_seconds` 不含随后 dev 和 checkpoint 保存。
- **生成预算：**`requested_max_new_tokens` 是请求上限，`max_new_tokens` 是实际使用上限。Test 遇模型窗口限制时可缩减预算，并以 `model_window` 标记停止原因；`generated_tokens` 是实际生成数量。

具体命令见[运行指南](scripts/README.md)。真实 Qwen3-4B 的训练稳定性、GPU 成本和 benchmark 效果需通过实际运行确认。
