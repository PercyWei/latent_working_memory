# 20261027_v4.1 在线 TTT 记忆

创建时间：20261027 16:34:06 Asia/Shanghai（UTC+08:00）

最后修订时间：20261029 15:18:45 Asia/Shanghai（UTC+08:00）

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

首版支持 full-attention Qwen3、固定 default RoPE、训练 microbatch 1，不启用随机 dropout。GPU 由 `CUDA_VISIBLE_DEVICES` 指定：离线准备按文档分配到可见 GPU，训练与评估使用首张可见卡。Inner 使用 recent 自然读取；outer 使用固定文本提示与 teacher 自然响应。Latent query 生成器与 self-study 保留为研究候选。

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

Train/dev 在同一设备保存独立 teacher 与 student，按边界顺序执行。Teacher 前向使用 SDPA，不建立梯度图，局部 AM 参照以 FP32 计算；其权重及完整原文 prefix KV 与 student 的元训练图共同占用显存。独立 test／benchmark 只加载 student。

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

1. 以完整已到达原文为共同前缀，按提示原顺序分批贪心生成，默认每卡 `batch_size=4`。每条最多 24 个新 token，遇 EOS 独立结束；最后一个 token 仍执行前向以取得其 query，不继续生成新答案。
2. 提示／响应前向使用 FlashInfer，分别计算压缩范围内的原始历史、recent、提示／响应三段 attention，并按各段 log-mass 合并为完整输出。前两段 KV 在批内共享，提示／响应 KV 按序列保存。
3. 直接复用压缩历史段的局部输出与 log-mass，缩为每个 query head 的三项统计：输出通道均值、通道方差及 log-mass。每批结束后，按原提示编号与序列内位置执行 reservoir sampling，均匀、无放回保留 4096 个位置及其统计。相同响应与 seed 得到相同位置池；候选不足即准备失败。
4. 从最终完整候选池的统计量计算 $\sigma_o,\sigma_z$，应用尺度下限，以 FP32 保存。完整 query、局部输出向量与临时统计缓冲均不持久化。

GPU 保存一份共享原文 KV，以及当前提示批的私有 KV 和三项统计，中间向量及时释放。CPU 保留固定大小的候选池：以 4096 位置、36 层、32 query heads 为例，三项 FP64 统计约占 **108 MiB／进程**；这不包含模型权重、KV 和当前批工作区。

离线准备在每张可见 GPU 上启动一个进程，各加载一份冻结 teacher。文档按预计计算量分配，同篇文档的全部边界始终由一个进程处理；原文 prefill 使用 Hugging Face SDPA，每次处理一篇，并跨边界延续 prefix KV。各进程写入同一缓存目录中互不重叠的文档文件，中断后可复用已完成边界。

`training.teacher_cache_dir` 默认 `null`，prepare 与 train 自动使用 `artifacts/v4-1/online-ttt-memory/teacher/p{pending_size}-r{recent_size}`；修改窗口后自动切换目录。训练实际配置保存解析后的路径，完整缓存身份仍由 `identity.json` 校验。

Prepare 的路径优先级为：`--output-dir` > 配置中的非空 `training.teacher_cache_dir` > 自动目录。显式 `--output-dir` 不修改原配置；使用自定义目录时，后续训练须将 `training.teacher_cache_dir` 设为同一路径。

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

Teacher 的完整原文 prefix KV 在一个 QA episode 内随压缩边界递增，结束后释放；同文档的其他 QA 或下个 epoch 重新计算。离线生成与统计、运行时 teacher forcing 使用同一配置字段 `batch_size`，默认每卡批处理大小为 4，可在各自启动时覆盖。

缓存绑定 teacher、tokenizer、输入模板、提示库、窗口边界、候选池、尺度规则与 seed。GPU 数量与 `batch_size` 只控制执行，不绑定缓存身份；调整时可以复用缓存。不同 attention 内核、GPU 或批量的数值舍入可能影响响应及尺度，不保证重新生成结果逐位一致。同格式、同配置的已完成边界可直接复用。独立 test／benchmark 不加载 teacher 或读取这些缓存。

## 3. 配置

唯一 JSON 结构为 `seed`、`model`、`teacher`、`training`，未知字段报错。缺省字段采用 [config.py](config.py) 中的默认值；[正式配置](../../../configs/v4-1/online_ttt_memory.json) 显式列出全部设置。路径支持 `~`，相对路径以运行目录为基准，脚本统一切到项目根目录。

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
| `batch_size` | $B_{\mathrm{teacher}}=4$ | 每卡 teacher 批处理大小，用于离线生成与统计、运行时 teacher forcing |

`prepare --batch-size N` 和 `train --teacher-batch-size N` 均覆盖 `teacher.batch_size`，省略时沿用配置。它控制 teacher 批处理大小，不计入训练 batch，也不改变候选池或 AM 采样数量。

### 3.3 Outer 训练与运行：`training`

| 字段 | 默认值 | 含义 |
|---|---|---|
| `dataset_dir` | `data/delta-mem-qasper` | QA 数据目录 |
| `teacher_cache_dir` | `null` | 按 pending／recent 容量自动确定缓存目录；可显式指定路径 |
| `output_dir` | `null` | 按方法设置与启动日期自动命名，恢复沿用原目录；可[显式指定](scripts/README.md#3-训练与恢复) |
| `max_steps` | $S_{\mathrm{outer}}=1000$ | Outer optimizer steps |
| `micro_batch_size`／`gradient_accumulation_steps` | 1／8 | 单卡全局 batch 为 8；epoch 尾批按实际样本数平均 |
| `learning_rate`／`lora_learning_rate` | $10^{-4}$／$2\times10^{-5}$ | 新增参数／LoRA 峰值学习率 |
| `weight_decay` | 0.01 | 仅解码器权重矩阵；其他慢参数为 0 |
| `beta1`／`beta2`／`adam_epsilon` | 0.9／0.999／$10^{-8}$ | Outer AdamW |
| `warmup_steps`／`min_lr_ratio` | 50／0.1 | 线性 warmup 后 cosine 降到峰值的 0.1 |
| `gradient_clip`／`lambda_am` | 1／1 | Outer 全局梯度范数上限／outer AM 权重 |
| `dev_every`／`save_every`／`save_total_limit` | 50／100／2 | 验证与保存步距，保留最近两个 checkpoint |
| `max_train_data_tokens` | 12288 | Train 原始正文长度上限；`null` 关闭 |
| `max_train_samples`／`max_dev_samples` | `null`／`null` | 过滤后的独立 QA 数量上限 |
| `tracking`／`swanlab_project` | `swanlab`／`latent-working-memory-v4` | 默认上传；设为 `disabled` 可关闭 |
| `group`／`tags` | `online-ttt-memory_20261028`／`[]` | 正式配置显式指定实验系列 group；自定义 tags 使用命名空间 |

训练参数 `--micro-batch-size-per-gpu` 对应 `training.micro_batch_size`（当前仅支持 1），`--gradient-accumulation-steps` 对应 `training.gradient_accumulation_steps`。单卡默认有效 batch 为 $1\times8=8$；命令行覆盖值写入实际配置、checkpoint 与 SwanLab config，恢复时须保持一致，建议读取训练目录的 `config.json`。

公共 `seed=42` 控制初始化、数据选样、训练顺序与 outer 查询采样。Checkpoint 保存实际配置、慢参数、optimizer、scheduler、数据游标、随机状态与数据／缓存身份；基座权重不重复保存。恢复使用同一配置和绑定数据，评估从 checkpoint 读取配置。

## 4. 代码与结果

```text
v4_1/
  config.py              # 唯一配置结构与输入校验
  data.py                # QA 编码、样本选择、压缩边界
  modeling/              # Memory 解码、局部 AM、Qwen3 接入与在线状态
  teacher/               # 固定提示、离线 token 缓存、运行时 teacher 回放
  training/              # 元训练、验证、checkpoint
  evaluation/            # QA／benchmark 评估与 HTML 展示
  scripts/               # 环境与设备入口、可复制命令
```

训练的 `qa_loss` 是样本内答案 token 平均 CE，`outer_am` 是完整历史参照下的写入质量，`inner_before`／`inner_after` 是局部更新诊断。Inner loss 下降不能直接说明 QA 有效。正式实验应结合生成得分、逐题回答及资源成本分析。

### 4.1 持久化约定

JSON／JSONL 均使用 UTF-8。训练指标每个 optimizer step 一行，逐题评估每个样本一行；模型恢复、实验记录与展示文件按下表区分。

| 产物 | 保存内容与用途 |
|---|---|
| Teacher `identity.json` 与边界 safetensors | 固定参考的权威来源；边界只含 2.2 节列出的五个张量，运行时 $q/o/z$ 按保存的 tokens 重算 |
| `checkpoints/global_step_<N>/state.pt` | 模型参数与完整训练恢复状态；恢复和评估以其中的配置、参数及绑定身份为准 |
| 训练 `config.json`、`data.json` | 实际配置、选样统计与绑定身份的可读快照，不替代 checkpoint |
| 训练 `metrics.jsonl` | 每步 train/dev 指标与资源记录；未执行 dev 的 step 不补值 |
| 训练 `summary.json` | 仅含 `step`、`checkpoint`、`output_dir`，用于定位完成步数与 checkpoint，不保存指标汇总 |
| 训练 `swanlab.json` | SwanLab run 的 `id/url`；启用上传时，恢复及评估续记还需要此文件 |
| 评估 `summary.json`、`details.jsonl` | 实验身份、协议、指标汇总与逐题原始输出；部分 details 不表示评估完成 |
| 评估 `report/index.html` | 从上述两个评估文件重建的展示，不是评分或恢复依据 |

`state.pt` 固定八个顶层字段：`config`、`model`、`optimizer`、`scheduler`、`cursor`、`rng`、`query_rng`、`binding`。`model` 只保存慢参数，不重复保存基座权重或样本临时 memory；`cursor.step` 是已完成的 optimizer step。

训练与评估 summary 的 `checkpoint` 均为 `state.pt` 的绝对路径。评估另保存 `training_run: {directory, swanlab}`，其中 `swanlab` 为 `null` 或 `{project, id, url}`；这是身份快照，续记仍读取训练目录中的 `swanlab.json`。

Teacher 边界与 checkpoint 先写临时文件，再原子替换正式文件。恢复训练时，`metrics.jsonl` 原子重写为不超过 checkpoint step 的完整记录，仅丢弃未完成末行；其他格式错误直接报错。报告可独立重建，不改写原始答案或评分。

### 4.2 坐标与单位

- **Teacher 位置：**`write_index` 从 1 开始；`positions` 的提示编号和序列内位置从 0 开始，后者相对“提示 + 响应”。`response_offsets`、`processed_end`、`compressed_end` 均使用右端不包含的边界；正文计数包含固定模板。尺度按 `[层数, query_heads]` 保存，head 轴不是 KV heads。
- **训练进度：**`step` 按已完成 optimizer updates 计数，`epoch` 从 0 开始。写入轮次与每轮 inner 步数不充当训练横轴。
- **指标与资源：**EM／F1／score 范围为 0–1，时间单位为秒；`peak_memory_gib` 使用 $2^{30}$ 字节，`peak_memory_bytes` 使用字节。`generation_seconds` 包含 prefill、inner 写入及 decode；训练 `step_seconds` 不含随后 dev 和 checkpoint 保存。
- **生成预算：**`requested_max_new_tokens` 是请求上限，`max_new_tokens` 是实际使用上限。Test 遇模型窗口限制时可缩减预算，并以 `model_window` 标记停止原因；`generated_tokens` 是实际生成数量。

具体命令见[运行指南](scripts/README.md)。真实 Qwen3-4B 的训练稳定性、GPU 成本和 benchmark 效果需通过实际运行确认。
