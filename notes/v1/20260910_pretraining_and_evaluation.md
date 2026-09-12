# 20260912_预训练与 AE/LM 评估（16:11:15 UTC+08:00）

创建时间：20260910 22:52:20 UTC+08:00
最后修订时间：20260912 16:11:15 UTC+08:00

## 1. 训练过程

预训练阶段学习将单段文本写入记忆，并通过记忆重建原文或续写后续文本。AE 监督信息存储的忠实性，LM 使语言模型适应以压缩记忆为前文的续写。阶段产出是写入与读取模块的参数，供后续连续记忆更新训练初始化。

每个样本包含一次写入和一个 AE 或 LM 目标。记 X 为写入文本，Y 为 X 后紧邻的原文；两类任务独立构造：

| 任务 | 写入内容 | 读取目标 |
|---|---|---|
| AE | X | 重建 X |
| LM | X | 续写 Y |

LM 在数据与指标中的任务名为 `continuation`。

数据来自 FineWeb `sample-10BT`，支持 semantic 完整句界样本和 random 随机截断样本。原始文档固定归属 train/dev/test，各类派生样本继承划分。数据构造与抽查见 [预训练数据构建](20260910_pretraining_data_construction.md)。训练前按配置选择来源比例、任务配额和长度范围，生成每个 run 的样本集；训练器读取准备好的数据。

每个 optimizer step 的流程为：

1. 按当前长度采样概率取一批样本，为每条样本抽取合法记忆容量 K。
2. 按输入、目标与记忆长度组织 microbatch，逐样本从空记忆开始，将完整 X 写入 K 个位置。
3. 根据任务提示，从记忆读取 X 或 Y，计算目标损失并反向传播。
4. 累积本次更新的全部 microbatch 梯度；多卡时同步梯度，再统一裁剪并更新参数。
5. 记录损失、样本与 token 使用量，按间隔进行 dev 评估和 checkpoint 保存。

样本之间独立初始化记忆。一次写入保留完整计算图，读取损失直接监督该次写入。全局 batch 为每卡 `batch_size × gradient_accumulation_steps × world_size`；step、课程进度、评估和保存间隔均按 optimizer 更新次数计量。

## 2. 记忆写入与读取

### 2.1 单次写入

冻结语言模型提取 X 的正文特征，经过可训练投影后加入固定正弦位置编码：

$$
H_X[i]=P_{\mathrm{in}}\bigl(\mathrm{LM}_{\mathrm{frozen}}(X)[i]\bigr)+\mathrm{PE}(i),
\qquad
M_X=\mathrm{Writer}(\varnothing,H_X;K).
$$

语言模型输入包含 BOS，正文局部位置从 1 开始；上式的正文特征位置 i 从 0 开始。每个样本的位置计数重新开始。写入使用关闭读取 LoRA 的语言模型基座。

空记忆包含零个位置；首次写入分配 K 个值为零的位置，加入固定位置编码作为 Writer 的查询。Writer 对文本特征执行注意力与前馈变换，输出记忆值。零值是分配时的数值初态，Writer 内的位置编码与来源类型向量用于区分位置和信息来源。

### 2.2 AE/LM 读取

记忆通过可训练读取投影映射回语言模型隐藏维度。读取序列为：

$$
[\mathrm{BOS};\ P(M_X);\ \mathrm{prompt}_{q};\ Z;\ \mathrm{EOS}],
\qquad
Z=\begin{cases}X,&q=\mathrm{AE},\\Y,&q=\mathrm{LM}.\end{cases}
$$

训练采用 teacher forcing。预测第 j 个目标 token 时，条件包含记忆、任务提示以及真实目标前缀 $Z_{<j}$。损失覆盖完整目标 Z 及 EOS，提示、记忆位置与 padding 均被屏蔽。LM 的监督区间是完整 Y，其压缩前文是 X。

### 2.3 可训练参数

| 模块 | 预训练阶段的作用与状态 |
|---|---|
| 语言模型基座 | 冻结；提取文本特征并执行读取 |
| 写入特征投影 | 训练；将语言模型特征映射到记忆维度 |
| Writer | 训练；从文本特征生成记忆，包含注意力、前馈、归一化、输出投影及来源类型向量 |
| 记忆读取投影 | 训练；将记忆映射为语言模型可读取的输入 |
| 读取 LoRA | 训练；适应记忆条件下的重建与续写 |

静态容量由采样器给定，容量网络在后续容量学习阶段训练。读取时梯度经过冻结语言模型传回记忆，再监督 Writer 和写入投影。

## 3. 样本、长度与容量采样

### 3.1 长度课程

按压缩输入 X 的 token 长度建立样本池，先抽取长度池，再从池内打乱的样本序列依次取样；池耗尽后重新打乱。任务与来源的数量比例由各池内的数据配额决定，单次更新的实际比例随采样变化。

设第 b 个长度池的起始、结束权重为 $w_b^{(0)}$、$w_b^{(1)}$，课程长度为 $S_L$ 步：

$$
u=\min(s/S_L,1),\qquad
w_b(s)=(1-u)w_b^{(0)}+uw_b^{(1)}.
$$

采样时归一化这些权重。起始阶段可以提高短样本占比，之后逐步过渡到目标分布。各长度池独立轮转，因此累计样本次数除以数据集大小表示使用量比例；各池的实际遍历进度由采样状态决定。

### 3.2 多压缩率训练

设 X 长度为 L，名义压缩率为 r，容量计算为：

$$
K=\min\!\left(K_{\max},\max\!\left(K_{\min},\left\lceil L/r\right\rceil\right)\right),
\qquad r_{\mathrm{eff}}=L/K.
$$

当前支持从配置的多个压缩率抽样，预训练采用 r>1，容量上限由 `k_limit` 指定。整数取整与容量约束会使有效压缩率偏离名义值。不同 r 得到相同 K 时合并采样权重；同一文本在不同训练访问中可使用不同容量。

压缩率权重也可按起始与结束分布线性过渡；两者相同时使用固定分布。长度课程控制输入难度，容量采样控制记忆预算，二者分别配置。

### 3.3 上下文预算

样本长度按基座 tokenizer 计算。训练前核对写入预算 `BOS + X`、读取预算 `BOS + K + prompt + target + EOS`；使用完整原文对照时，还需满足 `BOS + X + prompt + target + EOS` 的预算。AE 的 target 为 X，LM 的 target 为 Y。

数据准备的长度上限、运行时输入/目标上限、基座上下文窗口和记忆容量上限分别约束不同部分。实验选择阶段按完整样本筛选，并验证各候选容量及原文对照的预算。

## 4. 损失与参数更新

对任务 q 的第 i 条样本，令 $\widetilde Z_i$ 为含 EOS 的目标，$T_i=|\widetilde Z_i|$：

$$
\ell_i=-\frac{1}{T_i}\sum_{j=1}^{T_i}
\log p_\theta(\widetilde Z_{i,j}\mid M_{X_i},\mathrm{prompt}_q,\widetilde Z_{i,<j}),
\qquad
\mathcal L=\sum_{q\in\{\mathrm{AE},\mathrm{LM}\}}\alpha_q
\frac{1}{N_q}\sum_{i:q_i=q}\ell_i.
$$

$N_q$ 为本次全局更新中任务 q 的样本数，未出现的任务不贡献损失。每条样本先按自身目标 tokens 求均值，再在任务内等权平均，最后乘 `ae_weight` 或 `lm_weight`。任务损失权重与数据数量比例分别配置。

microbatch 和多卡均使用同一全局 $N_q$ 归一化，累积后求和同步梯度。优化器为 AdamW，支持学习率 warmup 后余弦衰减、梯度裁剪、BF16 和可配置的梯度检查点。具体数值由实验配置确定。

checkpoint 保存可训练模型参数、optimizer、step、采样器状态、累计使用量与各进程随机状态。恢复时核对训练配置、数据身份、评估面板与进程数，继续原采样位置和课程进度。

## 5. AE/LM 评估

### 5.1 固定面板与对照条件

从独立文档中固定抽取评估面板，每篇取一个样本，遍历该样本的各个唯一合法容量。不同模型共享面板、任务提示、参考目标和容量设置。初始化、周期 dev 和最终独立 test 使用相应的固定面板。

| 条件 | 读取上下文 | 读取 LoRA | 任务 |
|---|---|---|---|
| `memory` | X 写入后的记忆 + 任务提示 | 启用 | AE、LM |
| `wrong_memory` | 另一篇文档的同容量记忆 + 任务提示 | 启用 | AE、LM |
| `full_context` | 完整 X + 任务提示 | 启用 | AE、LM |
| `base_full_context` | 完整 X + 任务提示 | 关闭 | AE、LM |
| `no_memory` | 任务提示 | 启用 | LM |

同一评估样本的所有条件预测相同目标。完整原文位于任务提示之前；其损失与生成每篇计算一次，再复用于各容量的配对统计。错误记忆来自其他文档，以相同 K 写入，源长度可以不同。

正确与错误记忆的差异用于诊断内容依赖；LM 的空记忆对照衡量前文信息的贡献；完整原文条件衡量压缩后的差距，关闭 LoRA 的原文条件提供基座参照。AE 完整原文条件直接提供待重建文本，用于检查复制和读取路径。

### 5.2 指标

| 指标 | 计算方式 |
|---|---|
| AE/LM NLL、PPL | teacher forcing；NLL 按目标正文 token 数加权，PPL 为其指数；另存包含 EOS 的 NLL |
| AE BLEU-4 | 自回归重建的语料级 SacreBLEU，范围 0–100；分词与平滑签名随结果保存 |
| AE 正确前缀比例 | 从开头连续匹配的正文 token 数除以参考正文长度，再按生成记录平均 |

AE 四种条件均进行自由生成：以记忆或原文和任务提示为初始上下文，之后使用模型自己的输出，greedy decoding 遇 EOS 停止，最多生成参考正文长度加一个 token。生成使用 KV cache；预测文本和参考文本保存在逐条记录中。

NLL/PPL 衡量给定真实目标前缀时的条件预测，自由生成指标衡量独立重建。二者共同用于判断记忆读取适配与忠实还原能力。

### 5.3 汇总与展示

总体结果按测试来源与对照条件汇总；记忆条件另按输入长度 × 有效压缩率联合分组，两者均使用 2 的幂次上界分桶。保留评估次数、目标 token 数、生成次数及实际容量，分组缺失时留空。

SwanLab 的 `charts/*`、`tables/*`、`examples/*` 分别展示图、表和重建样例。单模型总体图按条件分组，测试来源相邻；跨模型比较按训练来源使用不同色系，同一训练来源的不同测试来源使用同色系深浅色。图例写明完整的条件、训练来源或测试来源，表格展示值最多四位小数，原始报告保留精度。

独立评估与比较 run 在媒体 step 0 发布一次，对应训练步数记录在 `checkpoint_step`。已有报告可从 JSONL 重新聚合并发布到新的 run 目录，发布入口与模型推理入口分别使用。

## 6. 配置与代码实现

| 文件 | 功能 |
|---|---|
| [experiment.py](../../src/latent_working_memory/data_preparation/experiment.py) | 按来源比例、长度和任务配额构造 run 样本集及共享评估集 |
| [config.py](../../src/latent_working_memory/v1/config.py)、[sampling.py](../../src/latent_working_memory/v1/sampling.py) | 训练配置、长度池轮转、长度与容量课程 |
| [backbone.py](../../src/latent_working_memory/v1/backbone.py)、[model.py](../../src/latent_working_memory/v1/model.py) | 冻结语言模型、可训练投影、读取 LoRA 与 Writer |
| [train.py](../../src/latent_working_memory/v1/train.py)、[training.py](../../src/latent_working_memory/v1/training.py) | AE/LM 训练、梯度累积与同步、dev、checkpoint 与恢复 |
| [evaluate.py](../../src/latent_working_memory/v1/evaluate.py)、[evaluation.py](../../src/latent_working_memory/v1/evaluation.py) | 独立评估、多容量对照、自由重建和指标聚合 |
| [reporting.py](../../src/latent_working_memory/v1/reporting.py)、[publish_reports.py](../../src/latent_working_memory/v1/publish_reports.py) | 图表、样例与跨模型比较发布 |

训练入口为 `python -m latent_working_memory.v1.train --phase pretrain`，接收配置、数据目录和输出目录；`--evaluation-dirs` 指向名称到评估目录的 JSON 映射，`--resume` 指向恢复 checkpoint。独立评估入口为 `python -m latent_working_memory.v1.evaluate`，指定 checkpoint、评估数据及 split；报告发布入口为 `python -m latent_working_memory.v1.publish_reports`。

训练输出配置、来源记录、逐步指标、dev 报告与 checkpoint；独立评估输出汇总 JSON 和同名逐条 JSONL。产物按 `artifacts/v1/<实验系列>/{train,eval,compare,plan}/` 组织。SwanLab 使用 `latent-working-memory-v1`，同一实验的训练、评估与比较共享显式 group，职责由 job_type 表示。

各实验的数据规模、超参数、运行安排、SwanLab 链接与结果见 [预训练数据类型对比实验](20260911_pretraining_data_comparison.md)。
