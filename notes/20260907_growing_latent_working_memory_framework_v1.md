# 20260907 可增长 Latent Working Memory 框架设计 v1（20260907 21:50:07 CST）

创建时间：20260907 11:30:02 CST（UTC+08:00）

最后修订时间：20260907 21:50:07 CST（UTC+08:00）

状态：第一版设计；M0–M2 以及 M3/P0 的工程链路已经实现，并以标准 Transformers/PEFT tiny Llama 通过 CPU 集成测试。服务器 Llama-2-7B-Chat smoke、16-episode P0 训练、P1–P3 和 GPU 实验尚未完成。当前主线为完整上下文输出蒸馏、学生 writer–reader 联合训练与容量策略交替学习；数值配置是 pilot 起点，不是已验证的最优参数。

## 1. 已达成的研究判断

本设计承接 [prior-art 调研](streaming_mutable_latent_memory_prior_art_2026_09.md) 与 [whole-prefix/streaming gap 实验](experiment_designs/20260903_whole_prefix_streaming_gap_experiment.md)，将后者的固定容量设定扩展为**学习容量增长与历史表示重组**。

研究对象是一组共同编码历史信息的 latent tokens。它们没有预先指定的 session、主题或实体归属；一个事实可以分布在多个位置，一个位置也可以参与多种信息的表达。新输入到来时，模型可以联合修改部分或全部旧状态，并增加 token 数量。

核心问题是：**能否学习一种可增长、可持续更新的表示，使仅访问旧 memory 和当前 chunk 的在线系统，在长期保留、读取质量及资源成本之间，接近重新压缩完整已到达前缀的权衡？**

训练监督与实验对照分开定义：固定完整上下文语言模型 $T$ 直接提供回答目标；各 writer 适配自己的 reader；全量压缩系统仅作为同容量对照单独训练。主训练不依赖一个预先训练好的全量 compressor，也不要求所有方法共享训练后的 reader。

已有讨论得到五点约束：

1. **可读不等于可更新。** 例如，只存均值而不存样本数，足够回答当前均值，却不足以在新数字到来后更新均值。因此，更新器失败可能来自旧表示缺少信息，而不仅是参数未训练好。
2. **扩容应支持旧内容重新分配。** 新位置可以承接旧 latent 中混杂的信息，不仅用于当前 chunk；晚于信息丢失的扩容无法恢复原文细节。
3. **更多 latent 不自动提高计算效率。** 它可能降低读取混淆，但 dense reader 的单次计算量通常增加；必须分别测信息利用、存储、写入和读取成本。
4. **几何形状只能提供候选信号。** 相似度、范数、有效秩和 attention 分散程度都不直接等于事实容量。几何特征应通过实际保留或扩容收益验证。
5. **创新不由单个组件名称成立。** 流式更新、可增长状态、全历史蒸馏和可学习预算各有先例；需要证明其在本问题中的结合解决了可辨认的限制。

| 相关工作 | 与本设计相关的已知机制 | 本设计需要单独验证的区别 |
|---|---|---|
| [C-DIC](https://arxiv.org/html/2606.12411v1) | 检索 thread states，重编码后替换最佳状态或新建状态 | 不使用主题归属与相似度阈值规定写入；允许整个 bank 改变分工 |
| [Chimera](https://arxiv.org/html/2606.21562v1)，2026 预印本 | 完整已到达视觉历史的 bottleneck 监督固定大小 recurrent memory | 语言接口、可增长状态及容量决策；不能将全历史蒸馏本身视为新贡献 |
| [KVM](https://arxiv.org/html/2605.09877v5)，2026 预印本 | 状态按预设计划增长，overflow tokens 追加或合并；作者提出学习扩容阈值作为后续方向 | 基于在线状态学习增长收益，并联合重组旧表示 |
| [ElasticMem](https://arxiv.org/html/2605.30690v1)，2026 预印本 | 对固定离线 bank 中检索到的内容，学习 query-time latent 预算 | 学习的是持续写入时的持久容量，而非已知 query 下的读取预算 |
| [PISCO](https://arxiv.org/html/2501.16075v1)，Findings ACL 2025 | 完整文档 teacher 产生答案，压缩器与 decoder 通过任务蒸馏适配 | 完整上下文蒸馏已有先例；本设计需验证持续更新、学习增长及长期质量—成本收益 |

当前 C-DIC 复现不能充当已验证的强效果基线：[现有评估](reproduction_results/20260906_c_dic_evaluation_alignment_record.md)中，训练后模型在 753 个计分轮次均走 fallback/insert。新框架独立实现，保留该复现及其证据边界。

## 2. 方法概述

本框架将流式记忆建模为可变容量的递归状态，联合学习信息的组织方式与新增容量的使用价值。本节给出运行逻辑和优化目标；后文规定模型参数、训练采样与实验协议。

### 2.1 可增长状态与容量决策

给定顺序到达的输入块 $x_1,\ldots,x_t$，令 $H_t=E(x_t)\in\mathbb R^{c_t\times d}$ 为当前输入特征，$M_{t-1}\in\mathbb R^{K_{t-1}\times d}$ 为历史记忆；$c_t$ 为输入特征数，$K_t$ 为 latent 数量，$d$ 为表示宽度。$E$ 保留源位置，并在不同更新分块下使用相同的基础编码特征。记忆从可学习的初始状态 $M_0$ 开始，各 latent 不预设语义归属。

容量价值网络 $V_\psi$ 根据当前可见状态 $s_t=(M_{t-1},H_t,N_{t-1})$，预测各增长动作相对于不增长的代价差，其中 $N_{t-1}$ 为已处理的源 token 数。一次状态转移为：

$$
\begin{aligned}
g_t&=\arg\min_{g\in\mathcal A_t}[V_\psi(s_t)]_g,\\
K_t&=K_{t-1}+g_t,\\
M_t&=U_\theta(M_{t-1},H_t;K_t).
\end{aligned}
$$

$\mathcal A_t$ 是满足当前资源约束的非负增长动作集合，包含 $g=0$；第一版不收缩。$U_\theta$ 在选定容量下共同重组旧记忆和新信息。在线运行只执行所选动作，不访问未来输入、问题或答案；更新后以 $M_t$ 继续递归，不保留原始历史。

### 2.2 联合更新与记忆读取

更新器以旧 latent 和新增槽位的初值构造输出查询 $Q_t$，以 $Z_t=[M_{t-1};H_t]$ 作为信息来源。忽略类型、位置标记及多头索引时，基本 attention 运算为：

$$
\operatorname{Attn}(A,B)
=\operatorname{softmax}\!\left(
\frac{(AW_Q)(BW_K)^\top}{\sqrt{d_k}}
\right)(BW_V).
$$

$W_Q,W_K,W_V$ 为可学习投影，$d_k$ 为 key 的维度。输出查询先通过 self-attention 协调各位置的表示，再通过对 $Z_t$ 的 cross-attention 读取旧内容与新内容，经前馈变换产生联合修正量 $\Delta_\theta$：

$$
M_t=[M_{t-1};\mathbf 0_{g_t\times d}]
+\Delta_\theta(M_{t-1},H_t,g_t).
$$

所有新旧位置均参与计算并可改变；新增位置也可以承接旧记忆中的信息，而不只保存当前输入。已有状态的直接通路提供保留旧表示的起点，但不保证更新无损。

给定问题 $q$，学生读取器 $R_\rho$ 通过投影 $P_\rho$ 接收全部 memory，并生成答案：

$$
p_S(y\mid q,M_t)
=\prod_{\ell=1}^{|y|}
p_{R_\rho}(y_\ell\mid y_{<\ell},q,P_\rho(M_t)).
$$

投影及 reader 的可训练参数记入 $\rho$，与 writer 参数 $\theta$ 联合学习；不同方法拥有各自的配套 reader。训练期间查询损失可以更新参数，但查询本身不写回 $M_t$；部署时所有参数固定。扩容可能缓解信息混杂，也会增加 dense 读写成本；更新的 attention 配对规模约为 $O(K_t^2+K_t(K_{t-1}+c_t))$。

### 2.3 完整上下文蒸馏与更新一致性

学生 writer 与 reader 共同学习正确回答、接近完整上下文输出及保持分块一致性。记学生参数为 $\eta=(\theta,\rho)$，总体目标为：

$$
\mathcal L_S(\eta)
=\mathcal L_{\mathrm{gold}}
+\lambda_T\mathcal L_{\mathrm{distill}}
+\lambda_P\mathcal L_{\mathrm{partition}}.
$$

**答案监督。** 令 $\ell_t(M;\rho)$ 为读取记忆 $M$ 时正确答案的平均 token 负对数似然，$\mathcal L_{\mathrm{gold}}=\mathbb E_t\ell_t(M_t;\rho)$。问题覆盖旧事实、新事实、纠正和组合，答案以该前缀的真值为准，不将 teacher 输出直接等同于真值。

**完整上下文蒸馏。** 固定 teacher $T$ 直接读取全部已到达原文 $X_{\le t}$ 和问题 $q$；学生 reader 读取在线记忆。对相同的正确答案上文 $y_{<j}$，比较：

$$
\begin{aligned}
p_T^{(j)}&=T(\cdot\mid X_{\le t},q,y_{<j}),\\
p_S^{(j)}&=R_\rho(\cdot\mid M_t,q,y_{<j}),\\
\mathcal L_{\mathrm{distill}}&=\mathbb E_{t,q,y,j}
D_{\mathrm{KL}}(p_T^{(j)}\Vert p_S^{(j)}).
\end{aligned}
$$

Teacher 不产生 latent，也不受学生容量 $K_t$ 限制；训练只对齐输出能力。问题可以进入 teacher 和学生 reader，不能进入 writer 或容量网络。首版采用同词表的 token-level KL；使用 teacher 生成答案的序列级监督作为独立后续对照，不同时引入两套主训练路径。

**分块一致性。** 从同一旧状态出发，以相同顺序分别一次写入 $a,b$ 或分两次写入：

$$
M^{\mathrm{joint}}=U^0(M,[H_a;H_b]),\qquad
M^{\mathrm{seq}}=U^0(U^0(M,H_a),H_b).
$$

$U^0$ 表示不扩容但仍更新记忆。两路径共享该学生当前的 reader，$\mathcal L_{\mathrm{partition}}$ 包含两路径答案损失的均值及读取分布的对称 KL：$\tfrac12[D_{\mathrm{KL}}(p_1\Vert p_2)+D_{\mathrm{KL}}(p_2\Vert p_1)]$，要求两边都答对且表现接近。除当前结束点外，还输入相同后续内容再比较，检查继续更新的能力。该损失共同训练 $\theta,\rho$；不同方法可以有不同 reader，但同一次路径对照不另训 reader。

### 2.4 基于反事实收益的容量学习

容量网络学习在当前 writer–reader 配对能力下，扩容是否值得额外资源消耗。生成标签时固定整个学生 $\eta=(\theta,\rho)$，从同一在线状态 $s_t$ 执行各合法增长动作，得到 $M_t^{(g)}$。各分支接收相同后续输入、暂不再次扩容，并用同一个 reader 回答相同问题；不为每个增长动作重新适配 reader。动作代价为：

$$
J_t(g;\eta)=\overline\ell_t^{(g)}(\rho)
+\lambda_s\overline C_{\mathrm{storage}}^{(g)}
+\lambda_w\overline C_{\mathrm{write}}^{(g)}
+\lambda_r\overline C_{\mathrm{read}}^{(g)}.
$$

其中 $\overline\ell$ 是基于 gold answers 的后续预测损失，三个 $\overline C$ 衡量存储、写入和读取代价；横线表示窗口内平均，各 $\lambda$ 为非负权重。资源项采用归一化代理，与实测成本分开报告。定义 $\Delta J_t(g;\eta)=J_t(g;\eta)-J_t(0;\eta)$：负值表示扩容有净收益。容量网络回归该相对代价：

$$
\mathcal L_V
=\mathbb E_{s_t,g\in\mathcal A_t}
\operatorname{SmoothL1}\!\left(
[V_\psi(s_t)]_g,\ \Delta J_t(g;\eta)
\right).
$$

运行时网络仅依据当前 $s_t$ 选择预测代价最低的动作，无需尝试全部分支。未来输入和答案只用于构造标签；teacher 输出不用于补全候选记忆。有限窗口及后续不扩容使它成为局部价值估计，不保证全局最优。Writer 或 reader 任一更新后都需刷新标签，不能混用不同学生版本的代价。

两个目标采用分阶段与交替训练：先联合训练学生 writer 与 reader，再固定该配对生成标签并训练容量网络；之后用学到的策略继续训练学生配对，并重新评估扩容收益。参数边界见第 7 节，梯度不穿过离散容量选择。

## 3. v1 的明确选择

| 项目 | v1 决定 |
|---|---|
| 持久状态 | `M: [K, 512]` 的 BF16 tensor，加已处理 token 数；没有 raw history 或持久 KV bank |
| 初始容量 | `K_init=16`；由共享可学习初始化函数产生，尚不代表任何历史事实 |
| 增长动作 | `g ∈ {0, 8, 16}`，`K_next=K+g`；首版不收缩 |
| 更新范围 | 所有旧 latent 和新输入共同参与计算，输出全部 `K_next` 个 latent；不设主题路由 |
| 容量策略 | 学习三个动作的相对长期代价，推理时选择代价最低的合法动作 |
| Teacher | 固定完整上下文语言模型 $T$，直接读取原文与问题输出分布；不产生压缩状态 |
| 学生 Reader | Llama-2-7B-Chat 基础权重冻结，memory projection 与 decoder LoRA 随 writer 联合训练；dense 读取全部 latent |
| 全量压缩对照 | 独立训练的 `C_phi + R_G`，可指定输出长度，与在线学生从同一 $T$ 学习；不是主训练前置条件 |
| 学习顺序 | 学生接口初始化/预热 → 在线 writer–reader 联合训练 → 冻结学生生成扩容标签 → 容量策略 → 交替刷新 |
| 几何方法 | v1 记录诊断，不用几何阈值决定增长；transport/敏感性约束是后续定向实验 |
| Pilot 资源保护 | `K_limit=512`，仅为单次实验可执行上限；报告触顶率，不将它当作方法必须固定大小的假设 |

首版选择 dense 更新是为了先确认表示和扩容是否有效。它允许所有位置改变，满足当前研究范围；后续若成本成为瓶颈，再学习稀疏读取/写入。不能将本版描述为已实现按内容选择更新子集，或具有恒定写入成本。

## 4. 时间、状态与输入边界

### 4.1 区分 encoder cell 与 update chunk

同一原文改变 chunk 切分时，预训练 encoder 的可见上下文也会改变。为先识别状态更新本身的路径依赖，v1 使用两个粒度：

- **Encoder cell**：按原始 token 顺序固定划分，每个 cell 最多 64 tokens，独立编码。
- **Update chunk**：由连续 1、2 或 4 个完整 cells 组成；更新器一次处理其全部特征。

同一方法在固定参数下的不同 update partitions 共享完全相同的 cells、特征和源位置编号。编码时每个 cell 加一个标准 BOS，提取 raw token 对应的末层 hidden states，去掉 BOS 对应行。最后一个不足 64 tokens 的 cell 仅在流结束时处理；v1 不引入持久 raw buffer。

这首先检验**更新边界**的影响，不等于任意 encoder offset。全量压缩对照拼接全部已到达 cells，使用相同冻结基础 encoder 的输出，但允许自己的投影与 reader 学习；它与学生匹配前端结构、初始化和训练预算。固定完整上下文 teacher 则直接读取原文，不受 cell 分块限制；它提供任务监督，不用于隔离 encoder 视野效应。

### 4.2 运行时状态

```python
@dataclass
class MemoryState:
    values: Tensor       # [K, d_mem]，首版每次只处理一个 episode
    seen_tokens: int     # 已提交的原始 token 数，不含 cell BOS
```

`K` 从 tensor 推导。slot 地址由行号计算，不另外持久化一份 embedding。episode ID、更新序号、动作、seed 和证据标注属于日志/数据，不进入 memory 内容。

部署时只能将 `MemoryState`、当前 chunk 及固定配置交给 writer。训练器可以持有完整 episode 生成监督；这个权限不能通过 writer 接口泄漏。更新完成后释放该 chunk 的原始 tokens、特征和临时 attention/KV；训练计算图不计入推理持久状态，但要报告训练峰值显存。

对于合成流，先 `write(chunk)`，再在已提交前缀上调用只读 `read(question)`。评估问题及答案不写回 memory。将来用于真实对话时，只有真实到达的对话内容才作为新 stream event；这不属于首版的生成闭环评估。

### 4.3 因果条件

时刻 `t` 的 writer、容量策略及全量压缩对照的 compressor 均不能读取评估问题、答案或未来 chunk。Teacher 和各方法的 reader 可以读取当前评估问题；teacher 的原文视野仅到该 probe 的 `prefix_end`。正确答案上文仅在 teacher forcing 的读出损失中可见，不写入 memory。每个 probe 以对应前缀的真值计分，不能用未来修订后的答案评分较早状态。

## 5. 模型与张量流

### 5.1 固定 teacher、基础前端与学生 reader

首版 teacher 与学生使用相同的 `meta-llama/Llama-2-7b-chat-hf` 基础模型和 tokenizer，便于逐 token 对齐输出；部署位置由配置提供，预期 `d_lm=4096`，实际读取 config 校验。Teacher 保持原始模型权重，学生另有可训练的 reader 适配参数。本版使用标准 causal-LM 接口，不沿用 ICAE/C-DIC 的 FT tokens 或自定义停止规则。

模块定义：

- `T`：固定原始语言模型，输入完整原文前缀、问题和 teacher-forcing 答案上文，输出目标位置分布。
- `E0`：冻结基础模型的 cell encoder，`[c] token IDs → [c, d_lm]`；不启用学生 reader 的 LoRA。
- `W_in`：线性投影 `d_lm → d_mem=512`，与更新器及 memory 初始化参数共同记入 writer 参数 `theta`。
- `P`：线性投影 `d_mem → d_lm`，与 decoder LoRA 共同记入 reader 参数 `rho`；P0/P1/P3 的学生训练阶段均可更新。
- Reader LoRA：首版仅作用于 `q_proj/v_proj`，rank 16、alpha 32、dropout 0，作为 pilot 起点。各比较方法采用相同基础模型、适配结构和初始化规则，但分别训练参数。
- `p_src(i)`、`p_slot(j)`：512 维 sinusoidal 编码；分别表示原始 token 绝对位置和计算用 slot 地址。源位置不会因 chunk 重分组而重置。

新输入特征为 `H = W_in(E0(chunk_cells)) + p_src(source_positions)`，形状 `[c, 512]`。不同 cells 先独立运行 `E0`，再拼接；不能将几个 cells 一起送入 backbone 以改变特征。位置编码先用 FP32 计算，加到 features/queries 前转换为对应 device/dtype。

Teacher 与学生分别使用以下输入，question 部分保持一致：

```text
Teacher: [BOS] + Emb(raw_prefix) + Emb("Question: " + question + "\nAnswer:")
Student: [BOS] + P(M)           + Emb("Question: " + question + "\nAnswer:")
```

训练在其后拼接单独 tokenization 的 `" " + answer` 及标准 EOS，只对 answer/EOS 位置计 gold loss 和 KL。两侧上下文长度不同，按答案相对位置对齐 logits，不按整条输入的绝对位置对齐。`position_ids=0..L-1`，不持久缓存跨 query 的 KV；QA 生成用 greedy、最多 64 tokens，停止 ID 从标准 tokenizer/model config 读取。

Teacher 在 `no_grad` 下产生 `[answer_length, vocab_size]` 的目标 logits，同一 prefix、question、answer 对可在不同容量/更新路径间复用。缓存按 teacher 版本、tokenizer、完整序列化输入和目标 IDs 绑定，不随学生改变而失效，也不得用于不匹配的前缀。训练目标不会进入 writer 或容量价值网络的输入。

`T/E0` 和基础语言模型权重始终冻结；`W_in/U/P/reader LoRA` 在学生训练时保留梯度。即使共享冻结基础权重，teacher 与 `E0` 也必须禁用学生 adapter，并验证其输出不随 reader LoRA 更新改变。训练 reader 的基础权重不更新，但损失经过它传到 memory 和 writer；不能将整个学生 reader 前向放进 `no_grad`。P2 标签生成时才冻结完整学生配对。

### 5.2 全量压缩对照 `C_phi + R_G`

本模块仅用于回答“在线更新是否接近全量压缩”，不作为 teacher，不是启动 P0–P3 的依赖。给定前缀全部 cell 特征 `H_prefix: [n,512]` 和输出数量 `K`：

1. 以 `q_seed + p_slot(j) + W_pool mean(H_prefix)` 初始化 `K` 个 queries。
2. 运行 3 层 resampler block：query self-attention → 对 `H_prefix` 的 cross-attention → FFN；均为 Pre-LayerNorm 残差结构。
3. 线性输出 `G_t(K): [K,512]`，交给其独立训练的投影 `P_G` 与 reader `R_G`。

每层使用 8 heads、FFN hidden size 2048、dropout 0。Self-attention 与 cross-attention 在已到达前缀内全可见。训练输出长度从 `16,24,...,512` 中抽样；较大长度不超过当前 prefix token 数，避免校准阶段把膨胀当作压缩。

对照的 `C_phi/W_in_G/P_G/reader LoRA_G` 使用相同 teacher $T$、gold 数据及蒸馏方式训练；匹配基础前端、reader 适配能力和训练预算，报告额外重读成本。各方法参数独立，不能让该对照共享在线学生训练后的 reader。它不是理论上界；未完成此对照时，只能报告完整上下文表现保持率，不能声称已接近全量压缩。

### 5.3 在线联合更新器 `U_theta`

输入 `M: [K,512]`、当前 `H: [c,512]` 和增长动作 `g`，令 `K'=K+g`。

构造读入源和待更新 queries：

$$
Z=[M+e_{old};\ H+e_{new}]\in\mathbb R^{(K+c)\times d},
$$

$$
Q^{(0)}=[M;\ B_g]+p_{slot}(0:K'),
$$

其中 `B_g` 是 `g` 个新槽位的初值，由共享 `birth_seed + W_birth mean(H) + V_birth mean(M)` 产生；不同新位置由 `p_slot` 区分，`g=0` 时为空。无需为每个新槽位创建新参数。

`Q` 经过 3 层 Pre-LayerNorm 残差 block，每层依次为 self-attention、对全部 `Z` 的 cross-attention 和 FFN；使用 8 heads、FFN hidden size 2048、dropout 0。输出：

$$
M'=[M;\mathbf 0_g]+W_{out}\operatorname{LN}(Q^{(3)}).
$$

`W_out` 使用标准差 `1e-3` 的零均值正态初始化、bias 为零，保留初始旧状态通路但不截断内部模块梯度。新旧全部行均可改变；残差地址不代表语义身份固定。在线 writer 从随机小模块参数及预训练基础模型出发，经 P0 预热后进入 P1；全量对照可复用 block 定义，但不共享训练后的参数。

`M0[j]=init_seed + W_init p_slot(j)`，`j<K_init`，其参数属于 writer。初始化与每次输出都显式转换为 BF16，训练时该转换保留梯度。运行时增长只改变 tensor 行数。`update` 返回新 tensor，不原位修改旧 tensor，便于反事实分支和梯度审计。

按固定宽度忽略投影/FFN 常数，一次更新的 attention 配对规模约为 `K'^2 + K'(K+c)`。写入成本仍需加上 `E0`、投影和 FFN；增长后 dense 更新会变贵。

### 5.4 容量价值网络 `V_psi`

首版先学习动作价值，再做硬选择，避免一开始直接用稀疏长期奖励训练整个系统。

输入向量为：

```text
concat(mean(M), std(M), mean(H), std(H),
       log1p(K), log1p(seen_tokens), log1p(c))  # [4*d_mem + 3] = [2051]
```

统计在 FP32 计算，`std` 使用 `unbiased=False`。网络为 `Linear(2051,256) → GELU → Linear(256,3)`，输出三个动作相对于 `g=0` 的预测代价差。计算容量选择时将输入 detach；writer 的梯度来自记忆任务，不穿过离散 argmin。

推理规则：将超出 `K_limit` 的动作设为非法，选择预测代价最小的合法动作；同分时选择较小 `g`。`g=0` 始终合法，不做隐式裁剪。记录 raw scores、合法动作、所选动作及触顶状态。

加入 mean/std 不意味着按几何阈值扩容；它们只是可学习价值网络的输入。首版不接入原文惊讶度阈值、主题分类器或固定的“每 chunk 增长一次”规则。

Mean/std 不是容量决策的充分统计量。若候选扩容已有稳定收益、但价值网络预测失败，应优先检查这个输入压缩；可在独立 ablation 中换成读取全体 `M/H` 的 learned attention pooling，不直接归因于 writer 或容量设想失效。

## 6. 监督目标与增长标签

### 6.1 学生 writer–reader 的联合目标

总体目标见第 2.3 节。Teacher 与学生使用相同原文截止位置、问题和 gold answer 上文；teacher 不压缩，故不匹配其 latent 容量。Gold answer/EOS 均计入目标 token，token-level KL 的 temperature 为 1，teacher 分支 stop-gradient；损失同时更新 `theta/rho`。分块项见第 6.2 节。

训练损失按 probe 先对 answer tokens 平均，再对 probe 平均；总体 PPL 另按全部目标 token 加权计算，不平均每条 PPL。保留含 EOS/不含 EOS 两组统计。首版 `lambda_T=0.1, lambda_P=0.1`，配置对应 `lambda_distill/lambda_partition`；每 4 个训练 episodes 抽取一个完整 TBPTT 窗口执行辅助分支。Teacher 的错误由独立 gold 评估暴露，不将蒸馏拟合程度当成事实保真度。

默认 pilot 的所有主训练前缀连同问题、答案均须在 teacher 合法窗口内；超长样本不能截断原文后仍称为完整上下文监督。长程评估可只用 gold，但必须标明 teacher/full-text 对照不可用。序列级蒸馏是单独实验选择，不与 token KL 隐式切换。

对被新证据修改的事实使用该 prefix 的新真值；旧值仅在明确询问历史时间的 probe 中作为答案。不能用无条件旧输出一致性保护所有历史回答。

### 6.2 分块一致性的可实现范围

从相同 detached 起始状态取连续 4 个 cells，比较 `[4]` 与 `[2,2]`，交替加入 `[1,1,1,1]`。这组辅助分支内部固定 `g=0`，输出容量相同，先隔离 updater 的路径差异；不强迫自主增长策略在每条路径选择同样的动作。

在相同结束前缀上的 probes 用同一个学生 reader 计算对称 KL，同时保留任务损失防止两边一起退化；该辅助项同时训练 writer 和 reader。额外对两状态输入同样的后续 2 个 cells、仍固定 `g=0`，再比较行为，检验当前等价是否能延续。不足 6 个 cells 时跳过该辅助样本并记录。

完整模型另外自由运行不同 partitions，比较最终行为、容量轨迹与成本；允许不同数量的 latent 表达相近质量，但不能仅靠更细分块带来更多扩容机会获得优势。

### 6.3 从相同在线状态产生扩容标签

训练器固定学生 checkpoint 中的 `W_in/U/P/reader LoRA`，选取其实际 rollout 中的状态 `s=(M,H,N)`；三个动作共享旧 memory、当前输入和配套 reader。冻结的是参数，后续记忆仍正常更新。

1. 对每个合法 `g` 调用 `U(M,H,g)`。
2. 在更新完成时，以及额外到达 2、4 个 cells 后读取 probes。后续统一每次更新 1 cell，强制 `g=0`；各分支使用同一后续序列。
3. 每个计分点抽取相同的 3 个 probes，尽量分别覆盖当前根前缀的仍有效事实、后续新事实、纠正/组合；不足某类时使用其他合法 probe 并记录类别。
4. 所有 probe 的答案按各自计分前缀确定；容量网络不能见到这些 probes、后续 cells 或目标答案。

第 2.4 节的动作代价在本节窗口内计算，使用以下确定性资源代理，同时单独测实际时间：

- `c_s=K_i/K_ref`，在该分支各提交状态平均，`K_ref=64`。
- `c_w=(K_i^2+K_i(K_prev+c_i))/(3*K_ref^2)`，按实际 writer 调用平均；分母对应 `K_prev=K_i=c_i=64`。这只是 attention 配对代理，前端在动作分支之间相同而抵消。
- `c_r=(K_i+q_i+a_i+2)^2/(K_ref+q_i+a_i+2)^2`，按 probe 平均；`q_i` 是完整序列化 question prompt 的 token 数，`a_i` 是不含 EOS 的 answer token 数，额外 2 对应 BOS/EOS。它近似 teacher-forced dense reader 的 attention 规模，不代表部署 wall-clock。

`V_psi` 用 Smooth-L1 拟合 `J_t(g;eta)-J_t(0;eta)`，非法动作不计损失；执行时选择预测代价最小者。标签质量项采用 gold-answer NLL，不额外依赖 teacher KL。初始代价权重为 `lambda_s=0.05, lambda_w=0.01, lambda_r=0.05`，只作为 pilot 预设，不能根据 test 结果调整。扫描代价权重时重新训练对应策略。

这个标签评价的是“本次多分配一些容量，在固定后续不扩容策略下是否有用”，不等于完整最优长期策略。选择具有足够后续 4 cells 的根状态生成标签；根 prefix 的旧事实必须进入未来 probe 抽样。延长 horizon 与放开后续策略属于必要的稳健性实验。

候选状态只能来自在线学生；完整上下文 teacher 不能将原文、答案或 hidden states 注入分支。每个动作使用相同的当前 reader，不额外训练 reader 后再计分。标签属于整个学生 checkpoint；`theta` 或 `rho` 任一改变后均须重新生成，保证比较的是当前配对的实际扩容收益。

## 7. 分阶段训练与梯度边界

学生目标 $\mathcal L_S$ 共同训练 $\eta=(\theta,\rho)$，容量回归目标 $\mathcal L_V$ 训练 $\psi$；teacher $T$ 与基础 encoder $E0$ 始终固定。两类目标分阶段启动、交替优化，不将它们相加后同时更新所有模块。冻结学生指冻结参数；记忆 $M_t$ 仍随输入持续更新。

| 阶段 | 更新参数 | 固定部分 | 产出与推进条件 |
|---|---|---|---|
| P0：学生接口初始化/预热 | 学生 `theta/rho`；不训练 teacher | `T/E0`、基础模型权重 | 单次压缩可读、teacher 协议可靠；无需独立 `C_phi` |
| P1：在线学生联合训练 | `W_in/U`、初始化参数、`P/reader LoRA` | `T/E0`、基础权重 | 自身状态 rollout；随机增长下能利用新增容量 |
| P2a：扩容标签生成 | 无 | 整个学生 `eta` 及固定模型 | 同一在线状态、同一 reader 下的分支代价 |
| P2b：容量价值回归 | `V_psi` | 学生 `eta` 与本轮标签 | 扩容价值预测器；held-out 选择优于简单增长计划 |
| P3：交替刷新 | 固定策略训练学生配对；再固定学生刷新标签训练策略 | `T/E0`、基础权重 | 适应最新学生的能力与状态分布；默认 2 轮 |

### 7.1 完整训练流程

**P0：准备 teacher，预热学生接口。** Teacher 直接采用已有的完整上下文语言模型，先用 gold 问题核对其能力与输入协议，不再训练独立压缩 teacher。默认预热从训练 episode 抽一个前缀与容量 $K$，初始化 $K$ 个 memory slots，将该前缀的 cells 作为一次较长输入交给同一个 $U_\theta$，以不增长方式写入；用 $\mathcal L_{gold}+\lambda_T\mathcal L_{distill}$ 共同训练 writer 和 reader。此阶段没有连续历史，暂不加入 partition 项。产物是学生参数的可读初始化，不是固定监督模型；P1 继续更新这些参数。成熟初始化与可选预训练见第 7.3 节。

**P1：学习在线写入与读取。** 容量网络尚未训练，因此先从探索分布抽取增长动作，执行完整在线递推。每个监督点让固定 $T$ 读取相同截止位置的原文，让学生 reader 读取自身 memory，计算 $\mathcal L_S$ 并共同更新 $\theta,\rho$。学生始终使用自己产生的状态，teacher 不提供用于重置 memory 的 latent。先确认扩容可被当前 writer–reader 配对利用，再训练容量价值网络。

**P2a/P2b：先评估动作，再拟合价值。** 固定 P1 得到的整个学生 $\eta^{(0)}=(\theta^{(0)},\rho^{(0)})$，继续用探索动作运行训练序列并抽取实际状态。按照第 6.3 节构造：

$$
\mathcal D_{cap}^{(0)}=
\left\{\left(s_t,g,\Delta J_t(g;\eta^{(0)})\right)\right\},\qquad
\Delta J_t(g;\eta)=J_t(g;\eta)-J_t(0;\eta).
$$

随后固定数据集，仅用 $\mathcal L_V$ 训练容量网络，得到 $\psi^{(0)}$。回归期间不改变 writer 或 reader，也不通过标签反传到学生。不同增长动作共用同一个当前 reader，不能各自重新训练后再比较；标签绑定完整学生版本。

**P3：使学生与策略交替适应。** 第 $r$ 轮先固定 $\psi^{(r)}$，让它选择增长动作，并沿对应轨迹用 $\mathcal L_S$ 训练 writer–reader；动作视为已确定，不对 argmin 求梯度。得到新学生后固定其参数，用当前策略重新运行序列、评估各动作并更新容量网络：

$$
\begin{aligned}
\eta^{(r+1)}&\leftarrow
\operatorname{Train}_{\mathcal L_S}(\eta^{(r)};\psi^{(r)}),\\
\mathcal D_{cap}^{(r+1)}&\leftarrow
\operatorname{EvaluateBranches}(\eta^{(r+1)},\psi^{(r)}),\\
\psi^{(r+1)}&\leftarrow
\operatorname{Train}_{\mathcal L_V}(\psi^{(r)};\mathcal D_{cap}^{(r+1)}).
\end{aligned}
$$

策略影响状态与容量轨迹；writer 的保存能力和 reader 的利用能力共同决定增长收益。因此任一学生组件更新都需刷新容量标签，teacher 的固定输出缓存则可以继续复用。每轮评估完整的 $(\eta^{(r+1)},\psi^{(r+1)})$，固定迭代次数不代表收敛。部署时所有参数固定，仅执行容量选择、状态更新和读取，不运行 teacher 或分支评估。

### 7.2 采样、优化与梯度范围

P0 每个 step 抽一个完整 cell 前缀、一个从 16 起以 8 递增且不超过 `min(512,prefix_length)` 的容量，以及 3 个 probes；一次写入使用 `initialize_state(K)` 与 `g=0`，teacher 仍读取原文。P1/P3 从默认 `K_init=16` 开始逐 chunk 更新，主损失在每累计 4 个 cells 及最终前缀计算，每点抽 3 个 probes，覆盖旧事实、最近信息和纠正/组合；类别不足时记录替代。

P1 首版随机动作概率为 `{0:0.70,8:0.25,16:0.05}`，非法动作去除后重新归一化；这只是训练探索，不是部署增长规则。新增状态不能未经任务训练就直接用于容量策略比较。

P1/P3 使用自身连续生成的 memory。一次 episode 中 `theta/rho/psi` 保持不变，按每 8 个 encoder cells 划分 TBPTT 窗口；用 1/2/4-cell chunks 对每个 4-cell 区间随机分组，确保监督点和窗口边界均可提交。窗口内累计损失并 backward，之后 detach memory；`theta/rho` 的梯度累积至 episode 结束才共同 optimizer step，`psi` 不参与本轮梯度更新。主损失按实际监督 probe 数缩放，辅助项单独平均。

这是一种按源 cells 划窗的 bounded TBPTT，不是完整长程反传。辅助 partition 分支从窗口起点 detach 后独立构图，两个分支共享当期 reader。Teacher/基础 encoder 始终无梯度；P2 的候选分支则将整个学生放在 `no_grad` 下。冻结 reader 的训练只用于独立控制实验，不是主方案。

可复用的基础特征缓存仅保存无梯度的 `E0` 输出；可训练 `W_in` 的投影按窗口/分支重新构图，不把带梯度的 `H` 缓存跨多个 backward。窗口内主损失与辅助分支损失先合并再反传，避免重复释放共享计算图。学生原始历史只存在于训练驱动器中，不进入部署状态。

初始优化配置：AdamW、`lr=1e-4`、`weight_decay=0.01`，对学生全部可训练参数联合执行 grad norm clip `1.0`；小模块和 reader LoRA 参数/optimizer states 为 FP32，BF16 activations/state，FP32 loss/统计。P0/P1 各最多 2000 steps，P3 每轮学生联合训练最多 500 steps；价值网络每轮最多 100 epochs、batch size 32。先在 dev 上选择配置和配套 checkpoint，再评 test；这些步数仅是 pilot 预算，不保证训练充分。

### 7.3 P0 的成熟方法参考与可选预训练

P0 现在只负责学生的初始化/接口预热；teacher $T$ 从已有语言模型直接取得。可参考成熟压缩训练，也可迁移原生模型作为对照，但不再为主训练额外培养一个压缩 teacher。下表中的预训练指压缩专用预训练，三种方法都以已有预训练语言模型为基础。

| 工作与来源 | 已有训练流程 | 本设计的借鉴范围 |
|---|---|---|
| [ICAE，ICLR 2024](https://arxiv.org/html/2307.06945v4)；[官方代码/权重](https://github.com/getao/icae) | 自编码重建＋续写预训练，再进行指令训练，decoder 固定 | 可选的表示初始化目标与原生读取对照；不要求学生继续冻结 reader |
| [PISCO，Findings ACL 2025](https://arxiv.org/html/2501.16075v1) | 原论文直接用完整文档 teacher 生成答案，联合训练 compressor/decoder，不要求独立压缩预训练 | 支持当前输出蒸馏方向；原文是 sequence-level distillation，不能与首版 token KL 混为同一目标 |
| [GMSA，ACL 2026](https://aclanthology.org/2026.acl-long.1324/)；[官方实现](https://github.com/Twilightaaa/GMSA) | 重建预训练训练 encoder 与对齐层，下游阶段固定 compressor、训练 decoder | 可选的语义对齐结构；不能把其预训练笼统写成“重建＋续写”，也不直接沿用其下游冻结 writer 的策略 |

**可选自监督初始化。** 默认 P0 直接进行 gold＋完整上下文 KL 预热；若要检验重建/续写是否改善学习效率，可先用同一学生做单次压缩：

$$
M_X=U_\theta(M_0^{(K)},E(X);K),\qquad
L_{AE}=-\log p_{R_\rho}(X\mid M_X,q_{reconstruct}),\qquad
L_{cont}=-\log p_{R_\rho}(Y\mid M_X,q_{continue}).
$$

$Y$ 是 $X$ 的后续文本，不进入 writer；提示只交给 reader。该可选阶段同样更新学生参数，之后仍进入 P1 继续联合学习。PISCO 的实验提示重建成绩不必然转化为 QA 收益，因此是否加入该阶段由独立 dev 结果和完整训练成本决定；原文重建所需输出长度另行设置，不能沿用短 QA 的生成上限。

**序列级蒸馏备选。** 固定同一个 $T$ 根据完整前缀 $X$ 和问题 $q$ 产生 $y^T$，再优化 $-\log p_{R_\rho}(y^T\mid M_X,q)$，可减少持久保存完整词表分布的需要。它是相对于 token KL 的独立实验；生成答案须按上下文核验，并保留独立 gold 测试。P0/P1/P3 使用同一 teacher 身份，不在 P1 切换为 compressor teacher。

**公开权重的边界。** 仓库已有 ICAE v1 工程入口，其典型输出为 128 个 4096 维向量，不能直接装入当前可变长 512 维 writer。原生模型可作为读取校准与独立系统对照；若采用其参数初始化，需明确接口适配并验证多容量行为。ICAE v2 的 multi-span concatenation 也不自动等同于历史 latent 联合重组。GMSA 当前官方代码默认 Qwen3，本次未确认可直接迁移到本模型的配套 checkpoint。

[PISCO 当前重构仓库](https://github.com/naver/pisco)使用 connector 和随文档长度变化的 memory token 数，并需要预训练 connector，与论文原实现不同；迁移时固定所用版本及其训练流程。当前 256 个合成 episodes、2000-step P0 仅用于机制 pilot，不构成成熟通用压缩能力的保证。

### 7.4 可更新表示与更长程监督

如果单次压缩可用而长 rollout 退化，优先检查训练状态分布、有限 TBPTT 与第 2.3 节的后续一致性，再考虑更长 continuation 监督或可合成的状态算子。全量压缩对照可用于定位同容量差距，但不重新成为训练依赖。几何引导、一般的 `Merge` 算子和测试时 latent 优化均作为后续独立扩展；主方案继续保留固定输出 teacher、联合学生和容量策略的分工。

## 8. 数据和实验起点

### 8.1 首批数据

首版使用受控英文事实流，与 teacher 和学生基础模型的语言接口一致；研究记录仍以中文为主。准备 train/dev/test 各 256/64/64 个 episodes，每条 512–2048 raw tokens，至少包含：

- 实体—属性—值、数字与随机标识符的精确绑定；
- 同一信息的重复与改述，检查不必要增长；
- 新的独立事实，检查新增容量收益；
- 当前值纠正与明确的历史时间查询；
- 早期事实延迟被使用、后续关系连接两个旧事实；
- 等长度不同信息密度，避免策略只按 token 数增长。

实体和值随机化；模板家族划分与实体/值组合划分独立记录，避免仅换随机 seed 就声称跨任务泛化。数据 seed 为 `20260907`，模型 pilot seed 为 `42`。仅用于探索，稳定后至少补充 3 个模型 seeds。

### 8.2 唯一数据格式

数据采用 JSONL，每行一个 episode。tokenizer 标识及 revision 由实验配置固定；生成器先构造原文，再用该 tokenizer 编译以下记录：

| 字段 | 类型与语义 |
|---|---|
| `schema_version` | `1` |
| `episode_id` | 唯一字符串 |
| `input_ids` | 不含自动 BOS/EOS 的原始完整 token IDs |
| `events` | 事实事件列表；仅训练器/评估器可见 |
| `probes` | 按可计分前缀组织的问题与答案；不进入 writer |

每个 event 固定包含 `event_id, token_start, token_end, entity, relation, value, operation`；span 为左闭右开，`operation` 首版取 `set` 或 `retract`，按到达顺序应用。`set` 替换当前 `(entity,relation)` 的值并记录历史版本；`retract` 的 value 固定为空字符串，移除该键的当前值但保留历史版本。只有 `token_end <= prefix_end` 的完整事件可用于确定该 prefix 的事实真值。

每个 probe 固定包含 `probe_id, prefix_end, question, answer, kind, evidence_event_ids`。`kind` 取 `recall/update/history/compose`；同一个 question 在不同 prefix 可以有不同 answer。所有完整 cell 边界及最终 prefix 都准备 probes；只能在 writer 实际提交到该 prefix 时评分。

离线事件解析器产生当前事实表及历史版本，负责 gold answers；不把 gold topic、事件 ID、事实表或 query 分布标签输入模型。历史问题只能引用原文可见的时间或事件描述，不能引用隐藏的 event ID。若询问尚未建立或已撤销的当前值，答案固定为 `unknown`。`input_ids` 可通过 tokenizer 解码回原文以审计，不再提供另一套竞争的文本输入格式。

### 8.3 校准和最小运行规模

先核对完整上下文 teacher 在 gold 问题上的表现，再用 16 个 train episodes 进行学生单次压缩与读取的过拟合诊断。独立 dev 同时检查 teacher/full-text、各学生的 no-memory 和压缩记忆表现。Teacher 失败的样本单独分层报告，不据此裁决 writer，也不把 teacher 视为理论上界。

首轮容量标签抽取 64 个 train rollout 根状态，最多 3 个动作、4 个后续 cells、3 个计分前缀 × 每点 3 个 probes，即最多 1728 次 reader probe 评估。另从 dev 收集独立根状态，报告预测代价误差和选错动作造成的实际 regret。该规模只验证链路与信号，标签不足时扩大采样，不能直接据此声称学出通用容量策略。

Label roots、候选状态与后续分支使用 train/dev 各自数据；test 只读评估。长程 OOD 从 4096 raw tokens 起，使用同样 64-token cells；若原文加问题/答案超过 teacher 合法窗口，保留 gold 评估并标明 teacher/full-text 不可用，不静默截断或仅用近期原文替代。学生 reader 自身输入也需在合法窗口内。

## 9. 对照、度量与否证条件

### 9.1 对照矩阵

| 条件 | 定义 | 回答的问题 |
|---|---|---|
| `no_memory` | 每个方法用自己的 reader 只读问题，不给 memory | 是否利用记忆而非适配后的模型先验 |
| `full_text_teacher` | 固定 $T$ 读取完整已到达原文和问题 | 监督来源及完整上下文表现参考，不算同容量方法 |
| `global_at_K` | 独立 `C_phi + R_G` 从同一 $T$ 训练，评估 `K=16/32/64/128/256/512` | 同容量全量压缩的实际质量 |
| `fixed_joint` | 固定 K、全部 latent 可更新，适配自己的 reader | 增长是否必要 |
| `scheduled_joint` | 每累计 4 个 cells 增加 8 个 latent，适配自己的 reader | 学习增长是否优于预设计划 |
| `append_only` | 旧行冻结，每 cell 新增 8 行，适配自己的 reader | 简单存储增长能否解释收益 |
| `learned_joint` | 本文完整的 writer–reader 配对与增长策略 | 记忆组织、利用和容量决策的综合收益 |

各方法匹配基础模型、tokenizer、冻结 `E0`、reader 适配结构/初始化规则、训练数据及预算，但 writer 投影和训练后的 reader 各自独立。都可使用同一 $T$ 的 gold＋输出蒸馏目标，在线方法另有对应的路径监督。`append_only` 必须训练自己的配对，不能截取联合 writer 输出当作强基线；它按 1 cell 更新，首个 cell 产生 16 行，此后每 cell 追加 8 行，触顶后停止追加并记录。所有方法在共同前缀评分。

固定计划的增长事件发生在原始 cell 位置；比较 partitions 时让所有路径在这些事件位置结束一个 chunk。扩容动作的可执行时间必须一致，不能为了匹配终点而假定另一个方法早期拥有不存在的容量。

进行三个层面的比较：

1. **主要系统结果**：各 writer 使用自己的 reader，在匹配适配/训练预算下自行分配容量，比较质量—存储—实测读写成本。配套系统更强不能直接归因于 writer 单独保存了更多信息；`global_at_K` 是高重读成本对照，不是主训练 teacher。
2. **受控消融**：回放同一合法增长时间表，比较有无联合重组、`lambda_T=0`、`lambda_P=0`；记录各自 reader 训练条件。冻结历史条件在 `g=0` 时不能保存当前新信息，需披露限制，并保留原生固定比例追加基线。仅匹配终点 K 不等于匹配早期信息机会。
3. **Reader 可学习性**：若要验证“好记忆更易读取”，固定训练好的 writer，从相同 reader 初始化出发，比较不同数据/计算预算下的读出学习曲线。固定 reader 和交叉 reader 仅作辅助诊断；直接互换 reader 失败可能只是编码约定不同，不能单独判定信息丢失。

未经 `global_at_K` 对照，只能报告完整上下文表现保持率及系统效率；不能将对 $T$ 的蒸馏拟合程度解释为已经接近同容量全量压缩。

### 9.2 必须输出的指标

- 事实/纠正/历史/组合 EM、token-weighted NLL/PPL；EM 仅去首尾空白并合并连续空白，不做大小写或数值改写。按事实年龄、更新次数、信息密度分层；本版的改写暴露次数指事实完整到达后经历的全局 writer 调用数，不伪称已定位其具体承载 slot。
- 同一 prefix 不同 partitions 的答案不一致率、分布距离、`K_t` 与成本差异。比较共同提交前缀；测试 update chunk 为 1/2/4 cells，并加入相同长度集合的混合路径。
- 学习增长相对 `g=0` 的真实改善、预测误差、动作选择 regret、无收益增长比例及上限触达率。
- 持久 bytes：`M.numel()*M.element_size()` 加实际保存的 metadata；不以 512 维 token 与 ICAE 4096 维 token 数直接比较容量。
- 容量—源长度曲线及 byte-token area：对提交点 `p_i`，累计 `B(M_i)*(p_{i+1}-p_i)`，另报告终点 bytes。它是对原始流长度的积分，不是 wall-clock byte-seconds。
- 实测 `encode/write/read` 时间、生成 TTFT、峰值显存、累计写入时间；CUDA 计时含同步，预热后同设备同负载测量。策略代理成本不能替代这些指标。
- 每种方法的 reader 可训练参数、适配数据和训练计算；teacher 目标生成、P0 预热与反事实标签生成成本单列，复用只计一次。Teacher 不属于部署推理路径，其成本不能混入部署 latency，也不能在训练预算中消失。

dense reader 读取所有 `K_t` 个 latents，首版不预设会因增长变快。若只提高质量、同时增加计算，就报告质量—成本权衡；后续学习检索才可能改变这条关系。

### 9.3 判定与停止方向

- 完整上下文 teacher 的答案不可靠：先检查任务与监督来源，不将 teacher 当成真值。
- 学生单次压缩后仍不能使用 memory：先处理学生接口/预热，不急于归因于递归更新。
- 给定更大容量仍无改善：优先审计 writer、reader 或表示；不要先训练更复杂容量策略。
- 相同学生单次写入有效，自身多轮 rollout 失效：定位误差累积、读取适配和训练状态分布。
- 固定增长计划达到相同质量—成本：暂不支持学习容量策略有价值。
- 追加式存储达到相同质量—成本：暂不支持联合重组的必要性。
- 优势仅来自更早/更多容量或更强 reader 训练：收缩系统收益的解释，不单独归因于重组。
- 不同 chunk 切分导致大量无收益增长：改进容量监督和长度泛化，不宣称 partition consistency。

首轮只判断信号是否存在；正式效应阈值在 pilot 后固定。主实验采用 episode 成对 bootstrap，并跨模型 seeds 检查稳定性。

## 10. 几何引导的预留位置

v1 将更新器最后一层 cross-attention 的来源分配、latent 范数/有效秩、少量 probe 的 slot 删除影响作为离线诊断。多 slot 分布式编码存在时，单 slot probe 失败不能证明信息未存储；谱或 attention 指标改善也不能单独证明容量利用改善。

若诊断能预测真实扩容收益，再逐项加入：

1. **学习 transport 分配**：在旧 memory 与新特征之间形成来源—输出分配矩阵，随 `K_next` 改变输出容量；用覆盖/容量约束引导重新分工。[ComprExIT](https://arxiv.org/html/2602.03784v4)提供静态分配先例，但其固定分段传输不验证多轮 latent 重分配，传输质量也不是信息量。
2. **Reader 敏感性约束**：在标注的未受影响 probes 上限制状态更新引起的输出改变；比较时固定同一个当前 reader，纠正内容不参与旧答案保护。Reader 参数改变后重新估计敏感性，梯度冲突不能手工等同于容量不足。
3. **可分析的统计草图**：学习特征后用 [Frequent Directions](https://arxiv.org/abs/1501.01711) 一类操作维护状态。协方差误差界不直接保证实体绑定与事实保真，需独立任务验证。

这三项不与首版主干同时启用，以免无法区分收益来自哪里。

## 11. 代码落点、接口和持久化

以下是目标实现位置；截至本次修订，配置、数据/状态、纯张量模型、目标函数、feature rollout、容量成本、基础指标、checkpoint、数据生成入口，以及实际 Transformers/PEFT backbone 和 P0 训练入口已经创建。P1–P3、baselines 与完整 evaluate 入口仍待对应里程碑实现。v1 作为完整的纵向切片放在独立目录中；未来版本使用同级 `v2/`、`v3/`，不覆盖 v1 的代码、配置、测试或产物。当前不预建后续版本目录，也不提前抽取跨版本兼容层；只有多个已实现版本确认共享稳定契约后，才把相同逻辑提升到版本目录之外。详细实施顺序见 [v1 实施总计划](v1/20260907_growing_latent_working_memory_implementation_plan.md)。

```text
src/latent_working_memory/
  v1/
    config.py          # v1 配置加载与严格契约
    data.py            # canonical episode、cell partition、probe/事件解析
    backbone.py        # 固定 T/E0、学生 reader/投影及输出位置对齐
    model.py           # joint updater、初始化与 growth value network
    baselines.py       # 独立全量压缩系统与追加式对照
    state.py           # MemoryState、动作合法性
    objectives.py      # gold、token KL 与 partition loss
    rollout.py         # 在线状态递推与只读 probe
    training.py        # 学生 P0/P1/P3 与联合梯度
    capacity.py        # 固定学生配对的反事实评估、代价回归
    evaluation.py      # paired partitions、增长轨迹、质量/成本指标
    checkpoint.py      # v1 模型与 runtime memory 的唯一持久化格式
    prepare_data.py    # 合成 episode 生成入口
    train.py           # phase 参数；单一 JSON 配置入口
    evaluate.py        # 方法与 checkpoint 评估入口
configs/v1/pilot.json
tests/v1/test_*.py
notes/v1/
```

核心签名与返回约定：

```python
frozen_cell_encoding(cells) -> CellEncoding                     # [c,d_lm]，无梯度
project_cell_encoding(cell_encoding) -> Tensor                  # [c,512]
teacher_output(prefix_input_ids, qa_tokens) -> ReaderOutput      # [L,V]
initialize_state(num_slots) -> MemoryState
predict_growth_costs(state, features) -> Tensor                 # [3]
choose_growth(costs, num_slots, slot_limit) -> int               # 0/8/16
update_memory(state, features, grow_by) -> MemoryState
student_output(memory, qa_tokens) -> ReaderOutput
build_capacity_targets(state, features, continuation, probes) -> CapacityTarget
```

`teacher_output` 只供训练器使用，返回答案相对位置上的完整词表 logits；`frozen_cell_encoding` 与可训练的 `project_cell_encoding` 分开，使后续阶段只能缓存无梯度的 `E0` 输出。`compress_prefix(prefix_features,num_slots)` 仅属于全量压缩对照。主学生训练与部署都不调用 `C_phi`。模型组件持有各自参数；容量评估器绑定同一个完整学生 checkpoint 后才展开分支。

部署主循环只有以下数据通路；`evaluation_queries` 由外部驱动器提供，不传入容量网络或 writer：

```python
state = initialize_state(config.k_init)
for cells in incoming_chunks:
    features = encode_cells(cells, state.seen_tokens)
    costs = predict_growth_costs(state, features)
    grow_by = choose_growth(costs, state.values.shape[0], config.k_limit)
    state = update_memory(state, features, grow_by)
    # 在已提交前缀上可进行只读查询；随后释放 cells/features。
```

`update_memory` 将 `seen_tokens` 增加 `features.shape[0]`；encoder 与调用方确保 source_start 等于旧 seen_tokens。接口拒绝空 chunk、错误宽度/非有限状态、非法增长动作及超出 reader 合法位置范围的输入，不做静默截断。`ReaderOutput` 至少含目标 token NLL、用于 KL 的目标位置 logits 和目标长度；生成接口另输出 token IDs。

容量标签 JSONL 的每行固定为 `schema_version, framework_version, root_id, episode_id, prefix_start, prefix_end, student_checkpoint, rollout_trace_path, policy_features, legal_actions, costs, target_deltas, continuation_end, probe_ids`。`framework_version` 在本目录中固定为 `v1`；`student_checkpoint` 同时绑定 `theta/rho`，`policy_features` 为 2051 个 FP32 值；动作数组按 `[0,8,16]` 排列，非法 cost/target 为 `null`。`costs` 保存 gold-answer NLL、三项代理及总值；价值网络不读取 probe 文本、teacher 输出或未来输入。Trace 用于复查根状态与当时的增长策略。

以下为 `configs/v1/pilot.json` 的核心字段示例；实现时从这些字段构造配置，正文规定的其余训练默认值也写入 resolved config：

```json
{
  "schema_version": 1,
  "framework_version": "v1",
  "model_name_or_path": "/data/bywei/models/meta-llama/Llama-2-7b-chat-hf",
  "model_revision": null,
  "teacher_model_name_or_path": "/data/bywei/models/meta-llama/Llama-2-7b-chat-hf",
  "teacher_model_revision": null,
  "distill_mode": "token_kl",
  "distill_temperature": 1.0,
  "reader_lora_rank": 16,
  "reader_lora_alpha": 32,
  "reader_lora_target_modules": ["q_proj", "v_proj"],
  "reader_lora_dropout": 0.0,
  "d_mem": 512,
  "num_layers": 3,
  "num_heads": 8,
  "ffn_dim": 2048,
  "cell_tokens": 64,
  "update_chunk_cells": [1, 2, 4],
  "k_init": 16,
  "k_limit": 512,
  "growth_actions": [0, 8, 16],
  "exploration_probs": [0.70, 0.25, 0.05],
  "data_seed": 20260907,
  "model_seed": 42,
  "train_episodes": 256,
  "dev_episodes": 64,
  "test_episodes": 64,
  "min_episode_tokens": 512,
  "max_episode_tokens": 2048,
  "learning_rate": 0.0001,
  "weight_decay": 0.01,
  "gradient_clip": 1.0,
  "bptt_cells": 8,
  "supervise_every_cells": 4,
  "probes_per_prefix": 3,
  "lambda_distill": 0.1,
  "lambda_partition": 0.1,
  "partition_every_episodes": 4,
  "capacity_label_roots": 64,
  "capacity_horizon_cells": 4,
  "capacity_score_offsets": [0, 2, 4],
  "cost_reference_slots": 64,
  "state_cost_weight": 0.05,
  "write_cost_weight": 0.01,
  "read_cost_weight": 0.05,
  "outer_rounds": 2,
  "max_new_tokens": 64
}
```

启动入口额外接收 `phase`、数据目录与 output directory；恢复时匹配 checkpoint 的 resolved config，并固定 teacher/tokenizer 的实际来源与版本。首版仅实现显式的 `token_kl`，不在输出词表不匹配时自动切换蒸馏方式。上述模型位置来自已有仓库配置，尚未在本设计任务中连接服务器确认，实际加载失败直接报告。

实现时一次一个 episode，不先引入 ragged batching 或多模型兼容层。PyTorch/Transformers/PEFT 为明确硬依赖，在模块顶部导入；实现阶段统一用 `uv` 声明并锁定，保持既有 reproduction 环境独立。普通函数不额外强制 keyword-only 参数。

配置只使用 JSON；episode/metrics/labels 只使用 JSONL；模型和 runtime memory 只使用一个明确的 `.pt` dict schema，不预建多个 fallback 格式。

模型 checkpoint 固定保存 `schema_version, framework_version, phase, config, model_state, optimizer_state, progress, rng_state`；`framework_version` 固定为 `v1`，加载器拒绝跨版本恢复，不设置兼容 fallback。`model_state` 包含 `W_in/U`、初始化、`P/reader LoRA` 和容量网络，固定 teacher/backbone 由 config 定位。P1/P3 仅在 episode 边界保存，记录 epoch、shuffle 顺序、下一 episode 及外层迭代位置；P0/P2b 记录各自训练游标。恢复全部 RNG 与优化器状态，P2 标签必须匹配产生它的完整学生版本。

运行时 memory checkpoint 固定保存 `schema_version, framework_version, model_checkpoint, values, seen_tokens`，写入前 detach，恢复时检查版本、宽度、dtype、长度、源位置及完整配对模型关联。不能在不记录变更的情况下替换 reader 或增长策略后继续同一运行；该文件不包含 teacher、原始历史或监督目标。

所有产物位于项目现有 Git-ignored 根目录：`data/v1/`、`checkpoints/v1/`、`artifacts/v1/<run_id>/`。记录 resolved config、模型初始化、数据 split、seed、时间、逐步 trace 与配对结果。跨版本比较只读取各版本导出的 predictions、metrics 与 resource usage，不直接加载另一个版本的内部 state。使用服务器时仅使用物理 GPU 0/1；第一阶段优先单卡，其余卡可运行独立对照，不默认引入分布式训练。

## 12. 实现验收顺序

1. **数据与状态**：cell 分组不改变 token 顺序/共享特征；prefix 真值正确；read 不修改状态；非法动作拒绝或 mask；保存恢复后的下一步一致。
2. **模型张量链路**：测试 `g=0/8/16` 的形状、`seen_tokens`、旧状态无原位修改；新位置能依赖旧 memory，旧位置能依赖当前输入；不能仅以 shape 正确代替语义实验。
3. **监督与梯度**：teacher 只见合法完整前缀，teacher/student 按答案相对位置对齐；学生 loss 到达 `W_in/U/P/reader LoRA`，`T/E0` 无梯度且输出不受学生 adapter 更新影响；TBPTT 与离散动作边界正确。
4. **P0 接口预热**：16-episode 单次压缩诊断与独立 dev 检查，验证学生可使用 memory；主流程在没有 `C_phi` checkpoint 时仍可训练。
5. **在线与扩容能力**：学生自身 rollout、固定容量更新与长程/partition 差距；同根分支固定整个配对，确认至少部分扩容动作有可重复收益。
6. **策略与系统评估**：writer 或 reader 改变都使旧容量标签失效；匹配各方法的 reader 训练预算，并独立完成全量压缩对照后再判断原始 gap。

文档 v1 固定的是可落地的模型和审计接口。全局可学习更新、增长策略、几何引导、可合成表示分别是否有效，仍需上述实验回答。
