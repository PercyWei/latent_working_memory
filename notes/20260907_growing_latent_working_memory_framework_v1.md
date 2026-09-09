# 20260909_可增长 Latent Working Memory 框架 v1（20:13:30 UTC+08:00）

创建时间：20260907 11:30:02 UTC+08:00

最后修订时间：20260909 20:13:30 UTC+08:00

本文定义语言模型基座上的可增长工作记忆、推理计算、三阶段训练和实验协议。模型参数在训练中学习，记忆状态在推理中随输入持续更新。工程接入与验证范围见第 11、12 节。

## 1. 研究目标与术语

研究对象是一组共同保存历史信息的 latent vectors（潜在向量）。系统在有限存储与读写计算下持续接收新信息，维护已有记忆，并为后续任务提供可用的历史内容。记忆位置采用分布式表示：一个事实可以涉及多个位置，一个位置可以参与表达多种信息。

核心研究问题是：仅访问已有记忆和当前输入的在线系统，能否通过联合更新与容量增长，在长期保持、任务质量及资源成本之间取得接近完整历史重读的权衡。

| 术语 | 定义 |
|---|---|
| 语言模型基座 $B$ | 提供文本处理、内部表示和语言生成能力的预训练模型 |
| 记忆 $M$ | 持久保存的 $K$ 个潜在向量，宽度为 $d_M$ |
| 写入事件 | 已经到达、按顺序提交给记忆系统的一段文本；对话中对应已完成的一轮交互 |
| 写入单元 $x_t$ | 基座完整处理并由更新器一次写入的变长文本；由段落、连续句子组或实际 turn 构成 |
| 预训练容量 $K_X$ | 从空记忆写入片段 $X$ 时，由训练采样器提供的位置数 |
| 目标数量压缩率 $r$ | 从输入长度计算候选容量的比例；取整和容量约束后的实际比例为 $r_{\mathrm{eff}}=L_X/K_X$ |
| 读取请求 $q$ | 当前任务提示、问题或对话发言，交给基座的读取路径 |
| 记忆读写参数 $\eta$ | 写入投影、联合更新器、读取投影和读取 LoRA 的参数 |
| 容量参数 $\psi$ | 预测各增长动作价值的网络参数 |

本框架以记忆的写入、保持、更新、重组、扩容和读取组织方法。压缩是将输入组织为有限记忆位置的技术机制。持久状态增长与模型参数学习分别由运行时计算和离线训练承担。

相关工作提供以下方法背景：

| 工作 | 已有机制 | 本框架的实验重点 |
|---|---|---|
| [ICAE，ICLR 2024](https://arxiv.org/html/2307.06945v4) | 基座加 LoRA 产生记忆，通过 AE 与续写学习可读表示 | 记忆读写预训练以及持续更新后的保持 |
| [C-DIC，2026 预印本](https://arxiv.org/html/2606.12411v1) | 从 ICAE 初始化，在 MSC 上以回复损失和 retrieval-aware TBPTT 适配增量记忆 | 全体记忆的联合重组及基于任务收益的增长 |
| [ComprExIT，2026 预印本](https://arxiv.org/html/2602.03784v4) | 从固定基座内部表示中聚合信息，再通过适配模块形成记忆 | 独立记忆模块的学习与信息利用 |
| [PCC，ACL 2025](https://aclanthology.org/2025.acl-long.1394.pdf) | 原文重建、续写预训练和任务适配，使用 converter 接入基座 | 记忆读写接口及其预训练 |
| [GMSA，ACL 2026](https://aclanthology.org/2026.acl-long.1324.pdf) | 分组聚合与 Layer Semantic Alignment | 记忆表示与基座读取能力的适配 |
| [Activation Beacon，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/fc797c61eb18f7b3ce9a74af0ef0d876-Paper-Conference.pdf) | 在训练中随机采样压缩率，学习多种 KV 压缩配置 | 同一读写模型在多种容量预算下的适应能力 |
| [Sentence-Anchored Gist Compression，2025 预印本](https://arxiv.org/html/2511.08128v1) | 在句末布置压缩位置，利用自然语言边界组织信息 | 以自然边界构造写入单元的实验依据 |

项目背景见 [prior-art 调研](streaming_mutable_latent_memory_prior_art_2026_09.md) 和 [历史前缀重读与流式更新实验](experiment_designs/20260903_whole_prefix_streaming_gap_experiment.md)。文献训练配方见 [真实公开数据与记忆训练路线](20260907_real_data_training_route_review.md)，本版数据与训练协议以本文为准。

## 2. 语言模型基座与新增模块

首版基座为 `meta-llama/Llama-2-7b-chat-hf`，隐藏维度 $d_B=4096$；记忆宽度 $d_M=512$。基座权重冻结，同一套权重承担两类调用：

- 写入前向处理当前文本，读取 LoRA 处于关闭状态。
- 读取前向接收记忆和当前请求，读取 LoRA 处于启用状态。

新增的可训练部分为五项：

| 模块 | 结构 | 职责 |
|---|---|---|
| 写入投影 $W_{\mathrm{in}}$ | $4096\rightarrow512$ 线性层，含 bias | 将当前文本的基座表示映射到记忆空间 |
| 联合更新器 $U_\theta$ | 3 层 self-attention、cross-attention、FFN，配套归一化、类型向量和输出投影 | 写入新信息，保持并重组旧记忆 |
| 读取投影 $P_\rho$ | $512\rightarrow4096$ 线性层，含 bias | 将记忆接入基座的输入空间 |
| 读取 LoRA | `q_proj/v_proj`，rank 16，alpha 32，dropout 0 | 学习利用记忆完成重建、续写和下游任务 |
| 容量网络 $V_\psi$ | `Linear(2051,256) → GELU → Linear(256,3)` | 预测增长动作的质量与资源代价 |

记写入投影和更新器参数为 $\theta$，读取投影和读取 LoRA 参数为 $\rho$，完整读写参数为 $\eta=(\theta,\rho)$。更新器内的 source 类型向量、LayerNorm、attention、FFN 和输出层均计入 $\theta$。

零填充、源位置向量、记忆位置向量及首次容量分配规则属于确定性操作。首次写入与后续增长共同采用零内容初值加固定位置向量。

512 维记忆减少持久存储和独立更新器的宽度。两个投影承担 $4096\rightarrow512\rightarrow4096$ 的接口转换；读取阶段的基座仍处理 $K$ 个 4096 维记忆输入。按 BF16 存储，每个 512 维记忆位置占 1 KiB。记忆宽度和位置数共同决定容量，跨方法比较采用实际 bytes。

## 3. 记忆状态、分块与因果边界

### 3.1 空状态与容量

运行时状态为：

```python
@dataclass
class MemoryState:
    values: Tensor       # [K, 512]，BF16
    seen_tokens: int     # 已提交的原始 token 数
```

初始状态：

$$
M_0\in\mathbb R^{0\times d_M},\qquad N_0=0.
$$

首次写入通过外部参数 $K_{\mathrm{first}}$ 分配位置。预训练中，该参数取当前样本的容量 $K_X$；动态训练与推理的 pilot 起点为 16。已有记忆上的增长动作是 $\{0,8,16\}$，容量上限为 $K_{\max}=512$。

新任务创建空状态，同一任务的后续轮次与 session 沿用已有状态。动态阶段的 $K_t$ 保存累计历史，增长决策依据旧记忆与当前输入。已有状态的恢复绑定产生它的模型 checkpoint。

### 3.2 自然边界与变长写入单元

预训练的写入单元为完整段落、连续句子组、单句或相邻段落组合。文本内容先按自然边界确定，随后计算实际 token 数。动态对话将已完成的一轮交互序列化为写入事件，在合法上下文预算内作为一个写入单元；推理沿用实际到达的事件边界。

基座对整个 $x_t$ 加标准 BOS 后执行一次前向，提取正文 tokens 对应的末层表示。基座在该单元内使用完整的因果上下文，更新器一次接收其全部表示。源位置按累计写入 tokens 编号。批处理按实际长度组织 padding，attention 与 loss mask 对应有效文本和有效记忆位置。

超出写入上下文预算的事件按原文顺序沿段落、句子边界形成多个合法单元，并依次提交。写入单元的长度上限由基座写入窗口与特殊 tokens 的预算确定；预训练样本另外检查 AE、LM 的完整提示与目标读取预算。来源 adapter 保留事件与内部单元的对应关系。

自然边界的改变同时改变基座上下文和记忆更新次数。切分对照共享原始文本、源位置、读取边界及容量安排，各路径重新计算对应单元的基座表示，比较最终任务表现。

### 3.3 写入与读取的边界

写入路径的输入为旧记忆、当前到达文本及固定配置；容量网络读取旧记忆、当前输入表示和长度统计。读取请求与目标序列由训练器或任务驱动器交给读取路径。

读取调用保持记忆状态原值。文档流按来源顺序提交文本，在指定前缀读取。对话按以下顺序运行：

1. 基于历史记忆和当前发言生成回复。
2. 将已经完成的当前发言与回复序列化为写入事件。
3. 提交该事件，得到下一轮使用的记忆。

训练中的历史对话使用数据集记录的人工回复；闭环推理将实际生成的回复写入后续历史。外部 QA 诊断的请求和答案仅用于读取与计分。源文档、证据 spans、参考答案和未来文本由训练驱动器管理，其权限通过上述输入边界约束。

## 4. 推理执行顺序

一次完整的写入和读取包含以下步骤：

| 步骤 | 输入 | 操作与产物 |
|---|---|---|
| 状态准备 | 任务标识、模型配置或已保存状态 | 创建空记忆或恢复已有记忆 |
| 当前文本处理 | 当前变长写入单元 | 基座前向、写入投影与源位置相加，得到 $H_t$ |
| 容量选择 | $M_{t-1},H_t,N_{t-1}$ | 首次按配置分配位置，推理 pilot 为 16；后续选择合法增长动作 |
| 位置准备 | 旧记忆、增长数量 | 为新增位置补零，为输出位置加入固定位置向量 |
| 联合更新 | 全部旧记忆、当前表示、输出位置 | 产生全部 $K_t$ 个新记忆向量 |
| 状态提交 | $M_t,c_t$ | 保存 $M_t,N_t$，释放当前文本和临时表示 |
| 记忆读取 | 已提交记忆、当前请求 | 读取投影接入启用 LoRA 的基座，生成输出 |

连续输入可以经历多次写入后再读取。持久状态为记忆张量和已处理长度；attention 中间结果与生成过程中的临时状态按调用管理。部署执行单个选定增长动作的前向计算。模型参数在整段推理期间固定。

## 5. 模块计算公式

### 5.1 基座表示与写入投影

令 $t$ 为写入计数，变长单元 $x_t$ 含 $c_t$ 个原始 tokens。$\operatorname{Hidden}_B$ 表示基座完整处理该单元后，正文 tokens 对应的末层表示：

$$
h_t=\operatorname{Hidden}_B(x_t)\in\mathbb R^{c_t\times d_B},
$$

$$
H_t=W_{\mathrm{in}}(h_t)
+P_{\mathrm{src}}(N_{t-1}:N_{t-1}+c_t)
\in\mathbb R^{c_t\times d_M}.
$$

$P_{\mathrm{src}}$ 和后文的 $P_{\mathrm{slot}}$ 为固定 sinusoidal 位置向量。位置计算使用 FP32，相加时转换到对应计算 dtype。

### 5.2 容量决策

对于 $K_{t-1}>0$，容量网络输入为：

$$
s_t=\operatorname{concat}\left(
\mu(M_{t-1}),\sigma(M_{t-1}),
\mu(H_t),\sigma(H_t),
\log(1+K_{t-1}),\log(1+N_{t-1}),\log(1+c_t)
\right).
$$

均值和总体标准差沿位置维计算，统计使用 FP32，得到 $4d_M+3=2051$ 维输入。网络输出三个动作相对于保持容量的预测代价差。

$$
\mathcal A_t=\{g\in\{0,8,16\}:K_{t-1}+g\le K_{\max}\},
$$

$$
g_t=
\begin{cases}
K_{\mathrm{first}}, & K_{t-1}=0,\\[1mm]
\displaystyle\arg\min_{g\in\mathcal A_t}[V_\psi(s_t)]_g,
& K_{t-1}>0,
\end{cases}
\qquad
K_t=K_{t-1}+g_t.
$$

首次写入使用调用方给定的 $K_{\mathrm{first}}$，预训练的取值见第 6.2 节；统计和价值预测作用于已有记忆的状态。同分时选择较小的 $g$。训练容量网络前，外部增长安排在已有记忆状态上提供 $g_t$，其余写入计算保持一致。

### 5.3 零初值与联合更新

为新增位置补零：

$$
\overline M_t=
\begin{bmatrix}
M_{t-1}\\
\mathbf 0_{g_t\times d_M}
\end{bmatrix},
\qquad
Q_t^{(0)}=\overline M_t+P_{\mathrm{slot}}(0:K_t).
$$

零向量表示新增位置的内容起点，固定位置向量提供各输出位置的计算地址。首次写入时 $\overline M_t$ 全部由零向量构成。

更新器的信息来源为：

$$
Z_t=
\begin{bmatrix}
M_{t-1}+e_{\mathrm{old}}\\
H_t+e_{\mathrm{new}}
\end{bmatrix}.
$$

$e_{\mathrm{old}},e_{\mathrm{new}}$ 是可学习的来源类型向量，分别加到旧记忆和当前输入上。首次写入时，$Z_t$ 由当前输入部分构成。

对 $\ell=1,2,3$：

$$
\begin{aligned}
A_t^{(\ell)}
&=Q_t^{(\ell-1)}
+\operatorname{SelfAttn}_{\ell}
  \left(\operatorname{LN}_{s,\ell}(Q_t^{(\ell-1)})\right),\\
C_t^{(\ell)}
&=A_t^{(\ell)}
+\operatorname{CrossAttn}_{\ell}
  \left(\operatorname{LN}_{q,\ell}(A_t^{(\ell)}),
        \operatorname{LN}_{z,\ell}(Z_t)\right),\\
Q_t^{(\ell)}
&=C_t^{(\ell)}
+\operatorname{FFN}_{\ell}
  \left(\operatorname{LN}_{f,\ell}(C_t^{(\ell)})\right).
\end{aligned}
$$

Cross-attention 的 query 来自输出位置，key/value 来自 $Z_t$。每层使用 8 heads、FFN 宽度 2048、dropout 0。最后输出：

$$
M_t=\overline M_t+
W_{\mathrm{out}}\left(\operatorname{LN}_{o}(Q_t^{(3)})\right),
\qquad
N_t=N_{t-1}+c_t.
$$

$W_{\mathrm{out}}$ 为含 bias 的线性层，权重以标准差 $10^{-3}$ 的零均值正态分布初始化，bias 为零。该参数属于每次写入使用的更新器。输出状态转换为 BF16。

所有记忆位置均参与联合更新。保持容量的动作 $g=0$ 仍写入新信息；扩容时，新位置可以承接旧记忆中的内容。新增位置共享同一更新器，模型参数量与运行时位置数分别确定。

### 5.4 记忆读取

读取投影产生：

$$
\widetilde M_t=P_\rho(M_t)
\in\mathbb R^{K_t\times d_B}.
$$

令 $q$ 包含实际使用的任务提示或对话格式，$\operatorname{Emb}$ 为基座的 token embedding，输出概率为：

$$
p_\eta(y\mid M_t,q)
=\prod_{j=1}^{|y|}
p_{B+\mathrm{LoRA}_\rho}
\left(
y_j\mid
[\operatorname{Emb}(\mathrm{BOS}),
\widetilde M_t,
\operatorname{Emb}(q),
\operatorname{Emb}(y_{<j})]
\right).
$$

读取使用全部 $K_t$ 个位置，空记忆对应空的记忆前缀。训练采用目标上文的 teacher forcing；推理逐 token 生成，停止条件使用基座的标准 EOS 和任务输出预算。输入长度预算包含记忆、提示、目标及特殊 tokens。

## 6. 监督目标与容量标签

### 6.1 统一读取损失

对一个读取样本，定义目标序列含 EOS 的平均 token 负对数似然：

$$
\ell_\eta(M,q,y)
=-\frac{1}{|y|}\sum_{j=1}^{|y|}
\log p_\eta(y_j\mid M,q,y_{<j}).
$$

样本损失先按目标 token 平均，再按有效读取样本平均。评估 NLL/PPL 另按所有目标 tokens 加权统计。多参考任务在训练时按固定规则选择一个合法目标，同一次分支比较使用相同目标；评估沿用官方多参考规则。

### 6.2 多容量记忆读写预训练

自然边界确定的 FineWeb 原文片段为 $S$。AE 样本以 $X=S$ 写入并重建；LM 样本在 $S$ 内选择合法句界，将其分为连续的前缀 $X$ 与后缀 $Y$，分别用于记忆写入和目标监督。两类任务各自进行一次写入，$L_X,L_Y$ 表示相应 token 数。候选数量压缩率为 $\mathcal R=\{2,4,8\}$，各比例对应容量：

$$
K(L_X,r)=\min\left(
K_{\max},
\max\left(K_{\min},\left\lceil\frac{L_X}{r}\right\rceil\right)
\right),
\qquad r\in\mathcal R.
$$

$K_{\min}$ 为满足 $1\le K_{\min}\le K_{\max}$ 的容量下限，依据短句分布与 dev 曲线确定，并在运行配置中固定；$K_{\max}=512$。各候选还需满足完整写入及 AE、LM 读取的上下文预算。合法容量构成去重后的集合 $\mathcal K(L_X,L_Y)$，训练采样器按课程分布 $p(K\mid L_X,L_Y)$ 选择 $K_X$；采样权重由预算课程设置，序列长度决定容量的合法范围。

基座完整处理 $X$，更新器从空记忆一次写入，得到：

$$
M_X^{(K_X)}
=\operatorname{Write}_\theta(\varnothing,X;K_X).
$$

首次创建的 $K_X$ 个位置采用零内容初值和固定位置向量。所有容量共享 $W_{\mathrm{in}},U,P$ 和读取 LoRA。容量网络在后续容量决策阶段训练。

自重建（AE）使用完整片段的记忆 $M_S^{(K_S)}$；后续语言建模（LM）使用片段前缀的记忆 $M_X^{(K_X)}$：

$$
\mathcal L_{\mathrm{AE}}
=\ell_\eta(M_S^{(K_S)},q_{\mathrm{AE}},S),\qquad
\mathcal L_{\mathrm{LM}}
=\ell_\eta(M_X^{(K_X)},q_{\mathrm{continue}},Y),
$$

$$
\mathcal L_{\mathrm{pre}}
=\lambda_{\mathrm{AE}}\mathbb E_{S,K_S}[\mathcal L_{\mathrm{AE}}]
+\lambda_{\mathrm{LM}}\mathbb E_{(X,Y),K_X}[\mathcal L_{\mathrm{LM}}].
$$

两项权重为运行配置中的非负数，至少一项为正数。$X$ 构成写入输入，$Y$ 作为未来目标交给读取路径。目标直接取自原文，训练保留完整序列及 EOS。生成评价的输出预算按任务固定。

每次采样记录 $L_X,K_X$ 和实际比例 $r_{\mathrm{eff}}=L_X/K_X$，容量取整及上下限的影响由实际比例表示。在容量范围内，240-token 片段的三个预算分别为 120、60、30 个位置。$r_{\mathrm{eff}}$ 描述 token 数量与位置数的关系；实际存储另按 512 维记忆的 bytes 计量。

### 6.3 动态记忆目标

对话以轮次 $k$ 计数，$\mathcal M_{k-1}$ 是此前已完成对话的记忆，$q_k,r_k$ 是当前发言与人工回复：

$$
\mathcal L_{\mathrm{dialogue}}
=\frac1T\sum_{k=1}^{T}
\ell_\eta(\mathcal M_{k-1},q_k,r_k),
$$

$$
\mathcal M_k
=\operatorname{WriteSequence}_\theta
\left(\mathcal M_{k-1},
\operatorname{Serialize}(q_k,r_k)\right).
$$

$\operatorname{WriteSequence}$ 表示按第 3.2 节的自然写入单元调用更新器；合法长度的一轮交互执行一次写入。当前回复的损失在该轮写入之前计算。初始轮的记忆为空，后续轮的损失沿已有记忆的更新链传回写入参数。

文档 QA 使用已提交前缀的记忆、当前问题及合法参考答案计算同一 $\ell_\eta$。动态阶段以真实目标的读取损失训练整套读写模块。

### 6.4 同根状态的容量标签

容量训练固定读写参数 $\eta$，从动态运行轨迹中抽取已有记忆的根状态 $(M,H,N)$。根状态的后续区间至少包含一个有效读取目标。

每个合法动作 $g$ 产生一个候选记忆。各分支接收相同后续文本，在相同读取边界使用同一提示、参考答案及读取参数；后续增长动作统一为 $g=0$。首轮标签区间覆盖当前 episode 的剩余部分。

动作代价为：

$$
J(g;\eta)
=\overline\ell^{(g)}
+\lambda_s\overline C_{\mathrm{storage}}^{(g)}
+\lambda_w\overline C_{\mathrm{write}}^{(g)}
+\lambda_r\overline C_{\mathrm{read}}^{(g)},
$$

$$
\Delta J(g;\eta)=J(g;\eta)-J(0;\eta).
$$

$\overline\ell$ 为该分支有效读取目标的平均损失。资源代理使用共同参考容量 $K_{\mathrm{ref}}=64$：

| 代理 | 定义 |
|---|---|
| 存储 | $K_i/K_{\mathrm{ref}}$，按已处理源长度加权平均 |
| 写入 | $\left(K_i^2+K_i(K_{\mathrm{prev}}+c_i)\right)/(3K_{\mathrm{ref}}^2)$，按写入调用平均 |
| 读取 | $(K_i+q_i+a_i+2)^2/(K_{\mathrm{ref}}+q_i+a_i+2)^2$，按读取目标平均 |

$q_i$ 为提示 token 数，$a_i$ 为目标正文 token 数，额外 2 对应 BOS/EOS。代理用于动作训练，实际存储与运行时间单独测量。初始资源权重为 $\lambda_s=0.05,\lambda_w=0.01,\lambda_r=0.05$，依据 dev 的质量—成本曲线校准。

容量网络拟合：

$$
\mathcal L_V
=\mathbb E_{s,g\in\mathcal A}
\operatorname{SmoothL1}
\left([V_\psi(s)]_g,\Delta J(g;\eta)\right).
$$

标签描述当前 episode 剩余区间、后续保持容量条件下的动作价值。完整策略的收益通过连续运行评价。根状态覆盖率、剩余长度及监督数量一并记录；标签绑定完整的 $\eta$ checkpoint。

## 7. 三阶段训练与优化

### 7.1 阶段与参数范围

| 阶段 | 数据与操作 | 更新参数 | 进入下一阶段的依据 |
|---|---|---|---|
| 记忆读写预训练 `pretrain` | FineWeb 自然片段，变长单次写入、多压缩率采样，AE 与 LM 联合目标 | $W_{\mathrm{in}},U,P,\mathrm{LoRA}_\rho$ | 独立原文上的重建、续写表现及正确记忆相对空记忆、错误记忆的收益 |
| 动态记忆训练 `dynamic` | MSC 原始对话，连续读写与外部增长安排；QA 数据承担事实保持实验 | 继续训练相同的 $\eta$ | 跨轮保持、长轨迹稳定性及实际扩容分支的可重复收益 |
| 容量决策训练 `capacity` | 固定读写配对，生成动作代价标签并进行监督回归 | $V_\psi$ | 连续运行的质量—成本优于简单容量安排 |

语言模型基座在三个阶段均保持固定。基座的写入前向使用关闭读取 LoRA 的模式；读取前向保留从损失到记忆的计算图。基座权重冻结与输入梯度传播同时成立。

### 7.2 记忆读写预训练

每次依次采样长度区间、源文档、粒度与任务片段，最后按实际写入长度采样有效容量。每个样本从空记忆开始，以一个容量完成一次写入，计算其 AE 或 LM 损失；每次参数更新分别按各任务有效样本数归一化后加权求和。再次访问同一片段时可以重新采样容量，预训练产物为动态阶段直接继承的 $\eta$。

完整段落、主题相近的连续句子组、随机连续句子组、单句和相邻段落组合在训练中持续混合。各粒度权重与源文档总采样权重在配置中固定，控制同一原文的重复曝光。文本长度区间用于分层统计和采样平衡，具体片段由自然边界确定。

多压缩率课程使用 $\{2,4,8\}$ 作为首轮范围。初期提高较宽松预算的权重，重建与记忆读取稳定后增加较高压缩率样本，同时保留各有效预算的训练。课程分布与调整条件在运行配置中固定，实际调整时点写入训练进度。

独立文档 dev 对同一批 $(X,Y)$ 分别使用多个容量，绘制重建、续写与资源代价曲线。结果按粒度、输入长度、$K_X$、实际压缩率和切分方式分层。短文本与长文本、宽松预算与紧张预算分别检验记忆利用。

16 条真实 train 样本的过拟合承担链路调试，正式训练从统一初始化开始。训练预算分别记录独立原文覆盖量、实际处理的源 tokens 与目标 tokens；学习曲线用于确定训练规模和停止点。

### 7.3 动态训练与时间反传

动态训练从预训练 checkpoint 开始，沿原始对话顺序使用自身产生的记忆，以实际 turn 形成变长写入单元。外部增长安排覆盖整段保持容量和多种增长速度；增长时间表依据已到达长度、当前容量和随机种子生成。预训练中的 $K_X$ 是空记忆写入预算，动态阶段的 $K_t$ 则由累计历史与当前输入共同决定。

已有记忆状态上的随机动作以 $\{0:0.70,8:0.25,16:0.05\}$ 为 pilot 起点，按合法动作重新归一化。首次写入采用配置中的首批分配，pilot 为 16。各扩容位置使用统一的零内容初值。

训练先使用短轨迹，再逐步提高长轨迹比例并延伸至多 session 对话。短轨迹采用完整的沿时间反向传播（BPTT）；长轨迹采用截断式时间反传（TBPTT），初始候选窗口为 1024 个写入 tokens，依据历史依赖跨度与训练显存调整。读取损失在状态 detach 前反传，截断位置取完整写入与读取操作结束后的边界。窗口内的实际源长度和被截断的历史依赖范围纳入记录。

同一 episode 内 $\eta$ 保持固定，各 TBPTT 段累积梯度并传递 detached memory，episode 结束后执行 optimizer step。训练统计包括从相关信息写入到监督之间的 token 距离、更新次数和反传覆盖率。

### 7.4 容量训练与按需适配

容量阶段先固定 $\eta$ 生成标签，再训练 $\psi$。标签生成中的记忆持续更新，读写参数保持原值。价值网络训练使用 detached 状态特征和独立的回归目标。

策略访问新的状态分布时，在固定 $\eta$ 下补充对应根状态的标签并更新 $\psi$。策略轨迹上的读写退化触发短程动态适配；更新后的 $\eta$ 对应重新生成的容量标签。这些操作由 dev 结果触发。

### 7.5 优化、精度与预算

读写训练使用 AdamW，pilot 学习率 $10^{-4}$、weight decay 0.01、梯度范数裁剪 1.0。训练模块和 optimizer states 使用 FP32，记忆与主要 activations 使用 BF16，损失和统计使用 FP32。

基座的固定内部表示可以按实际写入单元缓存，缓存绑定基座、tokenizer 和完整序列化输入；各切分路径使用对应缓存。写入投影和更新器按当前参数重新构图。预训练、动态训练、容量标签生成和价值回归的计算成本分别记录。训练规模与停止点依据有效原文 tokens、会话数量、独立目标数和 dev 学习曲线确定。数据 seed 为 `20260907`，模型 pilot seed 为 `42`，正式对照至少覆盖 3 个模型 seeds。

## 8. 真实公开数据路线

### 8.1 数据范围与阶段职责

训练、验证、测试、smoke test 和语义诊断使用真实公开原文、人工对话及有来源依据的问答数据。改良版本保留原始文本、角色、时间和证据的来源关系，转换包括格式清理、自然边界识别、连续原文片段组织和监督边界对齐。

数据路线为：FineWeb 原文的 AE/LM 预训练，MSC 的动态记忆训练，同源轨迹上的容量学习，SQuAD/MRQA 等任务上的事实保持实验。单次运行通过配置确定语料来源、子集、数据 seed 和取样规模。

### 8.2 统一 episode 契约

目标持久化格式为 JSONL，每行一条 episode，包含：

| 字段 | 定义 |
|---|---|
| `episode_id` | 唯一标识 |
| `input_ids` | 按来源顺序序列化的原始 token 流；包含实际提供的角色、标题和时间信息 |
| `write_ends` | 实际写入单元的递增结束位置，相邻边界对应一个 token span；事件与单元的关系保留在来源记录中 |
| `sources` | `source_id, document_id, token_start, token_end, provenance` |
| `reads` | 在指定记忆前缀执行的读取目标 |

每个 read 记录 `read_id, task, prefix_end, prompt, references`。`task` 取 `ae, continuation, dialogue, qa`。每份 reference 记录 `text, evidence_spans`。AE、continuation 和 dialogue 使用原始序列目标，`evidence_spans` 为空列表；QA 的证据 spans 绑定对应参考答案，依赖全文判定且证据列表为空的目标安排在完整文档结束处。所有 token spans 为左闭右开，`prefix_end` 对应已经提交的写入边界或初始位置 0。

对话 read 的 `prefix_end` 指向当前轮之前的历史，`prompt` 为当前发言，`text` 为人工回复。包含当前发言和回复的写入事件在该 read 完成后提交。AE 的目标为已写入原文；continuation 的目标为该前缀之后的原文；QA 参考答案由已到达证据支持。

每个 FineWeb 预训练 episode 对应一次写入和一个任务：`input_ids` 为 $X$，`write_ends=[L_X]`，唯一 read 的 `prefix_end=L_X`。AE 的目标与输入相同；continuation 的目标为片段内后缀 $Y$。来源记录保存父片段、输入与目标的原文字符位置，以及 `boundary_variant`、文档簇和模型质量判定。两套数据通过完成记录中的 `source_pool_id` 关联共同的来源划分。$K_X$ 在训练访问时采样并写入运行日志。

训练器依照写入单元和读取边界执行调度。写入接口仅取得当前单元的 tokens，读取接口取得对应 prompt，目标序列用于损失计算。切分变化保持来源位置和读取边界一致。

### 8.3 划分、监督与计分

先按源文档或完整会话链确定 split，再构造窗口。保留公开划分，内部 dev 从 train 按来源划出，最终评估集承担独立计分。共享历史前缀、同文档窗口和改良版本归属同一 split。

MSC 每个目标轮使用一条人工回复监督。QA 在共同提交前缀选取至多 3 个不同且合法的问题，按实际有效数量归一化；写入调度持续覆盖完整选定文本。重复读取旧问题用于保持实验，单个问题在 episode 内的累计训练权重受限。

QA 充分证据到达点由一份参考答案对应证据集合的最晚结束位置确定，再对齐到共同提交边界。QASPER 的不同 annotator 标注分别绑定答案和证据；依赖全文判定的答案在完整文档结束处评分。数据覆盖率、有效目标数及证据距离分布随结果记录。

### 8.4 数据集与实验角色

| 数据来源 | 本框架中的角色 | 使用方式 |
|---|---|---|
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 记忆读写预训练语料 | 原始段落、连续句子组及单句的多粒度重建与续写，按文档保留来源与划分 |
| [MSC](https://parl.ai/projects/msc/) | 动态记忆主训练与容量标签来源 | 人工众包角色对话，按原始顺序保持跨 session 历史，以人工下一回复监督 |
| [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/) | 可明确计分的事实保持实验 | 同篇文章已发布段落按原序组成多问流，问题在窗口固定后对齐 |
| [MRQA](https://github.com/mrqa/MRQA-Shared-Task-2019) | 多来源 QA 适配与泛化实验 | 训练来源覆盖 SQuAD、NewsQA、TriviaQA、SearchQA、HotpotQA、Natural Questions，按官方域内/域外协议使用 |
| [HotpotQA](https://hotpotqa.github.io/) | 跨段组合实验 | 保留完整候选上下文和原始 supporting facts，按证据到达计分 |
| [QASPER](https://huggingface.co/datasets/allenai/qasper) | 困难文档验证 | 论文原文、多参考答案和各自证据；全文与派生窗口分表报告 |

MSC 的人物角色和跨 session 时间间隔由众包任务设定。其三 session 与四 session 训练轨迹分别有 4,000 和 1,001 条，四 session 轨迹平均约 53.3 条 utterances；官方逐样本任务视图提供约 237k 训练 examples。[MSC 原论文](https://arxiv.org/pdf/2107.07567)

SQuAD/MRQA 实验分别报告动态训练模型直接读取和 QA 适配后的结果。QA 适配使用训练 split，属于同一动态记忆训练阶段的任务扩展。QASPER 首轮承担独立文档验证。

[MuSiQue](https://github.com/StonyBrookNLP/musique)承担困难组合推理；使用 SQuAD/NQ 训练来源时，按官方 dev/test 单跳种子清单隔离共享样本。[StreamingQA](https://proceedings.mlr.press/v162/liska22a.html)的人写问题及带日期新闻承担自然时间到达实验。[DynaQuest](https://github.com/nusnlp/DynaQuest)的修订实验绑定实际 Wikipedia 新旧版本、页面修订时间和事实有效范围。[LongBench](https://github.com/THUDM/LongBench)按组成数据的来源关系归并统计。

### 8.5 FineWeb 获取与多粒度样本构造

预训练来源为 `HuggingFaceFW/fineweb`，本地目录使用 `data/raw/HuggingFaceFW-fineweb/sample-10BT/`。程序按配置的子集、数据 seed 和文档预算直接分批读取本地 Parquet。公开的 `sample-10BT` 等子集可以提供固定抽样池，实际训练量由运行预算确定。[FineWeb 数据说明](https://huggingface.co/datasets/HuggingFaceFW/fineweb#smaller-sample-versions)

数据处理依次完成以下操作：

1. 读取固定候选池，保留原文、文档 ID、URL、抓取来源及日期。基础检查校验字段及最短文本长度，随后按 ID、规范化 URL、全文和近重复关系去重。
2. 为每个来源簇确定唯一 train/dev/test 归属，建立两套数据共用的来源登记。每簇保留一个通过基础检查的代表，其全部派生片段继承相同 split。
3. 识别原始段落和句子边界，构造多粒度 semantic 候选，每个粒度覆盖不同自然长度。合法自然片段 $S$ 生成 AE 样本；在内部合法句界随机切分，生成以 $X$ 写入、以剩余后缀 $Y$ 监督的独立 LM 样本。$Y$ 的全部 tokens 与 EOS 参与损失。单句片段用于 AE。
4. Qwen3.8-27B 判定最终 AE 的 $X$ 或 LM 的 $X/Y$，评价内容质量及句界完整性。模型保留且通过样本去重的候选计入配额，各 split 的 AE 与 LM 分别达到相同目标数量。构造过程持续访问候选并补足筛选造成的缺额。
5. 统计 semantic 入选样本在各 split、各任务中的实际输入长度区间分布。random 使用独立来源顺序和原文 token 起点，在所需区间构造 AE 或内部续写 LM 候选。模型沿用共同内容标准，并接受随机边界造成的句子或词片段。保留结果按任务和区间补足配额。
6. 两套数据分别保存到 `semantic/` 与 `random/`，来源位置保持可追溯。检查原文连续性、共享来源划分、容量、任务配额和区间计数后写入完成记录；比较文件记录输入分布、LM 目标长度和句界截断统计。正式训练验证使用 semantic。

`samples_per_task` 按 train/dev/test 指定每个版本、每项任务的最终数量，示例默认值为 100000／2000／2000。实际输入长度区间为 1–64、65–128、129–256、257–512、513–1024 tokens。random 按 semantic 入选后的区间计数构造，区间内的具体长度和来源独立选择；LM 目标长度与监督 token 总量单独统计。样本任务去重和跨 split 相同长输入检查在模型入选过程中执行。

模型评分返回 `keep / reject / uncertain` 与理由，`keep` 进入最终配额。提示词覆盖技术文字、叙事、对话、自然指代及话题变化，通过语义判定识别损坏文本与网页噪声。评分缓存绑定模型、提示协议和实际 X/Y。数据准备代码统一位于 `src/latent_working_memory/data_preparation/`，入口和契约见 [独立 AE/LM 数据构造实现](v1/20260909_independent_ae_lm_data_preparation.md)。

质量抽查使用独立后置入口，对完整数据抽取随机与分层面板，再由模型或人工复核。抽查结果用于定位筛选遗漏和调整下一次构造配方。强模型选型与资源预算见 [评分与后置抽查方案](v1/20260909_quality_scoring_and_inspection.md)。

扩大数据 pilot 的历史评分、阈值与助手抽查排除理由，以及两组学习率实验，见 [扩大数据预训练记录](v1/20260908_fineweb_generalization_pretraining_record.md)。

| 粒度 | 构造规则 | 训练作用 |
|---|---|---|
| 完整段落 | 保留原始段落 | 完整叙述和多句关系 |
| 主题相近的连续句子组 | 在原文连续句子间选择主题边界 | 语义较集中的信息写入 |
| 随机连续句子组 | 随机选择句间边界，保留原文顺序 | 混合主题与不规则到达边界 |
| 单句 | 保留原始完整句子 | 短输入与细粒度写入 |
| 相邻段落组合 | 合并同一文档中相邻段落 | 较长输入与跨段关系 |

原始段落、可靠句界识别和随机合并、拆分相邻句子组构成基础数据路线。主题边界样本由离线大模型标注提供：输入带编号的原文句子，输出连续索引区间，正文由程序按索引提取。标注结果保存复用，并检查索引顺序、范围和切分覆盖。其训练占比通过匹配长度分布与训练预算的 dev 对照确定。相邻句子表示的相似度可作为另一种边界标注对照。

多粒度视图持续混合，单句的局部表达与段落的上下文关系共同进入训练。长度、主题混合程度和容量预算分别记录。MSC train 的实际 turn 长度分布用于检查预训练覆盖；动态阶段直接使用人工对话及实际轮次边界。

预训练边界对照使用按任务对齐输入长度区间的两套独立数据，统一训练预算、容量策略和曝光统计，并在固定的 semantic 与 random 评估面板上比较。各条件记录实际来源覆盖、目标长度及监督 token 量，用于分析边界与数据组成的共同影响。

## 9. 评估与对照

### 9.1 系统对照

| 条件 | 定义 | 诊断目标 |
|---|---|---|
| `no_memory` | 使用本方法的读取接口，记忆前缀为空 | 基座与读取 LoRA 的先验表现 |
| `wrong_memory` | 使用来自其他来源的记忆，匹配容量 | 当前输出对正确历史内容的依赖 |
| `recent_text` | 读取近期原文 | 近期信息能够解释的任务表现 |
| `full_text` | 固定基座读取完整已到达历史 | 原文条件下的能力参考 |
| `fixed_joint` | 首次分配固定容量，持续联合更新 | 固定资源下的长期保持 |
| `scheduled_joint` | 按源位置的固定时间表增加容量并联合更新 | 预设增长计划的质量—成本 |
| `append_only` | 已有记忆保持原值，新信息写入追加位置 | 追加式存储的质量—成本 |
| `learned_joint` | 联合更新器与学得的容量策略 | 完整系统收益 |
| `global_at_K` | 独立训练的记忆读写配对，每次按合法自然单元重算完整前缀表示，再一次聚合为 $K$ 个位置 | 同容量完整历史重读的质量与成本 |

固定容量、计划增长、追加式、学习增长和完整前缀重读系统匹配基座、tokenizer、数据及训练预算，各结构训练配套的记忆模块和读取 LoRA。输入诊断复用被诊断模型的读取参数，完整原文参考使用冻结基座。增长对照记录首次分配边界、后续增长时间、容量轨迹和达到上限的行为。

受控切分比较共享原始文本、首次写入边界与增长事件位置，各路径在共同增长位置提交状态，并重新计算各单元的基座表示。主要结论依据完整系统的质量—成本比较；联合重组的归因依据匹配容量轨迹的对照。完整前缀重读结果用于估计在线记忆与同容量重读系统之间的实际差距。

### 9.2 指标

- 预训练：AE 重建 NLL、token 准确率、自由生成序列匹配与归一化 token 编辑距离，LM 续写 NLL/PPL，以及正确、空、错误记忆之间的差距；按粒度、输入长度、$K_X$、实际压缩率和切分方式分层。固定评估面板按粒度与长度选取独立文档，完整序列匹配包含 EOS，编辑距离在正文 tokens 上计算。
- 对话：目标 token 加权 NLL/PPL、BLEU、ROUGE，按 session、历史长度和轮次位置分层。人工历史条件与生成回复写回条件分别报告。
- QA：沿用官方 answer EM/F1、多参考规则；QASPER 使用 Answer F1。原始证据用于监督边界和保持距离分层。
- 动态保持：按信息写入后的 tokens、更新次数及 session 间隔统计表现，结合近期文本与错误记忆对照。
- 容量：$\Delta\ell$、$\Delta J$、价值预测误差、动作 regret、扩容比例、触顶率，以及最终容量和完整轨迹。
- 存储：`M.numel() * M.element_size()` 加实际持久元数据；源长度积分为 $\sum_i B(M_i)(p_{i+1}-p_i)$。
- 计算：基座文本前向、记忆更新、记忆读取的实测时间，TTFT、峰值显存和累计写入时间。GPU 计时包含同步和预热。
- 训练成本：数据预处理与边界标注、预训练、动态训练、分支标签生成和容量回归分别计量。

完整原文参考的输入预算覆盖历史、请求、答案与特殊 tokens。超过其合法窗口的样本使用真实目标评价，并记录参考的可用范围。长程验证从 4096 个写入 tokens 起，按来源与任务分别报告。

### 9.3 判断依据

预训练阶段以独立文本上的实际记忆利用作为推进依据；动态阶段以长轨迹保持和同根扩容收益作为容量学习依据；容量阶段以连续运行相对简单计划的收益作为部署策略依据。

评估联合重组时匹配容量轨迹和读取适配条件；评估读取可学习性时固定记忆写入参数，从相同读取初始化比较学习曲线。主要统计按源文档或会话链聚类进行成对 bootstrap，并结合多个模型 seeds 判断稳定性。

## 10. 结构与训练消融

| 实验变量 | 默认设置 | 对照设置 |
|---|---|---|
| 新位置初值 | 零向量加固定位置向量 | 共享可学习向量加同一固定位置向量 |
| 预训练目标 | AE 与 LM 任务分别构造，联合优化 | 分别使用 AE 或 LM |
| 文本边界 | 完整句界版本 | 各任务输入长度区间对齐的随机截断版本，以及两者混合 |
| 预训练容量 | 多压缩率采样 | 固定压缩率、固定位置数 |
| 预训练粒度 | 段落、连续句子组、单句及相邻段落组合混合 | 单一粒度 |
| 主题边界样本 | 原始结构与随机句界样本为基础 | 加入离线大模型或句子相似度标注的边界 |
| 读取适配 | $P$ 与读取 LoRA 联合训练 | 固定读取 LoRA、仅训练 $P$ 与记忆写入模块 |
| 更新切分 | 实际 turn 或自然片段为一个写入单元 | 同一原文的句界拆分、相邻单元合并及混合路径 |
| 时间反传 | 短轨迹完整反传，长轨迹按配置截断 | 匹配轨迹的更长反传窗 |
| 容量标签范围 | 当前 episode 剩余部分 | 相同根状态上的固定源长度窗口 |

可学习初值对照统一作用于首次创建和后续新增位置。两组系统从空历史开始，保持相同容量轨迹、数据、训练预算和读取结构，比较学习速度、任务表现及长期保持。

输出蒸馏与切分一致性属于由 dev 结果触发的辅助目标实验。输出蒸馏使用固定完整上下文基座的同词表分布；切分实验从同一状态、相同容量安排和相同读取目标出发，各路径按自身边界计算基座表示，在当前结束点及共同后续输入之后比较输出分布。每项辅助目标单独记录增益与计算成本。

## 11. 实现契约与工程落点

### 11.1 实现状态与接入范围

预训练链路已实现空状态、统一零初值、外部首批容量、自然单元完整前向、变长 batch、FineWeb 多粒度 adapter、AE/LM 联合目标、多压缩率采样、训练恢复和独立文档多容量评估。本地 65 项 v1 测试通过。当前数据流程已实现共享来源池、semantic/random 独立构造、最终样本模型判定、等量任务配额、入选长度区间对齐及独立抽查，并通过本地 HTTP 测试服务验证完整命令链。Qwen3.8-27B 权重已下载到服务器，真实评分服务与正式数据生成是下一执行步骤。实现见 [独立 AE/LM 数据构造](v1/20260909_independent_ae_lm_data_preparation.md)。

FineWeb `sample-10BT` 已下载到服务器，Llama-2-7B-Chat 单卡训练、保存恢复与多容量评估已跑通。历史扩大数据实验准备了 10,000 篇训练文档及各 512 篇 dev/test 文档，训练集包含 126,278 个样本对；两组学习率预训练各完成 2000 步；dev 选择 3e-5 第 2000 步 checkpoint。512 文档独立 test 的 AE/LM 正确记忆 NLL 为 2.2532/2.3914，相对空记忆收益为 0.1409/0.1242；64 文档、三个容量的自由重建完整匹配为 0/192，平均归一化 token 编辑距离为 0.9391。结果详见 [扩大数据预训练记录](v1/20260908_fineweb_generalization_pretraining_record.md)。此前逐样本配对数据每套包含 226,022／11,610／11,685 个 train/dev/test 样本，作为历史记录保留，见 [配对数据构造](v1/20260909_paired_boundary_pretraining_data.md)。后续框架任务为 MSC 对话调度、动态训练、容量学习及完整系统对照。

| 文件 | 本版目标职责 |
|---|---|
| `state.py` | 空记忆、容量、源位置及状态持久化契约 |
| `backbone.py` | 自然单元完整前向、变长 batch 与 mask、写入/读取投影和读取 LoRA |
| `model.py` | 零填充、联合更新器、来源类型向量和容量网络 |
| `data.py / fineweb.py / prepare_data.py` | FineWeb 来源与自然边界、多粒度片段、读取目标与统一 episode；后续接入 MSC turn |
| `objectives.py` | AE、LM、回复/QA 损失及辅助目标 |
| `sampling.py / rollout.py / training.py` | 粒度与容量采样、AE/LM 联合目标与恢复；后续接入变长对话调度、时间反传和容量训练 |
| `capacity.py` | 同根动作分支、代价标签与价值回归 |
| `baselines.py / evaluation.py` | 配套系统对照、任务与资源指标 |
| `config.py / checkpoint.py` | 单次实验配置、训练恢复与运行时状态 |
| `train.py / evaluate.py` | 训练和评估入口 |

详细工程任务见 [v1 实施总计划](v1/20260907_growing_latent_working_memory_implementation_plan.md)；结构、数据与训练协议以本文为准。

### 11.2 输入、状态与恢复

写入事件携带非空原始 tokens，当前表示的源起点等于 `seen_tokens`。状态允许 $K=0$ 的初始情形，首次分配满足 $1\le K_{\mathrm{first}}\le K_{\max}$；预训练由样本容量 $K_X$ 提供该参数，动态训练与推理由配置提供。已有记忆按合法增长动作维护容量。接口校验维度、dtype/device、有限数值、源位置和基座上下文预算。

模型 checkpoint 保存 `phase, config, model_state, optimizer_state, progress, rng_state`，其中 `model_state` 包含五项可训练模块。预训练保存文档与样本采样进度、容量课程进度和随机状态，恢复时复现下一样本及其 $K_X$；动态训练在 episode 边界保存优化进度与随机状态。运行时记忆保存 `model_checkpoint, values, seen_tokens`，恢复时匹配完整读写与容量参数。

容量标签保存 `root_id, episode_id, prefix_start, student_checkpoint, rollout_trace, policy_features, legal_actions, costs, target_deltas, continuation_end, read_ids`。`student_checkpoint` 绑定完整 $\eta$，`policy_features` 为 2051 维 FP32 值。动作按 `[0,8,16]` 排列，有效目标由合法动作集合确定。空状态通过首次分配规则处理。

配置采用 JSON，episode、容量标签和指标采用 JSONL，模型及运行时状态采用单一 `.pt` 字典格式。实验 provenance 记录基座、tokenizer、数据及软件的实际版本与来源。

### 11.3 配置与资源

结构和优化的 pilot 起点为：

```json
{
  "model_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
  "d_mem": 512,
  "num_layers": 3,
  "num_heads": 8,
  "ffn_dim": 2048,
  "reader_lora_rank": 16,
  "reader_lora_alpha": 32,
  "reader_lora_target_modules": ["q_proj", "v_proj"],
  "reader_lora_dropout": 0.0,
  "pretrain_dataset": "HuggingFaceFW/fineweb",
  "pretrain_compression_ratios": [2, 4, 8],
  "dynamic_k_first": 16,
  "k_limit": 512,
  "growth_actions": [0, 8, 16],
  "exploration_probs": [0.70, 0.25, 0.05],
  "learning_rate": 0.0001,
  "weight_decay": 0.01,
  "gradient_clip": 1.0,
  "bptt_tokens": 1024,
  "data_seed": 20260907,
  "model_seed": 42
}
```

运行配置另外固定 FineWeb 子集与来源划分、粒度采样权重、边界标注来源与占比、预训练 $K_{\min}$、容量课程、AE/LM 权重、合法上下文与输出预算、训练规模、资源代价权重及评估设置。上述预训练参数在启动前完整解析。记录实际文本长度、$K_X$、$r_{\mathrm{eff}}$、目标长度与有效监督量。

代码位于 `src/latent_working_memory/v1/`，产物位于 `data/v1/`、`checkpoints/v1/`、`artifacts/v1/experiments/<run_id>/`。独立评估保存到 `artifacts/v1/evaluations/<run_id>/`，本地历史产物的分类见 [产物整理记录](v1/20260909_artifact_organization.md)。实现采用 PyTorch、Transformers、PEFT 和 `uv` 管理的环境。SwanLab 记录训练与独立评估指标、分层曲线和重建样例，实验 ID 与可视化选项独立于模型配置保存。GPU 实验限定物理 GPU 0、1；启动命令显式设置 `CUDA_VISIBLE_DEVICES=0`、`1` 或 `0,1`。

## 12. 实施与验收顺序

1. 空状态、外部首批容量、零初值增长和位置向量形成可执行的记忆转移；自然单元完整前向及变长 batch 的有效位置对齐。
2. FineWeb adapter 完成来源隔离、自然边界、多粒度 $(X,Y)$ 与统一 episode；MSC adapter 对齐原始 turn 的读取、写回与来源位置。
3. 多压缩率预训练的 AE/LM 损失到达 $W_{\mathrm{in}},U,P,\mathrm{LoRA}_\rho$；完成真实样本调试、采样与容量恢复、独立文档多容量曲线验证。
4. 动态训练沿自身记忆连续运行，完成变长 turn、长轨迹保持、自然切分和实际扩容收益验证。
5. 固定读写配对生成分支标签，训练容量网络，并完成固定容量、计划增长、追加式与学习增长的质量—成本对照。
6. 在相同实验预算下完成初始化消融、长程验证及多 seed 统计，记录能够由实验支持的系统与机制结论。
