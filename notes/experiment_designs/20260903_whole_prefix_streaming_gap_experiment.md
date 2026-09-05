# Whole-prefix 与 Streaming Latent Compression Gap 实验方案（2026-09-03 19:43 CST）

创建时间：2026-09-03 19:43:59 CST（UTC+08:00）
状态：初始方案，待完成小规模 pilot 后预注册主实验阈值

## 术语与符号

- **Causal prefix（因果前缀）**：时刻 $t$ 已经到达的完整输入 $X_{\le t}$，不包括未来文本或未来 query。
- **Whole-prefix compressor** $G$：每个评估时刻重新读取 $X_{\le t}$，将其压缩为固定数量 $K$ 的 latent slots。本文将它作为高写入成本的同容量参考，不视为可部署的在线方法。
- **Streaming writer** $S$：每次只读取旧 memory $M_{t-1}$ 和当前 chunk $x_t$，递归产生 $M_t$，不重读完整前缀。
- **Latent slot**：一个可被固定 reader 消费的连续向量或连续状态。若方法使用多组 thread states，实验按总 latent 数和实际持久字节统一折算为 $K$。
- **Information atom（信息原子）** $u_i$：具有明确真值、可单独查询和计分的最小事实、事件、关系或状态更新。
- **Functional gap（功能差距）**：同一 reader 在 $G(X_{\le t})$ 与 $S(X_{\le t})$ 上的行为差异，记为 $\Delta_t$。
- **Coverage（覆盖率）**：能够从至少一个 slot 或一组 slots 中恢复的信息原子比例。
- **Functional redundancy（功能冗余）**：不同 slots 对相同信息或相同 query 集提供近似可替代的贡献。
- **Path dependence（路径依赖）**：最终 token 序列相同，但分块边界或递归合并顺序不同，导致 memory 功能显著不同。
- **Order sensitivity（顺序敏感性）**：信息集合相同，但相互独立原子的到达顺序不同，导致 memory 功能显著不同。它与严格的同序列路径依赖分开报告。
- **Functional distance** $D_{\mathrm{func}}$：在固定 probe/query 集上比较两个 memories 的输出分布或任务行为，而非逐 slot 均方误差。

## 背景与目标

在 latent 数量、持久状态字节和 reader 相同的条件下，whole-prefix compressor 可以对完整 causal prefix 进行联合分配；streaming writer 只能根据既有压缩状态和当前 chunk 做局部更新。前者可能更容易减少 slots 重复、覆盖更多信息，并降低分块与更新顺序造成的表示偏差。

该判断目前只是待验证假设。实验首先确认 whole-prefix 与 streaming 之间是否存在稳定、具有实际意义的功能差距；若存在，再判断差距主要来自以下哪种机制：

1. 多个 slots 保存了可替代的信息，造成有效容量下降；
2. 部分信息没有写入任何 slot，形成 coverage 缺口；
3. 相同内容经过不同 streaming 路径后产生不同 memory，表现出路径依赖；
4. 信息已经写入，但 reader 无法正确寻址或解码。

## 研究问题与假设

### Q1：同容量 whole-prefix 优势是否存在

在相同 causal prefix、总 latent 数、持久字节、reader、训练数据和基础模型下，比较：

$$
M_t^G=G(X_{\le t};K),
\qquad
M_t^S=S(M_{t-1}^S,x_t;K).
$$

主要假设 $H_1$：$M_t^G$ 在未来 query、精确信息恢复和分块稳定性上优于 $M_t^S$。
零假设 $H_0$：匹配容量后不存在稳定差距，或简单的分块随机化已经消除差距。

### Q2：差距由什么机制产生

- $H_{\mathrm{red}}$：streaming memory 中存在更多功能重复 slots，且重复与 uncovered atoms 同时出现。
- $H_{\mathrm{cov}}$：streaming memory 中无法从任何 slot 或合理 slot 组合恢复的信息原子更多。
- $H_{\mathrm{path}}$：不同合法 streaming 路径造成的功能方差显著高于随机种子方差。
- $H_{\mathrm{read}}$：若 oracle routing 或 forced read 能恢复信息，则差距至少部分来自 reader，而非 writer storage。

## 实验对象与对照

### 方法

| 标记 | 方法 | 用途 |
|---|---|---|
| $F$ | 完整原始 causal prefix | reader 可达到的任务参考，不参与同容量结论 |
| $M_0$ | 无 memory | 估计 decoder 先验和数据可猜性 |
| $G$ | whole-prefix compressor，输出固定 $K$ slots | 同容量高写入成本参考 |
| $S_{\mathrm{base}}$ | 简单 recurrent 或局部 merge writer | streaming 基线 |
| $S_{\mathrm{C\text{-}DIC}}$ | C-DIC 式检索、重编码与局部替换 | 直接相关基线 |
| $S_{\mathrm{new}}$ | 后续提出的联合重分配 writer | 仅在确认 gap 后加入 |

### 必须固定的条件

- 每个评估 checkpoint 对应完全相同的 causal prefix。
- $G$ 与所有 streaming 方法输出相同总数 $K$ 的 latent slots。
- 统计全部持久状态：latent values、逐层 KV、address、质量权重、索引和其他 metadata。
- reader、query、生成参数、基础模型和 tokenizer 保持相同。
- whole-prefix 方法可以使用更多写入计算，但必须单独报告 write FLOPs、峰值显存和延迟。
- writer 在推理时不得访问未来文本、未来 query 或答案。
- 训练预算尽量匹配；无法匹配时，将差异作为限制单独报告。

## 数据设计

### 阶段 A：受控合成流

先使用具有完整原子标注和 query 覆盖的数据，避免自然文本的答案歧义掩盖机制。

基础设置：

- 每条流包含约 64 个信息原子和 8 个主题；
- 使用 $K\in\{8,16,32\}$ 检查容量区间；
- 每个 checkpoint 查询全部仍然有效的历史原子；
- 原子类型至少包括实体属性、数值、二元关系、否定、时间事件和跨原子组合关系。

受控因素：

- **Topic layout**：同主题连续出现或多个主题交错出现；
- **Boundary**：改变 chunk size 和 offset，并有意将实体、关系、值拆到不同 chunks；
- **Near-duplicate**：加入表面相似但值、主体或极性不同的事实；
- **Update**：加入事实修正、失效和恢复，明确区分新增信息与覆盖旧状态；
- **Distractor**：控制无关信息比例和信息密度；
- **Order**：只对相互独立的原子改变到达顺序；时间更新任务保持事件顺序不变。

### 阶段 B：自然文本

只有阶段 A 确认 gap 和机制信号后，才进入自然对话或长文本。自然数据需要保存 atom、source span、时间、有效性和 query，且不得仅依赖自动生成答案作为唯一真值。

## 核心测量

### 1. 总体功能差距

令 $A(M_t)$ 为固定 query 集上的加权准确率或标准化负对数似然：

$$
\Delta_t=A(M_t^G)-A(M_t^S).
$$

同时报告原子类型、出现时间、经历更新次数和 chunk 边界位置上的分层结果。主要统计使用配对 bootstrap 置信区间；实际意义阈值在 pilot 后、主实验前固定。

### 2. Information-atom × slot 功能矩阵

对每个信息原子 $u_i$ 构造 query $q_i$。限制 reader 每次只读取 slot $m_k$，定义：

$$
R_{ik}=\operatorname{Recover}(u_i\mid q_i,m_k).
$$

$R\in\mathbb R^{N\times K}$ 是主要诊断对象：

- 相似列表示候选功能冗余；
- 无 slot 能恢复的行表示候选 coverage 缺口；
- 不同 partition 下列结构或空白行发生变化，表示候选路径依赖。

单个事实可能分布式编码在多个 slots 中，因此单-slot probe 不能单独作为最终结论。还需加入 top-$r$ slot probe、slot-pair probe 和 leave-one-slot-out 测试。

### 3. Coverage

单-slot coverage 定义为：

$$
C_1(M)=
\frac{\sum_i w_i\mathbf 1[\max_k R_{ik}>\tau]}{\sum_i w_i}.
$$

另外报告 $C_r(M)$：允许 oracle 选择不超过 $r$ 个 slots 时的覆盖率。若正常读取失败但 $C_r$ 较高，问题更可能来自寻址或 reader utilization；若 $C_r$ 仍低，则更可能是 writer 没有保存信息。

### 4. Functional redundancy

对每个 slot 计算删除后的任务损失变化：

$$
\delta_k=L(M\setminus\{m_k\})-L(M).
$$

若多个 $\delta_k$ 接近零，同时仍有 uncovered atoms，则存在容量浪费。报告：

- leave-one-slot-out utility $\delta_k$ 的分布；
- $R$ 各列在信息原子集合上的 weighted overlap；
- 有效 slot 数 $K_{\mathrm{eff}}=|\{k:\delta_k>\epsilon\}|$；
- 删除一个 slot 后，其他 slots 能否替代其行为。

$\epsilon$ 和恢复阈值 $\tau$ 必须由独立 calibration split 或 pilot 决定。

### 5. Path dependence

对同一 token 序列生成一组合法路径 $\Pi(X)$，包括不同 chunk size、offset 和递归合并树。定义：

$$
\operatorname{PDI}(X)=
\mathbb E_{\pi,\pi'\sim\Pi(X)}
D_{\mathrm{func}}(M^{(\pi)},M^{(\pi')}).
$$

$D_{\mathrm{func}}$ 优先使用固定 probe 集上的输出 KL、答案不一致率或 reader readout 距离。另测：

$$
D_{\mathrm{part}}
=D_{\mathrm{func}}
\left(U(M,[A;B]),U(U(M,A),B)\right),
$$

用于衡量一次处理与分块递归处理的差异。对于语义独立的 $A,B$，另外测量顺序敏感性：

$$
D_{\mathrm{comm}}
=D_{\mathrm{func}}
\left(U(U(M,A),B),U(U(M,B),A)\right).
$$

需要将路径方差与同路径不同随机种子的方差分开报告。只有路径效应稳定高于 seed noise，才支持路径依赖解释。

## 机制归因实验

观察到指标相关性后，必须进行固定容量的定向修复，才能支持因果归因。

### Reader repair

- 使用 oracle slot routing；
- 强制读取包含目标信息的 slot 或 slot 组合；
- 保持 memory 内容不变。

若该修复显著缩小 gap，则差距至少部分来自 addressing 或 decoding，而不是 storage coverage。

### Redundancy–coverage repair

1. 找出 leave-one-slot-out utility 最低的冗余 slot；
2. 删除该 slot；
3. 用只编码一个 uncovered atom 或 uncovered atom group 的 oracle slot 替换；
4. 保持总数 $K$、reader 和其余 memory 不变。

定义 gap closure：

$$
\rho_{\mathrm{repair}}
=
\frac{A(M_{\mathrm{repair}})-A(M^S)}
{A(M^G)-A(M^S)}.
$$

该 oracle repair 只用于机制诊断，不计入可部署方法结果。若替换少量冗余 slots 即稳定恢复较大比例的 gap，才支持“重复导致 coverage 浪费”的解释。

### Path repair

在输入、总容量和 reader 不变时，将 left-fold 更新改为 balanced-tree merge、周期性联合重分配或其他降低更新深度的方案。比较修复前后的 PDI、coverage 和最终行为。任何额外缓存和写入计算都必须计入状态与成本。

## 判定逻辑

| 观测 | 更可能的解释 |
|---|---|
| 正常读取失败，forced read 成功 | addressing 或 decoder utilization 问题 |
| forced read 和 slot 组合仍失败 | writer storage/coverage 问题 |
| $K_{\mathrm{eff}}<K$ 且存在 uncovered atoms | 功能冗余造成容量浪费 |
| 改变 partition 后缺失原子与 slot 分工系统变化 | 路径依赖 |
| 路径方差不高于 seed 方差 | 暂不支持路径依赖 |
| oracle redundancy–coverage repair 大幅闭合 gap | 支持重复和 coverage 的因果解释 |
| 几何指标改善但行为不改善 | 所选表示空间或指标缺乏任务相关性 |

## Go/No-go 标准

继续方法研究至少需要满足：

1. $G$ 相对 streaming 基线存在跨 seed、跨 partition 的稳定功能优势；
2. 优势在匹配总 latent 和持久字节后仍存在；
3. 至少一种机制指标能够预测逐样本或逐原子的 gap；
4. 对应的定向 repair 能稳定闭合部分 gap。

以下情况应停止或调整主要动机：

- matched-capacity $G$ 没有稳定优势；
- 简单随机分块训练已经消除差距；
- gap 主要来自 reader，而非 writer 更新；
- redundancy、coverage 和路径指标与行为 gap 无稳定关系；
- 完整状态和成本核算后，方法不再具有容量或效率优势。

## 实验产物与记录要求

每次运行至少保存：

- 配置、代码版本、随机种子和精确运行时间；
- 原始 causal prefix、chunk boundaries 和 path identifier；
- 信息原子、source span、有效时间和全部 queries；
- 总 latent 数、dtype、逐层 KV、metadata 和持久字节；
- 每个 checkpoint 的输出、$R$ 矩阵、coverage、redundancy 和 PDI；
- oracle repair 的目标 slot、被替换内容和 gap closure；
- 写入/读取 FLOPs、延迟、峰值显存和失败日志。

首轮实现顺序：受控数据与 probe → $G$/$S$ 同容量比较 → 功能矩阵与路径实验 → oracle repair → 决定是否设计新的联合重分配 writer。
