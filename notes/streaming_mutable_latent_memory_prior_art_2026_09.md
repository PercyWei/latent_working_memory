# 流式可变 Latent Working Memory：Soft Context Compression 与 Memory 方向查重

> 截止日期：**2026-09-03**。
> 文档目的：围绕一个流式 working-memory 设想，分两层核查已有研究是否已经覆盖其动机与技术结构。第一层从相近技术出发，整理 learned soft/latent context compression；第二层转向 soft context compression 与动态 memory 的交叉，重点检查 inference-time state update、历史 latent revision、可变容量以及连续有损更新。
> 证据约定：正式论文与预印本在正文中注明；压缩表示的数量、decoder 实际长度、持久状态字节数和端到端成本不视为同一个指标。

## 待核查的研究设想

该设想包含两个背景动机和一个暂定技术问题：

1. TTT layers、δ-mem 等方法将持续增长的上下文写入固定大小的矩阵或参数状态，成本较低，但固定容量可能随上下文增长而饱和；即使容量尚未耗尽，状态中内容增多也可能增加寻址和利用难度。这是需要实证检验的直觉，而不是已经成立的一般结论。
2. Soft context compression 提供了一个中间方案：不像完整 KV cache 那样保存全部 token，也不必把全部历史压进一个固定状态，而是维护数量远少于原文的 latent tokens。已有方法通常一次性压缩静态长文本，或逐 chunk 产生后不再改变的 latent；真实对话和持续文本流则要求 memory 随新输入不断演化。
3. 暂定技术问题是：同一 causal prefix 的整体重压缩与在线分块压缩之间是否存在稳定差距；如果存在，能否通过修改历史 latent 的在线 writer 缩小该差距，同时避免多轮有损更新造成远期信息持续衰减。

下文将“推理时状态更新”与“test-time optimization/TTT”严格区分。前者只要求输入到来时持久状态发生变化；后者通常还要求以测试样本定义损失并执行梯度或显式优化步骤。

# 第一章　Soft Context Compression：从“存储”到“使用”

## 1.1 研究对象与总体脉络

本章关注 **learned sequence-level latent/gist context compression**：writer 将原始上下文

$$
X=(x_1,\ldots,x_T)
$$

编码为更短的连续状态

$$
M=(m_1,\ldots,m_K),\qquad K\ll T,
$$

使后续生成主要依赖 $M$ 而非完整原始 token/KV。载体可以是 final-layer summary vectors、learned compression tokens 的逐层 KV、pooled hidden states 或 query-independent latent blocks。

传统 KV eviction/quantization、硬 prompt compression、RAG、TTT fast weights 和训练期参数 memory 与此方向相邻，但没有必要在本章逐项划界：关键判断标准是，方法是否训练了一个把当前输入实例编码成新连续语义状态的 writer。另一个需要保留的边界是，UniGist 等方法支持灵活的 execution chunk，并不等于最终表示已经对任意文本分块方式保持不变。

截至 2026-09，研究重点已经从“能否把上下文压成少量 latent”转向两类问题：

- **存储**：压缩载体、边界、容量、slot 分工以及 writer 能否保留未来可能需要的信息；
- **使用**：latent 与 decoder 表示空间是否兼容、reader 是否会调用其中的信息，以及是否需要按 query 恢复更高分辨率的证据。

## 1.2 存储：压缩表示怎样产生和分配

### 1.2.1 基础载体：summary、gist 与逐层 KV

| 工作 | 压缩载体与机制 | 已确认的边界 |
|---|---|---|
| [Gist Tokens](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html)，NeurIPS 2023 | 用特殊 attention mask 强制 instruction 信息经过少量 gist tokens | 主要压缩 task instruction，不是任意长文本流 |
| [AutoCompressors](https://aclanthology.org/2023.emnlp-main.232/)，EMNLP 2023 | 每段末尾产生 final-layer summary vectors，并把历段 summary 累积后输入下一段 | 状态随 segment 数增长；random segment length 不等于严格 partition consistency |
| [Sentinel Tokens](https://aclanthology.org/2023.emnlp-main.794/)，EMNLP 2023 | 将选定 span 的中间 activations 逐步压缩到 sentinel representations | 主要证据来自 language modeling 与开放式生成，精确长程检索验证较弱 |
| [Activation Beacon](https://arxiv.org/abs/2401.03462)，预印本 | 每若干 raw tokens 插入 beacon；删除原 token KV，只保留和累积 beacon 的逐层 KV | 属于逐 chunk、旧压缩 KV 冻结的流式 writer；训练中随机 ratio 不等于内容自适应容量 |
| [UniGist](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c93b738747776f9d3fdd077a5a07114-Abstract-Conference.html)，NeurIPS 2025 | 在完整训练序列上构造统一 causal gist layout，避免旧式 chunk-wise 深层循环训练图 | 改善的是训练和执行布局；固定 ratio/cell 下没有验证任意 offset 的行为一致性 |

AutoCompressors 的核心递推为

$$
\sigma_{<i}=[\sigma_1;\ldots;\sigma_{i-1}],
$$

并用未来 segment 的 next-token loss 训练 writer。其 Llama-2 版本可把 6,144 个历史 token 表示为 150 个 summary vectors，但仍明显弱于更长 plain-context full attention。Activation Beacon 则将 carrier 从 final-layer vectors 改为逐层 KV；论文报告约 2× inference acceleration 和 8× KV reduction，但这些系统指标不能直接解释为等比例的信息保留。

UniGist 在 4× compression、HELMET 汇总上的主结果如下：

| Backbone | Full attention | AutoCompressors | Activation Beacon | UniGist |
|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 63.9 | 33.6 | 51.8 | **60.4** |
| Llama-3.2-3B-Instruct | 51.9 | 23.9 | 37.9 | **47.8** |

这些数字支持 unified training layout 优于早期 chunk-wise writer，但 UniGist 使用了 16B continual-pretraining tokens 和 1B SFT tokens，且没有对同一文本的不同 partition 做成对比较。

### 1.2.2 压缩边界：从固定位置到动态与语义划分

[A Silver Bullet or a Compromise for Full Attention?](https://aclanthology.org/2025.acl-long.241/)（ACL 2025）把 gist compression 的典型失败概括为 boundary、surprise 和长精确字符串在生成途中丢失。后续方法分别从切段规则、内容难度和 query relevance 调整边界：

| 工作 | 边界或分配信号 | 主要限制 |
|---|---|---|
| [Surprisal-Based Dynamic Segmentation](https://openreview.net/forum?id=W9PYwtUbKh)，COLM 2025 | 累积 causal token surprisal 达到阈值后切段 | OPT-1.3B 上改进较小，阈值与实际压缩率缺少系统审计 |
| [Sentence-Anchored Gist](https://arxiv.org/abs/2511.08128)，预印本 | 在句末标点插入 gist | 简单可执行，但对标点规则敏感 |
| [DAST](https://aclanthology.org/2025.findings-acl.1055/)，Findings ACL 2025 | 结合 chunk perplexity 与全局 query attention 分配不同数量的 soft tokens | 依赖已知 query 和全局信息，不是 query-independent causal writer |
| [Density-aware Semi-Dynamic Compression](https://arxiv.org/abs/2603.25926)，预印本 | 按整篇输入从若干预设 ratio 中选择一个 | 是 global ratio selection，不是流内部的局部容量分配 |
| [SeCo](https://arxiv.org/abs/2605.09463)，预印本 | 以 query-relevant tokens 为 semantic centers，再按一致性聚合其他 token | 不按物理位置分块，但同样依赖已知 query |

这些工作证明“动态边界”并非单一概念。现有成熟度最低的仍是：未来 query 未知时，仅依赖已到达的 causal stream，为同一长流的不同局部分配不同数量的 slots。

### 1.2.3 Slot 分工与容量分配

[Cognitive Chunking for Soft Prompts（PIC）](https://arxiv.org/abs/2602.13980) 观察到 global soft-prompt compressor 训练后常自然形成近似对角的局部分工，因此直接用 block-wise causal mask 固定每个 memory token 的负责区域。论文报告 16× compressor 达到 baseline 峰值所需训练时间约减少 40%，但实验仍以较短 context 为主；它证明局部硬分工可能更易训练，没有证明流式表示能够逼近整体压缩。

[ComprExIT](https://arxiv.org/abs/2602.03784) 则走向相反方向：它用跨层 gated aggregation 保留中间层特征，再用 Sinkhorn optimal transport 联合分配 token anchors 到 compression slots，以减少多个 slots 重复覆盖同一区域。主实验 context 约 512，较长设置约 8K，因此更适合作为全局 slot assignment 的结构证据。

[No Mean Feat](https://arxiv.org/abs/2510.20797) 的 BenchPress 实验表明，mean pooling 和 bidirectional compression tokens 可以显著强于常见 causal compression-token baseline。这说明复杂 writer 的收益必须与 encoder 可见范围、表示对齐和训练预算共同解释，不能只归因于 token 形式。

[Cramming 1568 Tokens into a Single Vector and Back Again](https://aclanthology.org/2025.acl-long.948/)（ACL 2025）对每个样本直接优化连续向量，并找到可恢复最长约 1,568 tokens 的解。它给出的是 representation-capacity upper bound：单向量可能存在高密度编码，不代表 amortized、低成本、causal writer 能稳定找到这种表示，更不证明固定状态可无限可靠存储。

### 1.2.4 规模化训练与可部署压缩对象

[End-to-End Context Compression at Scale（LCLM）](https://arxiv.org/abs/2606.09659) 使用 0.6B encoder、4B decoder，并为每个模型训练超过 350B encoder/decoder tokens，分别支持 4×、8× 和 16×。其结果显示 encoder view、pooling、adapter 和 decoder adaptation 都显著影响压缩质量。encoder window 从 16 增至 256 时明显改善，增至 1024 时仍继续改善；简单增加邻窗 overlap 却没有改善 pretraining loss。这是“更完整的共同视野有利于压缩”的直接证据，但还不是整体压缩优于流式分段压缩的受控证明。

[CompLLM](https://arxiv.org/abs/2509.19228) 将约 20-token segments 独立压成可缓存、可跨 query 复用的 concept embeddings，并用完整上下文 teacher 的 answer hidden states 蒸馏 frozen decoder。它偏向语义信息，对 typo、字符计数、代码和 OOD language 较弱；独立 segment 的设计也没有检验任意 boundary offset。

[Training Transformers for KV Cache Compressibility（KV-CAT）](https://arxiv.org/abs/2605.05971) 从 backbone 侧说明：同一序列函数可以由易压缩或难压缩的 KV 表示实现。continued pretraining 中主动 sparsify/mask KV 能提高后续 compressibility，因此外挂 writer 的上限不仅取决于压缩器，也取决于基础模型是否学会了可压缩表示。

## 1.3 使用：compressed memory 怎样被 decoder 消费

### 1.3.1 Representation-space 对齐

[Semantic-Anchor Compression（SAC）](https://arxiv.org/abs/2510.08907)（ICLR 2026）不从随机 special tokens 学表示，而是选取原 context tokens 作为 anchors，并保留这些 anchors 的逐层 KV。这同时减小了 compression-token 身份差异和 final-hidden-state/逐层-KV carrier 差异；但其输入仍被固定切成约 510-token 子块，encoder 还是 bidirectional。

[GMSA](https://aclanthology.org/2026.acl-long.1324/)（ACL 2026）把 storage 与 alignment 拆开：group mean pooling 使每个 slot 均匀聚合对应组，Layer Semantic Alignment 再把 encoder 末层的高层抽象表示映射回 decoder 熟悉的低层空间。两部分消融都会显著下降，说明“写入是否合理”和“decoder 是否读得懂”是不同问题。

[Frozen LLMs are Native Decoders for High-Norm Semantic Vectors](https://aclanthology.org/2026.acl-long.1717/)（ACL 2026）发现成功 compressor 产生的连续向量 norm 往往远高于普通 token embeddings，改变 norm 会显著影响冻结 decoder 的重建。reader gap 因而可能同时包含语义、尺度与 attention dominance，而不只是抽象层级不一致。

### 1.3.2 从记住到会用

[Bridging the Memorization–Utilization Gap](https://aclanthology.org/2026.acl-long.682/)（ACL 2026）采用每 16 个 tokens 做 mean pooling 的简单 writer，再依次使用 continual pretraining、SFT 和 outcome-based RL。RL 后 decoder 会先隐式展开问题需要的局部内容，再完成推理。

论文所称“16× 下恢复超过 98% full-context performance”是 QA 成绩保持率，不是保留 98% 原文；每个 memory embedding 还包含两个 boundary tokens，因此 decoder 实际长度约缩短 5.3×。尽管如此，它有力说明高压缩表示的 storage capacity 与模型的 utilization ability 必须分别评估。

### 1.3.3 Query-conditioned 读取与选择性展开

- [Simplified Sparse Attention（SSA/H-SSA）](https://arxiv.org/abs/2604.20920) 先用 gist 浏览和路由，再恢复 top-k gist 对应的原始 chunks。它依赖原文或可重算 raw KV，因此更接近 learned index 与 selective expansion，而非不可逆 latent-only memory。
- LCLM-Agent 允许执行 `EXPAND(i)` 返回某个原始块，承认全局语义浏览和精确细节可以由不同载体承担。
- [SeDeM](https://arxiv.org/abs/2608.00311) 选择 query-relevant compressed blocks 后，将其解压为 decoder 中间层可消费的 hidden states，避免要求 decoder 直接从高度压缩 soft prefix 生成答案。其主压缩率为 4×，且 selector 使用 block-level evidence supervision，因此不能把最终提升全部归因于 writer 保真度。

### 1.3.4 成本必须按 write/read 生命周期统计

可缓存 latent object 的部署成本应写成

$$
C_{write}(X)+Q\cdot C_{read}(q,M).
$$

如果每段内容只被查询一次，encoder、test-time compilation、raw expansion 和 I/O 可能抵消 decoder 的节省。应分别报告 latent/byte state、实际 decoder 长度、写入成本、读取成本、TTFT 与峰值 memory，而不是用一个 headline compression ratio 代替。

## 1.4 第一层调研的客观结论

已经有人做过或部分解决的内容包括：

- 分段产生并累积 latent，以及用未来 token 监督 writer；
- 随机 segment/ratio、语义边界、已知 query 下的动态容量分配；
- 局部硬分工与全局联合 slot assignment；
- 原 token anchors、跨层聚合、latent–decoder alignment 和 norm calibration；
- 通过 CPT/SFT/RL 提高 compressed memory utilization；
- 用 gist 路由并按 query 恢复 raw tokens 或 intermediate hidden states。

仍缺少系统证据的问题包括：

- 相同 causal prefix、state budget 和 reader 下，整体压缩与不同 streaming partitions 的差距；
- query-independent、causal、local adaptive capacity；
- latent writer 的 storage fidelity 与 reader utilization 的因果分离；
- 多次状态改写后，远期信息是否以更新次数而非物理距离为主导发生退化。

# 第二章　从 Soft Compression 到流式可变 Working Memory

## 2.1 客观调研：动态压缩状态的四种形态

流式方法可以沿两个轴组织：持久状态是否增长，以及历史压缩状态是否改变。

| 形态 | 代表工作 | 状态转移 |
|---|---|---|
| 增长、旧状态冻结 | AutoCompressors、Activation Beacon、CCM-concat、UltraGist、Still | 每个新 chunk 产生并追加独立 latent/KV，旧表示不再修改 |
| 固定、旧状态更新 | CCM-merge、R³Mem、MELODI 的 STM、MemoryLLM | 新输入被递归写入有限状态，旧信息可能被覆盖或逐步遗忘 |
| 固定更新＋增长归档 | HMT、MELODI、M+ | 用固定 recurrent state 汇总近期或全局内容，同时保留窗口化或可增长的历史 bank |
| 可增长、选择性更新 | C-DIC、KVM | 根据当前输入修改合适的历史 slot/state row，并在容量允许或主题变化时创建新状态 |

这个表只比较持久状态的更新结构，不表示这些工作具有相同训练方式或载体：C-DIC 属于 soft latent dialogue compressor；KVM 是随模型训练的 attention/recurrent architecture；R³Mem 使用跨段 virtual tokens；MemoryLLM/M+ 更接近模型内生记忆。

## 2.2 客观调研：主要相关工作

### 2.2.1 增长但不修改旧 latent

[Compressed Context Memory（CCM）](https://arxiv.org/abs/2312.03414)（ICLR 2024）直接研究 online continual context。`CCM-concat` 将各步产生的 compressed KV 持续追加，保留能力随状态增长而提高；`CCM-merge` 则用累计平均维持固定状态。它已经给出“增长但冻结”与“固定但更新”两个端点，却没有维护一个既可增长又可重写的 latent bank。

[UltraGist](https://arxiv.org/abs/2405.16635) 明确把新 context 增量压缩并加入既有 compression result；实现上把旧 gist KV 与新 gist KV 拼接，因此仍属于 `w/o change`。它和 Activation Beacon 来自同一作者路线，不应作为两个独立思想重复计数。

[Still](https://arxiv.org/abs/2606.07878) 让 compact chunks 与一个 raw working chunk 共存；新 raw chunk 在旧 compact prefix 条件下编码，再压缩成新的 compact KV 并追加。状态以低于原文的速度增长，但旧 compact chunks 不会被重新写入。

[MAC](https://arxiv.org/abs/2403.04317) 把每个到来的文档一次性压成 compact modulation，并存入可检索的 memory bank。它支持在线增加文档，但旧 modulation 保持静态，关注点是模块化存储与查询聚合，而非连续 revision。

### 2.2.2 固定或混合 recurrent state

[Hierarchical Memory Transformer（HMT）](https://arxiv.org/abs/2405.06067) 用当前 segment 的 summary query 检索历史 memory embeddings，再把相关历史、近期 sensory tokens 和当前段共同编码为新的 memory embedding。新状态递归包含先前内容，而旧缓存本身不修改；论文也明确指出越早的信息经历的损失更大。

[MELODI](https://arxiv.org/abs/2410.03156) 同时维护两级状态：固定大小 short-term memory 每个窗口都会读取旧 STM 并产生新 STM；high-capacity long-term memory 则把各窗口的 compressed KV 追加进有界 FIFO。它直接暴露了两种策略的互补关系：mutable fixed state 成本稳定但容易忘记，append-only long-term bank 提高容量但继续增长。

[R³Mem](https://aclanthology.org/2025.findings-acl.235/)（Findings ACL 2025）在每个 segment 前后插入固定数量的 read/write virtual memory tokens；上一段的 write outputs 成为下一段的 read inputs。其 reversible Transformer 与 context–query hierarchical training 针对 retention/retrieval 损失，但持久载体仍是固定数量的 recurrent tokens，而不是可变 slot bank。

[MemoryLLM](https://arxiv.org/abs/2402.04624) 使用固定大小 latent pool，将新信息写入部分 slots 并替换旧状态；[M+](https://arxiv.org/abs/2502.00592) 进一步把被 STM 淘汰的状态放入可增长的 long-term memory，再按 query 检索。M+ 因而是“mutable fixed STM＋append-only growing LTM”，尚未修改已经进入 LTM 的旧 slots。

[Dynamic LSTM-based Memory Encoder For Long-term LLM Interactions（Pref-LSTM）](https://arxiv.org/abs/2507.03042) 用门控公式更新固定 memory vector，再将其投影成 pseudo-token soft prompt。它与流式 state update 在结构上相关，但任务局限于偏好记忆，而且论文没有观察到有效的 preference-following 改善。

### 2.2.3 可增长状态与历史 slot revision：最直接的重合

[Context-Driven Incremental Compression（C-DIC）](https://arxiv.org/abs/2606.12411)（ICML 2026）维护一组可修改的 per-thread latent states，每轮执行：

```text
query current memory
        ↓
retrieve relevant thread states
        ↓
compress retrieved states + current turn into new state
        ↓
topic continuation: replace best-matching slot
topic shift: insert a new slot
```

它是目前与“inference-time update＋历史 latent revision”最接近的 soft context compression 工作。需要精确理解其更新：推理阶段完全 gradient-free；被选中的旧 slot 不是原位梯度优化，而是由 learned compressor 重新编码并整体替换。总 memory bank 没有形式上的硬上限，只有生成时检索的子集可以保持很小。

[Key-Value Means（KVM）](https://arxiv.org/abs/2605.09877)（2026 预印本）维护一个可固定或按计划增长的 compressed KV state。旧 block 离开 sliding window 后：

1. 预设 budget 决定本轮可以增加多少 rows；
2. 与已有 state 最不相似的 overflow tokens 优先成为新 rows；
3. 其余 tokens 被路由到最相似的非 sink row；
4. learned merge gate 控制它们对该 row 的 key/value running sums 的贡献。

KVM 因而实现了 inference-time `append-or-merge`，已有 state rows 会真实改变。但它不是对全部历史 latent 的联合重优化：新增 slot 数由静态 budget schedule 决定，旧 rows 之间也不会根据新全局上下文重新划分。它还是原生 attention/block-recurrent architecture，而不是面向冻结 decoder 的独立 soft compressor。

### 2.2.4 Storage 与 use 已结合成 memory system

[LycheeMemory](https://aclanthology.org/2026.acl-long.365/)（ACL 2026）把每个 4096-token chunk 压成 KV-style memory blocks，再由 Gate 根据 query 和当前 plaintext working memory 选择 blocks，Reasoner 通过多步推理持续更新 working memory。其重点不是修改旧 compressed blocks，而是把压缩存储、选择性读取和动态显式工作记忆组合为完整系统。

这项工作之所以重要，是因为它把 context compression 从“静态表示替换”推进到了 memory 生命周期：writer 产生长期压缩块，gate 决定何时读取，reasoner 把读取结果写入短期 working memory。它也表明，仅改进 latent storage 不必然解决跨步使用和 late binding。

还有几项工作虽然直接引用 CCM 或共享在线动机，但载体不同：

- [InfiniteICL](https://arxiv.org/abs/2504.01707) 通过连续参数更新处理 streaming contexts，属于 TTT/parameter memory，而非 latent-token bank；
- [Generative Adapter](https://arxiv.org/abs/2411.05877) 为完整 context 一次性生成低秩 adapter，没有递归修改此前状态；
- 文本摘要、RAG 和 KV pruning 可以作为系统 baseline，但不能直接证明 learned latent writer 的行为。

## 2.3 客观证据：三个关键命题分别得到多少支持

### 2.3.1 固定大小状态会不会随上下文增长而耗尽

现有证据支持“固定状态在 recall-heavy、低冗余长流中更容易退化”，但没有证明所有固定状态必然在某一长度耗尽。KVM 的 growable schedule 在长程 recall 上优于固定状态，MELODI 和 M+ 也都通过额外长期 bank 弥补固定 STM；然而这些结果同时改变了容量、结构和训练方式，不能单独识别容量耗尽机制。

因此，该动机适合作为背景和实验假设，不适合作为无条件定理。需要在同一 writer/reader 下逐步增加上下文的信息密度，并区分：物理容量不足、寻址竞争、旧信息覆盖以及 decoder 不会利用。

### 2.3.2 整体压缩是否优于分块压缩且旧状态冻结

LCLM 的 encoder-window 实验表明，更大的共同可见范围通常提高压缩质量；gist failure study 也证明物理边界会丢失跨段信息。但目前没有工作在同一 causal prefix、latent/byte budget、reader 和训练量下，系统比较：

- 每一步重新读取完整前缀并整体压缩；
- 分块压缩并冻结历史 latents；
- 不再读取原文、只修改既有 latents 的在线压缩。

“整体压缩更好”因此仍是待验证假设。长输入上的整体 writer 也可能因为固定容量过载、优化困难或训练长度失配而弱于具有局部归纳偏置的分块方法。

### 2.3.3 多轮有损更新是否造成累计损伤

C-DIC 给出了目前最直接的证据：把为 one-shot compression 训练的 ICAE 反复应用到自身 latent outputs，会产生 latent drift 和 error compounding，生成质量在连续数轮后迅速下降。ICAE-append 和每轮重新编码完整前缀在中短长度更稳定，但最终因状态或输入增长而 OOM。

这证明“naive `w/ change`”可能比 `w/o change` 更差，也说明连续 revision 必须通过长程 unroll、专门的更新目标或结构约束训练。它尚未证明 C-DIC 的局部 revision 已经逼近同预算 whole-prefix compression。

## 2.4 综合讨论：重合边界与仍可研究的问题

### 2.4.1 与当前设想的重合程度

从状态转移逻辑看，重合度最高的是 C-DIC 和 KVM：二者都在推理时根据新输入修改历史压缩状态，并在需要时建立新状态。但更准确的共同表述是：

> **inference-time state update + revisable persistent latent/KV state**

而不是“测试时优化历史 latent”。C-DIC 是语义 thread retrieval 后的重编码与替换；KVM 是预设容量计划下的局部相似性 merge。二者都没有根据完整新前缀重新联合分配所有历史 latent 的内容。

部分重合的工作包括：

- Still、UltraGist、CCM-concat：支持流式增长，但历史 latent 冻结；
- R³Mem、CCM-merge：历史状态持续更新，但表示长度固定；
- MELODI、HMT、M+：把固定 recurrent state 与追加式历史 bank 组合；
- LycheeMemory：重点解决 compressed memory 的检索与多步使用，而不是历史压缩块 revision。

因此，“流式输入”“历史 latent 会改变”以及“merge-or-create”分别都已有明确先例；它们可以作为大背景引出问题，不能单独承担 novelty。

### 2.4.2 更可守的技术问题

仍未被完整覆盖的问题可以写成：

> 在相同 causal prefix、持久 state budget 和 reader 下，能否通过容量受控的 mutable latent memory，缩小在线分块压缩相对于 whole-prefix recompression 的行为差距，同时抑制多轮重写的累计损伤与 partition sensitivity？

它比“首次做流式 soft compression”或“首次修改旧 latent”更窄，也更可证伪。其真正区分点可能包括：

1. **same-prefix global oracle**：teacher 只看到当前已到达的完整前缀，不偷看未来文本或未来 query；
2. **matched persistent budget**：匹配 latent 数、字节数、索引/metadata 和被保留的 raw state；
3. **paired multi-partition evaluation**：对同一文本改变 chunk size、offset 和语义切分，而不是只在训练时随机分段；
4. **global or selective reallocation**：旧 latent 不只是吸收新内容，还能在新证据到来后改变 slot 分工；
5. **long-unroll retention**：按一条信息经历的更新次数测量退化，而不只按 token 距离汇总。

最低限度的三组 writer 对照为：

$$
G_t=C(x_{\le t}) \quad \text{whole-prefix recompression},
$$

$$
A_t=A(A_{t-1},x_t) \quad \text{append-only / frozen history},
$$

$$
U_t=U(U_{t-1},x_t) \quad \text{mutable online state}.
$$

只有先验证 $G_t$ 相对 $A_t$ 存在稳定优势，再验证 $U_t$ 能缩小这一差距，整体压缩与分段压缩的 gap 才足以成为方法动机。Global 可以被定义为允许重读前缀、计算更昂贵的 oracle，但必须明确区分表示容量差距和额外写入计算带来的优势。

### 2.4.3 “使用”可以成为第二条技术线，但应先诊断

动态 revision 还可能产生一个静态 compressor 较弱的问题：同一 slot 的语义角色随时间变化后，reader 的地址、尺度和表示分布可能漂移。第一章中的几条路线可提供机制灵感：

- SAC、GMSA 与 high-norm work：稳定 carrier 身份、层级语义和几何尺度；
- Near-Lossless：通过专门训练让 decoder 学会调用或隐式展开 latent；
- SSA/H-SSA、SeDeM、LycheeMemory：把压缩存储与 query-conditioned retrieval/decompression 分开；
- address–content decoupling：让可检索地址与信息丰富的内容使用不同表示，减少 revision 对寻址稳定性的破坏。

但 storage 和 utilization 不应在第一轮实验中同时变化。较清楚的验证顺序是：先用同一固定 reader 比较 global、append-only 和 mutable writers；再固定 writer 比较不同 readers；最后做 `writer × reader` 交叉实验。否则无法判断差距缩小来自更好的状态更新，还是更强的检索与解码能力。

### 2.4.4 当前结论

两个原始动机都已有相关研究讨论，但仍足以作为论文的大背景。需要避免的不是复用这些动机，而是把已经出现的 inference-time revision 或可增长状态误写成首次提出。

当前最值得先做的是一个小规模、严格匹配的 go/no-go 实验：确认 whole-prefix、append-only 与 mutable compression 的差距是否真实存在，差距来自 storage 还是 use，以及专门训练的 mutable writer 能否在长程更新中稳定缩小差距。若 whole-prefix 本身没有稳定优势，或简单的 append-only/partition augmentation 已能达到相同结果，则不应继续围绕该 gap 构造复杂方法；若优势随分块扰动和更新次数系统扩大，则后续的 global-teacher、capacity reallocation 和 reader design 才有充分依据。
