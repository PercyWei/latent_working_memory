# 20260909_真实公开数据与记忆训练路线（20:13:30 UTC+08:00）

创建时间：20260907 22:31:33 UTC+08:00

最后修订时间：20260909 20:13:30 UTC+08:00

本文整理公开数据的训练用途、相关文献的训练配方及本项目的数据协议。结构与计算公式见 [可增长 Latent Working Memory 框架 v1](20260907_growing_latent_working_memory_framework_v1.md)。本文中的方案是实验执行依据，效果由对应训练与评估验证。

## 1. 结构与训练主线

系统在冻结的语言模型基座上增加五项可训练模块：写入投影 $W_{\mathrm{in}}$、联合更新器 $U$、读取投影 $P$、读取 LoRA 和容量网络 $V$。前四项构成记忆读写参数 $\eta$，容量网络参数记为 $\psi$。

写入时，基座处理当前文本，$W_{\mathrm{in}}$ 将内部表示映射到记忆空间，$U$ 结合旧记忆与新输入生成全部记忆位置。读取时，$P$ 将记忆映射为基座的输入向量，启用读取 LoRA 的基座完成当前任务。$V$ 在已有记忆的状态上选择增长数量。

记忆从形状为 $[0,512]$ 的空张量开始，动态阶段首次写入分配 16 个位置，随后按动作 $\{0,8,16\}$ 增长至上限 512。新位置采用零内容初值与固定位置向量。共享可学习初值是独立的结构消融项。

训练分为三个阶段：

| 阶段 | 学习能力 | 数据与目标 | 更新参数 |
|---|---|---|---|
| 记忆读写预训练 | 将原文保存为可读取的记忆 | 公开原文的自重建（AE）与后续语言建模（LM） | $\eta$ |
| 动态记忆训练 | 连续写入、历史保持和增长后的联合重组 | MSC 人工下一回复；QA 数据用于事实保持实验 | $\eta$ |
| 容量决策训练 | 根据任务收益与资源代价分配位置 | 固定读写模型在同根动作分支上的代价标签 | $\psi$ |

三个阶段共享相同的基座、记忆结构和读取接口。预训练 checkpoint 直接进入动态阶段；动态 checkpoint 固定后提供容量监督。数据来源、规模和停止点依据有效监督量与独立 dev 学习曲线确定。

## 2. Context compression 文献的训练数据与启示

相关文献采用原文自监督、下游任务监督或两者结合。数据名称需要与实际使用的字段、目标及训练阶段一同解释。

| 工作 | 训练数据与目标 | 对本项目的意义 |
|---|---|---|
| [ICAE，ICLR 2024](https://arxiv.org/html/2307.06945v4) | 在 The Pile 原文上进行 AE 与 LM 预训练；指令适配使用 PwC，其问题和答案由 GPT-4 基于原文生成 | 原文重建与续写可以先建立可读取的记忆表示 |
| [PCC，ACL 2025](https://aclanthology.org/2025.acl-long.1394.pdf) | 使用 FineWeb 进行预训练，包含 32M tokens 的短文本预热和 5B tokens 的后续训练；分别在 SQuAD、GSM8K、HPD 上适配 | 通用原文训练与任务适配分别承担基础表示和下游使用能力 |
| [PISCO，2025 原论文](https://arxiv.org/html/2501.16075v1#S4.SS1) | 使用约 453k 个 multi_qa 问题、检索得到的 Wikipedia-KILT 文档及 teacher 答案，进行序列级蒸馏 | 任务监督可以直接训练压缩记忆；标签来源与读取任务共同决定训练范围 |
| [ComprExIT，2026 预印本](https://arxiv.org/html/2602.03784v4#S6.SS1) | 在 SlimPajama 的 1B tokens 上进行 next-token 预训练，再使用 MRQA 进行任务训练 | 原文预训练加公开 QA 适配具有直接的实现参照 |
| [Cognitive Chunking，2026 预印本](https://arxiv.org/html/2602.13980v1#S4.SS1.SSS1) | FineWeb 3B tokens 预训练，随后在 SQuAD、GSM8K 上适配 | 分块表示也通过原文与任务数据分阶段学习 |
| [GMSA，ACL 2026](https://aclanthology.org/2026.acl-long.1324.pdf) | 重建实验使用 PwC 的原文 context；通用任务模型经过 AE 预训练，再使用 NQ、2WikiMultihopQA、HotpotQA、NarrativeQA、MultiNews，各取 20k 条进行任务训练 | 同一数据资源的原文字段与问答字段可以服务不同目标 |
| [C-DIC，2026 预印本](https://arxiv.org/html/2606.12411v1) | 从预训练 ICAE 初始化，在 MSC 的 1,001 条完整四 session 训练轨迹上，以人工回复 NLL 和 retrieval-aware TBPTT 训练 2 epochs | 动态适配可以集中在一个阶段，已有预训练表示为该阶段提供起点 |
| [LCLM，2026 预印本](https://arxiv.org/html/2606.09659v1#A3) | 使用 Nemotron、OLMo 长文等语料，重建数据包含 FineWiki、RedPajama arXiv、代码与数学文本；随后进行长上下文任务适配 | 大规模训练覆盖基础记忆能力与任务使用能力，具体阶段服务其架构和参数训练范围 |

这些结果支持本项目的阶段划分：先以公开原文训练读写能力，再以真实顺序任务训练动态记忆。C-DIC 的 retrieval-aware TBPTT 是动态阶段的梯度传播方式，其训练起点已经包含 ICAE 预训练。容量决策在本框架中具有独立参数与反事实监督，因此形成第三阶段。

本项目的预训练目标取自公开原文，动态回复目标取自人工对话，QA 目标采用来源数据的标注。文献中的任务适配数据用于解释各自的训练配方，具体接入资源由下文的数据路线确定。

## 3. 原文预训练与 MSC 动态训练

### 3.1 原文 AE 与 LM

预训练使用 [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 的 `sample-10BT` 子集。原文通过字段与最短长度检查后，按来源与近重复簇去重并固定 train/dev/test，再构造段落、连续句子组、单句和相邻段落组。最终样本的内容质量由强模型判定。

完整自然父片段为 $S$。AE 写入并重建 $S$；LM 在 $S$ 的合法内部句界随机切分，写入前缀 $X$，监督剩余后缀 $Y$ 的全部 tokens 与 EOS。两项分别构造成独立 episode，每个 episode 完成一次写入与一次任务读取。实际写入文本 $T$ 的记忆为：

$$
M_T=\operatorname{Write}_\theta(\varnothing,T;K_T),
$$

$$
\mathcal L_{\mathrm{pre}}
=\lambda_{\mathrm{AE}}\mathbb E[\ell_\eta(M_S,q_{\mathrm{AE}},S)]
+\lambda_{\mathrm{LM}}\mathbb E[\ell_\eta(M_X,q_{\mathrm{continue}},Y)].
$$

基座完整处理每个实际写入片段。当前输入上限为 1024 tokens，LM 目标上限为 256 tokens；单句父片段用于 AE。容量根据实际输入长度按压缩率 $\{2,4,8\}$ 采样，取整后落在 $[1,512]$ 并满足完整读取预算。每次参数更新分别按两项任务的有效样本数归一化损失，权重由配置指定。

数据构造完整句界 `semantic` 与随机截断 `random` 两套独立版本，共用固定来源划分。先构造 semantic 并由 Qwen3.8-27B 判定最终 X/Y，再统计各 split、各任务的入选输入长度区间；random 根据该分布独立选择来源和原文跨度，并接受模型判定。两类提示词共享内容标准，分别处理完整句界和随机截断要求。

每个版本、每个 split 的 AE 与 LM 各自达到相同的最终入选配额，示例配置为每项任务 train/dev/test 各 100000／2000／2000 条。random 匹配 semantic 的任务内输入区间计数，LM 目标长度与监督 token 量单独记录。数据完成后通过独立模型或人工抽查评估质量，正式训练验证使用 semantic。代码流程与验证范围见 [独立 AE/LM 数据构造实现](v1/20260909_independent_ae_lm_data_preparation.md)。

预训练评价包括重建 NLL、token 准确率、序列匹配和续写 NLL/PPL，并比较正确记忆、空记忆和其他文档的记忆。16 条公开 train 样本的过拟合用于链路调试；正式训练从统一初始化开始，以独立文档上的记忆利用作为推进依据。

### 3.2 MSC 的数据性质与使用单位

[MSC](https://parl.ai/projects/msc/)由人工众包的多 session 角色对话组成，人物设定和 session 时间间隔属于采集任务条件。其三 session 训练轨迹有 4,000 条，四 session 轨迹有 1,001 条；完整四 session 轨迹平均约 53.3 条 utterances。官方任务视图约含 237k 个训练 examples，完整会话链与逐回复 examples 分别计数。[MSC 原论文](https://arxiv.org/pdf/2107.07567)

MSC 的人工下一回复直接提供训练目标。回复的开放性通过条件语言建模处理：训练最大化数据集中实际回复的概率，生成质量使用对话指标与分层评估衡量。它为连续读写提供较密集的监督，并保留跨 session 历史依赖。

训练单位为原始会话链。同一链条的多个 session、共享前缀及派生窗口归入同一 split；三 session 与四 session 视图按原始链条标识归并。正式实验记录完整链条数、session 数、目标回复数和目标 tokens。

### 3.3 读取、写回与梯度传播

对轮次 $k$，已有历史记忆为 $\mathcal M_{k-1}$，当前发言为 $q_k$，人工回复为 $r_k$：

$$
\mathcal L_k=\ell_\eta(\mathcal M_{k-1},q_k,r_k),
$$

$$
\mathcal M_k=
\operatorname{WriteSequence}_\theta
\left(\mathcal M_{k-1},\operatorname{Serialize}(q_k,r_k)\right).
$$

执行顺序为：读取旧记忆与当前发言、计算回复损失、写入已完成的一轮交互。当前人工回复从下一轮起成为历史。闭环推理写回实际生成的回复；人工历史条件与闭环生成条件分别评价。

每轮完成后封闭写入事件，事件内部按固定 cells 切分，尾部保留实际长度。主动态训练每次联合更新最多处理 2 cells。更新器沿自身产生的记忆连续运行，session 边界沿用已有记忆。

该阶段继续更新 $W_{\mathrm{in}},U,P$ 和读取 LoRA。外部增长安排覆盖保持容量与多种增长速度，使旧位置更新和新增位置利用共同获得训练。短轨迹采用完整 BPTT，长轨迹采用 TBPTT；1024 个写入 tokens 是初始候选窗口，最终长度依据历史依赖跨度和显存确定。

每个 TBPTT 段在边界前完成有效读取损失的反传，随后传递 detached memory。同一 episode 的各段累积梯度，episode 结束时执行 optimizer step。长程评价同时记录信息写入到目标回复的距离、更新次数及反传覆盖范围。

## 4. 容量监督与训练验收

容量训练使用动态模型产生的真实记忆轨迹，固定完整读写参数 $\eta$。每个已有记忆的根状态分别执行合法动作 $g\in\{0,8,16\}$，各分支共享后续文本、读取边界、提示和参考答案，后续动作保持 $g=0$。

首轮标签范围为当前 episode 的剩余部分。根状态的后续区间至少包含一个有效目标；已可回答的旧问题与新到达问题按统一调度进入分支。目标由人工回复或原始 QA 提供，记忆状态由模型自身递推。

$$
J(g)=\overline\ell^{(g)}
+\lambda_s\overline C_{\mathrm{storage}}^{(g)}
+\lambda_w\overline C_{\mathrm{write}}^{(g)}
+\lambda_r\overline C_{\mathrm{read}}^{(g)},
\qquad
\Delta J(g)=J(g)-J(0).
$$

容量网络以 SmoothL1 拟合 $\Delta J$。代价归一化和初始资源权重见框架第 6.4 节。标签绑定完整读写 checkpoint，记录合法动作、监督数量、剩余长度和根状态覆盖率。

这组标签定义“当前增长、后续保持容量”的条件价值。学习策略通过完整连续运行评价，比较固定容量、计划增长、追加式存储与联合增长的质量—成本曲线。分支纯质量差 $\Delta\ell$ 和加资源项后的 $\Delta J$ 分别报告。

阶段验收依据为：

| 阶段 | 核心证据 |
|---|---|
| 预训练 | 独立原文上的重建与续写提升，以及正确记忆带来的稳定收益 |
| 动态训练 | 多轮保持、长轨迹稳定性和同根扩容分支的可重复质量收益 |
| 容量训练 | 实际策略相对简单容量安排改善质量—成本权衡 |

策略访问新的状态分布时，在固定读写模型下补充根状态标签并更新容量网络。策略轨迹上的读写退化触发短程动态适配，新的读写 checkpoint 对应重新生成的容量标签。适配与刷新由 dev 结果触发。

## 5. QA 与时间流的实验分工

MSC 承担动态主训练；QA 数据提供证据明确、答案可计分的事实保持与组合实验。SQuAD/MRQA 分别报告 MSC 训练模型的直接迁移，以及使用 QA train 适配后的结果。QA 适配属于动态训练阶段，数据与计算预算按实验条件记录。

| 数据来源 | 实验角色 | 数据处理与结论范围 |
|---|---|---|
| [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/) | 基础事实摄入与保持 | 同一文章已发布段落按原序组成多问流，窗口固定后对齐人工问题与答案 spans |
| [MRQA](https://github.com/mrqa/MRQA-Shared-Task-2019) | 多来源 QA 适配及域外泛化 | 训练来源为 SQuAD、NewsQA、TriviaQA、SearchQA、HotpotQA、NQ；按官方训练、域内与域外评估协议组织 |
| [HotpotQA](https://hotpotqa.github.io/) | 跨段组合 | 保留完整候选上下文及 supporting facts，衡量给定候选上下文中的流式组合能力 |
| [QASPER](https://huggingface.co/datasets/allenai/qasper) | 困难文档验证 | 5,049 问、1,585 篇论文；保留论文顺序、多参考答案及其证据，全文与派生窗口分表报告 |
| [Natural Questions](https://github.com/google-research-datasets/natural-questions) | 真实搜索提问分布与 QA 扩展 | 使用带原文和答案标注的数据，完成 HTML 到 token 的证据对齐；单独记录 MRQA NQ 子集与原始 NQ 的来源关系 |
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | 困难多跳组合 | 使用公开组合问题及原始段落，按官方清单隔离 dev/test 的单跳种子来源 |
| [StreamingQA](https://github.com/google-deepmind/streamingqa) | 自然时间到达 | 选取人写问题，连接带日期的新闻原文与 evidence ID，按文章发布时间和问题时间调度 |
| [DynaQuest](https://github.com/nusnlp/DynaQuest) | Wikipedia 修订与事实更新 | 将公开问题、答案和 revision ID 连接到新旧版本原文，标注修订时间、对应事实及其有效范围 |

SQuAD 使用的数据单位是已发布的文章段落集合。HotpotQA 候选上下文本身经过任务组织，其结果解释为该候选集合中的保持与组合能力。QASPER 首轮作为独立文档验证，评价专业文本中的长期依赖与答案生成。

StreamingQA 的人写问题与真实新闻流用于新知识摄入和保持。[原论文](https://proceedings.mlr.press/v162/liska22a.html)提供其时间流设置。DynaQuest 修订实验分别记录页面变化的观察时间与事实在现实中的有效时间；新旧事实及适用条件构成更新指标的标注依据。[DynaQuest 论文](https://aclanthology.org/2025.findings-acl.1380/)

时间泛化按两种协议分表报告：页面隔离协议将每个页面的完整轨迹归入同一 split；已见页面未来到达协议固定参数训练截止时刻，并让后续版本按时间更新评估记忆。每项实验记录可见历史、起始记忆及评估时间范围。

## 6. 数据契约、监督边界与评估

### 6.1 统一 episode 与来源隔离

episode 使用单一 JSONL 契约，核心字段为 `episode_id, input_ids, write_ends, sources, reads`。每个读取目标包含 `read_id, task, prefix_end, prompt, references`，任务类型为 `ae, continuation, dialogue, qa`。字段定义及执行调度见框架第 8.2 节。

写入输入按原始来源顺序构造。文档窗口依据预先固定的长度与结构规则划分，问题和证据在窗口确定后对齐。公开数据的改良版本保留文本、角色、日期和参考标注的来源关系。

来源划分先于窗口构造。同源文档、会话链、重复前缀和派生样本归入同一 split；跨数据集复用也按来源归并。[MuSiQue 官方清单](https://github.com/StonyBrookNLP/musique#data)用于隔离来自 SQuAD/NQ 的 dev/test 单跳种子，[LongBench](https://github.com/THUDM/LongBench)结果按组成数据的原始来源处理。

### 6.2 QA 证据与多参考答案

对参考答案 $y$ 对应的一份充分证据集合 $E_y$，定义：

$$
b_y=\max_{e\in E_y}\operatorname{end}(e).
$$

计分位置取第一个覆盖 $b_y$ 的共同提交边界。该位置表示标注能够支持的充分证据到达点。多份证据分别绑定对应答案，在每个前缀使用已有证据支持的参考答案。

QASPER 按 annotator 保存答案与证据关系，依赖完整文档判定的答案在全文结束处评价。[QASPER 原论文](https://aclanthology.org/2021.naacl-main.365.pdf)同时包含抽取、生成、Yes/No 和不可回答等类型，各类型沿用来源标注。

QA 读取点使用至多 3 个不同的合法问题，按实际有效数量归一化。合法问题池为空的前缀继续写入，读取损失在后续有效目标处计算。重复读取旧问题衡量保持，单个问题在 episode 中的累计训练权重受限。

多参考训练按固定规则选择一个当前合法目标，同一次容量分支比较保持目标一致。训练保留完整 gold，评估使用官方多参考规则；生成上限按任务在 dev 上固定，并报告达到上限的比例。报告有效问题数、监督前缀覆盖率、证据到达位置及保持距离。

### 6.3 对照、资源与统计

预训练报告 AE 与 LM 指标；MSC 报告 token 加权 NLL/PPL、BLEU、ROUGE，并按 session 和历史距离分层；SQuAD/HotpotQA 使用官方 answer EM/F1，QASPER 使用官方 Answer F1。证据标注服务监督边界与距离分析。

记忆利用对照包括空记忆、错误来源记忆、近期原文和完整已到达原文。完整原文参考使用冻结基座，其合法上下文预算包含历史、提示、目标和特殊 tokens；长文本结果记录该参考的实际覆盖范围。

固定容量、计划增长、追加式、学习增长和全局重读系统匹配基座、tokenizer、训练数据、读写适配预算和容量机会，各结构训练配套读取接口。联合重组的机制对照进一步匹配容量轨迹；全局重读系统在相同容量下重新处理完整前缀。

存储采用实际 bytes 与随源长度累积的占用量。计算分开记录基座文本前向、记忆更新、读取时间、TTFT 和峰值显存，训练成本按三个阶段分别计量。正式结果采用至少 3 个模型 seeds，并按源文档或会话链聚类进行成对 bootstrap。

输出蒸馏、分块一致性和可学习初值各自构成独立消融，使用相同来源划分和训练预算。主路线依次验证可读记忆、动态保持与容量决策，扩展任务分别检验事实保持、组合推理和时间更新。
