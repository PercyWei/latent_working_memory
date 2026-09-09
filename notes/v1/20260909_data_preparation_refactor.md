# 20260909_数据准备流程重构（15:15:21 UTC+08:00）

创建时间：20260909 11:40:55 UTC+08:00

最后修订时间：20260909 15:15:21 UTC+08:00

数据准备统一位于 `src/latent_working_memory/data_preparation/`，训练、模型和评估位于 `src/latent_working_memory/v1/`。准备入口读取本地公开语料与已安装模型，输出可供训练直接读取的数据及审计记录。

## 1. 模块与流程

| 模块 | 职责 |
|---|---|
| `__main__.py` | 命令行入口，解析实验配置、准备配方、本地路径与设备 |
| `config.py` | 数据准备配方及持久化边界校验 |
| `sources.py` | 混合读取本地 Parquet 文件、row group 和批内文档 |
| `quality.py` | 文档粗筛、句段规则与待复核特征统计 |
| `segmentation.py` | 段落、句界及原文字符位置 |
| `dedup.py` | 文档 ID、URL、全文精确归并及词 5-gram 近重复聚类 |
| `scoring.py` | 独立窗口 NLL、模型句段判定与评分缓存 |
| `fineweb.py` | 自然粒度候选、AE/LM 目标、容量合法性和数据契约 |
| `pipeline.py` | 候选池、划分、筛选、样本输出和完成记录 |
| `audit.py` | 原文连续性、来源隔离、长度统计与采样预演 |
| `inspection.py` | 完成数据的独立随机与分层抽查、复核结果汇总 |

执行顺序为：候选读取与粗筛 → 文档聚类及来源划分 → 已配置模型评分 → 自然片段构造 → 来源与预算审计 → 写入完成记录。候选池固定后，近重复簇 ID 使用簇内最小归一化 URL；各簇按固定哈希分配 train/dev/test，保留按候选顺序遇到的首篇具有合法视图的文档。

近重复判定采用归一化词 5-gram 的 Jaccard 相似度。全局词频顺序的前缀索引缩小候选范围，再计算完整集合交集确认相似度；默认阈值为 0.9，至少 64 个词的文档参加近重复比较。完整原文和来源精确去重覆盖全部长度。

## 2. 样本、评分与采样

每个合法输入 X 具有 AE 目标；存在合格紧邻原文 Y 时附加 continuation 目标。episode 的 reads 顺序为 AE，或 AE 后接 continuation。AE-only 视图的 y_char_span 与 continuation_sentence_range 为 null。文末完整句段和噪声前的合格输入可以参与 AE。

段落、连续句子组、单句和相邻段落组沿原文边界构造。句子组从连续合格区间采样；每个粒度的视图选择轮流覆盖候选具有的长度区间。续写终点从 token 预算内的合法完整句末位置均匀采样，目标文本通过原文字符区间直接提取。离线主题标注提供连续句子索引范围。

NLL 评分将全文划为连续 token 窗口，各窗口独立添加 BOS、独立前向，再按有效目标 token 数汇总。记录包含每个窗口的位置、token 数及 NLL 总和。NLL 作为诊断信号持久化，筛选决定来自规则及已配置的模型判定。

语义质检模型读取带编号的完整句子块，返回 decision、reason、rejected_sentences。decision 取 keep、reject 或 review；局部缺陷转为原文字符区间，reject 与 review 对应的完整块进入排除区间。超出质检模型上下文的完整句子记录为 review。模型原始输出与判定理由一起保存，标准 JSON 契约在模型输出边界检查。

评分缓存按文档原文、模型、提示词和评分协议区分。修改输入长度采样或训练学习率时可以复用相同评分。模型加载统一使用 local_files_only=True；模型在原目录替换后应使用新的评分缓存。

input_length_weights 配置长度采样。启用后依次选择长度区间、文档、粒度、视图和 memory 容量；每个长度区间内部按打乱的文档队列循环访问。正权重区间需要有合法候选。input_length_weights 为 null 时使用文档循环采样。全部队列、游标、访问次数与 RNG 状态进入 sampler checkpoint。

新的 `configs/v1/pretrain_a800.json` 提供长度 1–32、33–128、129–512、513–1024 tokens 的 10%／30%／40%／20% 访问权重，作为待对照验证的工程起点。该权重约束样本访问次数，实际 token 曝光另行记录。长度区间仅参与候选选择与采样，正文保持自然边界。

每次参数更新的目标为：

\[
L=\lambda_{AE}\frac{\sum_i L_{AE,i}}{N_{AE}}+
\lambda_{LM}\frac{\sum_{i\in I_{LM}}L_{LM,i}}{N_{LM}}.
\]

其中单个样本的任务损失先按其目标 tokens 求均值，N_AE 与 N_LM 分别为整次参数更新内该任务的有效样本数；LM 项在存在有效续写时参与计算。microbatch 使用同一组分母累积梯度。AE-only 评估输出实际存在的 AE 条件，续写对照使用具有 Y 的样本。

## 3. 配置与输出

`configs/data_preparation/fineweb.json` 指定候选预算、来源配额、近重复参数与评分模型。抽查规模及 seed 由独立抽查命令指定。实验配置保存基座、自然片段长度预算、任务提示、长度采样及容量参数。默认准备配方使用规则质检；设置 fluency_model_name_or_path、review_model_name_or_path 可分别启用本地 NLL 和语义质检模型。

```bash
uv run python -m latent_working_memory.data_preparation \
  --config configs/v1/pretrain_a800.json \
  --recipe configs/data_preparation/fineweb.json \
  --dataset-dir data/raw/HuggingFaceFW-fineweb \
  --output-dir data/v1/fineweb-pretrain
```

启用本地模型评分时增加 `--score-cache data/v1/quality-cache/scores.jsonl`，GPU 运行增加 `CUDA_VISIBLE_DEVICES=0` 和 `--device cuda`。新增模型下载由用户确认。

| 产物 | 内容 |
|---|---|
| `candidates.jsonl` | 本轮固定候选原文及来源 |
| `document-decisions.jsonl` | 每个候选的划分、状态、理由与评分结果 |
| `documents.jsonl` | 入选文档原文、重复簇及排除区间 |
| `train.jsonl`、`dev.jsonl`、`test.jsonl` | 统一 episode 格式的训练与评估视图 |
| `audit.json` | 连续性、来源隔离、合法容量、长度分位数及采样预演 |
| `preparation.json` | 所有构造及审计完成后写入的训练消费记录 |

审计统计包含输入和续写的样本数、token 总数、均值、P50/P90/P95/P99。确定性预演模拟 1000 次采样访问，报告长度、粒度、容量、有效压缩率与独立文档覆盖。预演的容量课程 step 按 batch_size × gradient_accumulation_steps 换算；正文统计使用内容 tokens，训练累计目标数包含 EOS。

数据配额和自动契约检查完成后写入 preparation.json，训练与独立评估直接使用固定数据目录。质量抽查在数据构造完成后单独执行，输出到数据目录之外；随机视图估计该 split 的均匀样本缺陷率，分层视图定位长度及粒度问题。复核状态与汇总保存在独立抽查目录中，具体操作见 [评分与后置抽查方案](20260909_quality_scoring_and_inspection.md)。

本次数据与 sampler checkpoint 按新接口生成；历史实验使用各自冻结的代码和数据复现。上轮 ID 排除清单由助手抽查产生，记录保留在历史实验目录中。

## 4. 验证与后续实验

v1 的 61 项测试通过。独立抽查新增覆盖固定 seed 的可重复性、准备数据保持原样、复核存疑状态及缺陷比例汇总。新增覆盖 AE 文末保留、完整句末续写、独立 NLL 窗口、评分缓存复用与失效、质量输出契约、近重复分组、局部排除、准备失败的完成记录、长度采样恢复、混合目标的梯度归一化及 AE-only 评估。

完整链路测试通过本地 Parquet、tiny Llama NLL 评分、缓存、数据准备、审计、一次训练更新、评估和 checkpoint 保存；该测试屏蔽网络连接，验证工程行为。语义判定协议使用固定响应及原文区间测试，其实际筛选质量由选定强模型后的独立复核评估。

近重复聚类另用 80 组随机文档集合与逐对穷举 Jaccard 比较，簇划分一致；结果保存在 `artifacts/data-preparation-refactor-20260909/dedup-oracle.json`。

根目录全量测试为 63 项通过、2 项 C-DIC 目录断言失败：源文件清单检查旧的 checkpoint.py、icae_adapter.py 路径，产物检查将 `.venv` 中的 `_virtualenv.pth` 计入禁止文件。源文件路径失败在本次重构前已存在。全量与 v1 的 JUnit 记录分别保存在 `artifacts/data-preparation-refactor-20260909/tests.xml` 和 `v1-tests.xml`。

实验按变量分别比较：先固定语料与模型比较长度采样，再在相同长度分布和训练 token 预算下比较质量筛选。本次配置和测试为后续公开语料实验提供可复现入口。
