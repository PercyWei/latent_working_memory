# 20260910_预训练数据构建策略与实现（16:40:58 UTC+08:00）

创建时间：20260910 15:44:35 UTC+08:00
最后修订时间：20260910 16:40:58 UTC+08:00

## 1. 数据来源

使用 HuggingFaceFW/FineWeb 的 `sample-10BT` 子集。FineWeb 是从 Common Crawl 网页提取、清洗的英语文本语料，`sample-10BT` 为约 100 亿 tokens 的抽样子集。原始 Parquet 已保存在服务器项目目录下的 `data/raw/HuggingFaceFW-fineweb/sample-10BT/`。

每条记录对应一篇网页文档，包含 `text`、文档 ID、URL、抓取日期及来源信息。构建过程保留原文和字符位置，AE 与 LM 的目标均直接截取原文。当前候选池预算为 100,000 篇文档；按固定 seed 混合读取文件、row group 和批内记录。

## 2. 数据类型与构建规则

### 数据类型

令 X 为写入记忆的文本，Y 为 X 后紧邻的原文。任务类型与边界类型交叉组合为四类数据：

| 类型 | 输入与目标 | 边界规则 |
|---|---|---|
| semantic AE | 写入 X，重建 X | X 从完整句界开始并在完整句界结束 |
| semantic LM | 写入 X，续写 Y | X 的起点、X/Y 切点及 Y 的终点均为完整句界 |
| random AE | 写入 X，重建 X | 从原文 token 位置随机截取 X，可在词或句子内部截断 |
| random LM | 写入 X，续写 Y | 随机选择 X 起点及 X/Y 长度，允许随机边界截断 |

AE 与 LM 独立抽样，每个 episode 只有一次写入和一个读取目标；代码中的 LM 任务名为 `continuation`。semantic 与 random 共用来源划分和任务内长度配额，但具体文本独立抽取，不要求逐条对应或长度完全相同。相同来源可提供多个片段，各类样本也可能重叠。

注：

- 共用来源划分：某篇原文属于 train，它在 semantic 和 random 中生成的所有样本都属于 train。
- 共用任务内长度配额：以 AE 为例，LM 同理。假如在 semantic AE 训练集中，32–64 tokens 的样本要求有 14,000 条，random AE 对应区间也要求有 14,000 条。

### 长度与配额

所有长度按训练基座 tokenizer 对最终文本切片独立分词计算，不含提示与特殊 token。本轮使用 Llama-2-7B-Chat tokenizer。

- AE 的 X、LM 的 X 与 Y 均为 **32–4096 tokens**；LM 满足 **0.3 ≤ |X| / (|X| + |Y|) ≤ 0.7**，X 与 Y 对应相邻字符区间。
- 按 X 长度划分七档：32–64、65–128、129–256、257–512、513–1024、1025–2048、2049–4096。
- 每个 split 的四类数据分别均衡分配七档配额。总数不能被七整除时，余数依次分配到前几档，各档最多相差一条；仅通过模型判定及去重的入选样本计入配额。

### 文本粒度

文本粒度描述输入 X 对应的原文结构。

| 粒度 | 含义及 semantic 构造方式 |
|---|---|
| 单句 `sentence` | 取分句器识别的一条完整句子 |
| 段落 `paragraph` | 取同一原文段落内识别出的完整句子范围 |
| 连续句子集 `sentence_group` | 按长度档选取原文中连续的多句话，可跨段落，不要求主题相同 |
| 主题区间 `topic_group` | 按外部主题标注选取连续句子范围；仅在提供 `--topic-annotations` 时使用 |

semantic 的标签记录实际采用的候选构造方式。同一原文范围可能同时满足多种结构，例如单句段落；最终样本保留被选中候选的标签，因此这些名称不表示互斥的语言学类别。

random 的粒度固定为 `random`。

semantic 的粒度不设独立配额，最终占比由候选分布、长度配额和质量筛选共同决定。`audit.json` 在各 split、任务和长度档下统计各粒度的样本数、样本占比、输入 token 数及其占比；random 在相同统计结构中只有 `random` 一类。独立抽查对 semantic 按任务、粒度和长度分层，对 random 按任务和长度分层。

## 3. 构建流程

### 主流程

**固定来源池 → 属性检查与去重划分 → semantic 构造与评分 → random 构造与评分 → 自动审计。**

1. **来源检查与划分。** 检查 ID、URL 和文本字段，剔除去首尾空白后不足 64 字符的文档。按 ID、规范化 URL、全文和近重复关系聚类；近重复采用词 5-gram Jaccard，相似度阈值 0.9，至少 64 词的文档参与。每簇保留一个通过基础检查的代表，按簇和 seed 固定 train/dev/test。全部派生片段继承来源 split。
2. **选择任务与长度档。** 按剩余配额加权选择任务及 X 长度档，再使用 AE、LM 各自的随机序列抽取文本。每个版本、每篇文档最多尝试 64 次候选构造。
3. **构造原文片段。** semantic 从句子、段落、连续句子集提供的候选区间抽取 X；LM 再从 X 后选择满足长度和比例的 Y。random 独立抽取 token 起点及长度，映射为字符跨度后重新分词检查。未提供 `--topic-annotations` 时不使用主题标注。
4. **去重与模型判定。** 同一版本内按任务和规范化 X/Y 去重，跨 split 排除相同长输入；random 同时对照已完成 semantic 的输入登记。Qwen/Qwen3.8-27B 判断最终 AE 的 X 或 LM 的 X/Y，仅 `keep` 入选，`reject`、`uncertain` 和 `error` 不计入配额。已评分候选的理由及处理状态写入决策日志。
5. **补足配额并审计。** 遍历来源，补足任务及长度档缺额。来源池耗尽仍不足时，报告具体 split、任务和长度档。自动检查原文跨度、token 长度、比例、任务提示、来源隔离及入选判定，输出统计；检查通过后写入 `preparation.json`。random 还须通过两版本来源契约及任务内长度分布核对，再写入 `comparison.json` 和完成记录。

模型质量标准为内容可读、连贯且有实际信息，排除乱码、错误拼接及占主导的导航或广告模板；接受技术文字、叙事、口语、自然指代和话题变化。semantic 额外检查真实句法边界；random 的预期边界残缺不视为质量缺陷。Y 可以包含 X 中没有的新信息。

评分通过独立 HTTP 服务直连调用，temperature 为 0，关闭 thinking，返回 `decision` 和不超过 240 字符的 `reason`。缓存绑定完整 X/Y、模型、地址和评分协议，逐条保存。输出截断或判定格式错误时重试一次，仍失败则记为 `error` 并跳过候选；连接或服务错误直接中止阶段。完成目录保留。运行中的阶段按固定来源窗口提交进度，恢复使用已提交位置、已有样本与评分缓存。

### 独立抽查

在某个版本完成后，独立执行 **sample → judge → summarize**，结果写到数据目录之外。

- `sample` 生成两个面板：随机面板无放回均匀抽样，用于估计成品样本质量；分层面板对 semantic 覆盖任务、粒度和长度档，对 random 覆盖任务和长度档，每篇文档最多一个样本，用于定位问题。样本包含实际 X/Y 和来源位置。
- `judge` 使用独立抽查提示词与缓存协议重新判定，不将“已入选”作为质量依据。结果映射为 `pass/fail/uncertain`，逐批保存；接口返回 `error` 的样本保留为未复核。人工也可填写 `judgment`、`review_reason` 和 `reviewer`。
- `summarize` 分面板及分层汇总通过、失败、存疑和未复核数量。存在未决样本时报告缺陷比例上下界；分层面板不用于直接估计总体缺陷率。同模型复核可能重复筛选偏差，应结合人工复核分析。

抽查发现系统性问题后，修改配方并生成下一份数据，保留原数据及其复核记录。

## 4. 代码实现

代码位于 [`src/latent_working_memory/data_preparation/`](../../src/latent_working_memory/data_preparation/)。

| 模块 | 职责 |
|---|---|
| `config.py`、`__main__.py` | 配方、长度及配额契约；`sources/semantic/random/all` 阶段入口 |
| `sources.py`、`quality.py`、`dedup.py` | Parquet 混合读取、基本属性检查、来源去重聚类 |
| `segmentation.py`、`fineweb.py`、`truncation.py` | 原文句界与字符位置；semantic 和 random 的独立任务构造 |
| `scoring.py` | 样本评分提示、结构化响应、并发请求与缓存 |
| `pipeline.py`、`audit.py` | 来源登记、入选配额、样本去重、完成审计与跨版本比较 |
| `inspection.py` | 独立抽样、模型／人工复核与统计汇总 |

准备配方控制来源预算、长度、数量和评分服务；实验配置提供 tokenizer、来源划分、seed 与任务提示。训练通过 `v1/data.py` 读取单任务 episode。相关测试位于 `tests/v1/test_preparation*.py`、`test_truncation.py` 和 `test_sample_scoring.py`。

输出布局：

```text
fineweb-independent-409k-20260910/
  source-pool.json          来源池配置与统计
  sources.jsonl             候选原文、重复簇、split 和基础判定
  semantic/ random/
    train.jsonl  dev.jsonl  test.jsonl
    documents.jsonl         入选样本的来源原文
    sample-decisions.jsonl  已评分候选的判定及配额处理
    audit.json              长度、任务、粒度与来源统计
    preparation.json        通过审计后的完成记录
  comparison.json           两版本来源与配额分布核对
```

样本保存文档标识、重复簇、原文字符跨度、边界类型、粒度与评分缓存键；完成记录关联来源池和数据准备标识。`audit.json` 同时报告样本数和输入 token 数的组成比例。

## 5. 本轮运行与结果

### 运行配置与命令

本轮目标为 **408,800 条**入选样本，候选来源预算为 100,000 篇文档。以下是计划配额，实际产量待构建完成后填写。

| 每种组合（semantic/random × AE/LM） | train | dev | test |
|---|---:|---:|---:|
| 总样本配额 | 98,000 | 2,100 | 2,100 |
| 每个 X 长度档 | 14,000 | 300 | 300 |
| 四种组合合计 | 392,000 | 8,400 | 8,400 |

运行根目录为服务器 `/data/bywei/projects/latent_working_memory/artifacts/v1/data-preparation/independent-409k-20260910`，数据目录为 `/data/bywei/projects/latent_working_memory/data/v1/fineweb-independent-409k-20260910`。本地同步记录位于 [`artifacts/v1/data-preparation/independent-409k-20260910/`](../../artifacts/v1/data-preparation/independent-409k-20260910/)。本轮使用该目录的 `config.json`、`recipe.json`，不是仓库默认的 100,000／2,000／2,000 单任务配额。

Qwen 评分服务使用物理 GPU 1、BF16、单卡，窗口 16,384 tokens；客户端并发 16、最大输出 1,024 tokens、超时 600 秒。服务入口为运行目录的 `serve.sh`，使用 xgrammar 紧凑 JSON 输出，禁止字段间生成多余空白。数据构造进程通过 HTTP 请求评分，不加载 GPU 模型。

通用完整构造命令如下，需使用尚不存在的数据输出目录；分阶段运行时依次指定 `--stage sources`、`semantic`、`random`：

```bash
cd /data/bywei/projects/latent_working_memory
run_dir=/data/bywei/projects/latent_working_memory/artifacts/v1/data-preparation/independent-409k-20260910
uv run python -m latent_working_memory.data_preparation \
  --config "$run_dir/config.json" \
  --recipe "$run_dir/recipe.json" \
  --dataset-dir data/raw/HuggingFaceFW-fineweb \
  --output-dir data/v1/fineweb-independent-409k-20260910 \
  --score-cache "$run_dir/selection-bounded-cache.jsonl"
```

本轮恢复运行使用 `run.json` 记录的 `code-review-errors/` 快照和 `construct_review_errors.py`，依次执行 semantic、random；来源池已建立。实际解释器、环境与阶段命令保存在该脚本中，最终代码记录在构建结束后汇总。

成品抽查命令示例（semantic train，每个面板最多 200 条；其余版本与 split 分别执行）：

```bash
uv run python -m latent_working_memory.data_preparation.inspection sample \
  --data-dir data/v1/fineweb-independent-409k-20260910/semantic \
  --output-dir "$run_dir/inspection/semantic/train" \
  --split train --examples 200 --seed 20260909
uv run python -m latent_working_memory.data_preparation.inspection judge \
  --inspection-dir "$run_dir/inspection/semantic/train" \
  --recipe "$run_dir/recipe.json" \
  --score-cache "$run_dir/inspection-cache.jsonl"
```

`judge` 自动生成 `summary.json`；人工修改后使用同一模块的 `summarize --inspection-dir ...` 重新汇总。

### 结果状态

本地 `status.json` 在 20260910 13:31:49 UTC+08:00 记录 semantic 阶段运行中，尚无可确认全量完成的本地记录。完成后补充：最终代码与配置、起止时间和耗时、各 split/task 的入选文档数与样本数、X/Y token 总量及长度分布、模型判定与错误数量、自动审计结果，以及独立抽查统计和主要缺陷。

历史数据与本轮分开保存：`data/v1/fineweb-paired-20260909/` 为旧逐样本等长配对方案，每个版本 train/dev/test 为 226,022／11,610／11,685 条，本地报告位于 `artifacts/v1/data-preparation/fineweb-paired-20260909/`。旧独立方案的 semantic/random 各 100 条复核记录位于 `artifacts/v1/data-preparation/independent-100-20260909/`，不作为本轮 4k 数据的质量统计。

## 6. 执行调度与恢复

候选窗口由 `candidate_window_documents` 控制，默认包含四篇来源文档；评分并发数由 `scoring_batch_size` 单独控制。每个窗口按当前剩余配额抽取候选，只选择该文档能够提供的任务与长度档。每篇文档仍最多尝试 `candidates_per_document` 次，质量提示词、长度约束和最终配额保持一致。窗口改变了旧流程逐小批更新抽样权重的时点，后续候选序列按新窗口规则生成。

CPU 工作线程提前处理下一个窗口的分句、批量分词及合法续写范围，主线程提交当前窗口的评分请求。评分端保持有界并发，请求完成后立即补入下一个，并逐条持久化评分缓存。窗口内按候选原始顺序接收判定、去重和扣减配额，HTTP 返回先后顺序只影响评分缓存中的行顺序。窗口大小固定时，客户端并发数不会改变候选及接收顺序；模型推理自身的数值差异仍可能影响边缘判定。

每个窗口完成后，先将样本、来源和判定文件刷入磁盘，再原子更新阶段目录中的 `progress.json`，记录下一来源位置、文件提交偏移、计数及运行契约。恢复命令在原阶段参数上添加 `--resume`；程序检查契约，回退文件中超出最后提交偏移的尾部，从保存位置继续，未提交窗口的评分结果由缓存复用。允许调整评分并发与请求超时，数据配方、窗口大小和评分协议保持一致。

旧版暂停目录通过 `--adopt-paused` 显式导入：从已有样本和判定恢复配额与去重状态，并从最后出现判定的文档之后继续。已有记录原样保留，已判定 episode 和已入选文本用于避免重复；之后使用正常 `--resume`。旧文件无法完整恢复文档内部随机游标，因此导入时跳过最后一篇文档的剩余尝试，保持每篇文档的候选尝试上限；总配额继续由后续文档补足。

对应实现为 `pipeline.py` 的窗口预取与按序接收、`scoring.py` 的有界评分、`fineweb.py` 的批量候选分词及续写缓存，以及 `resume.py` 的进度提交与恢复。

本轮单卡评分基准选择并发 32，结果与限制见 [执行效率优化记录](../experiment_results/20260910_data_preparation_efficiency.md)。
