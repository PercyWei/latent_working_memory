# 20260929_FineWeb 事实 QA 构造

创建时间：20260929 16:03:54 UTC+08:00

最后修订时间：20261008 23:20:24 UTC+08:00

从 FineWeb 原始 Parquet 构造多段正文、事实问答及逐段使用安排。与多段文本构造共用[分段逻辑](../segmentation.py)，先冻结字符区间，再生成、核验和补齐 QA；构造不加载 tokenizer。

## 配置

以 [K512 配置](../../../../configs/data_preparation/fineweb-factqa/fineweb-factqa-k512-seg1to3x_train1000.json) 为例：

| 参数 | 示例值 | 含义 |
|---|---:|---|
| `source_dir` | `data/raw/HuggingFaceFW-fineweb/sample-10BT` | FineWeb 原始 Parquet 目录 |
| `source_batch_size` | 100000 | 每批原文读取量，候选不足时继续读取 |
| `source_seed` | 20260907 | 来源顺序与去重簇划分的随机种子 |
| `selection_seed` | 20260928 | 来源排序、文档内段数与段长的随机种子 |
| `split_counts` | 1000 / 100 / 100 | train / dev / test 成品精确配额，也用于确定来源划分比例 |
| `window.capacity` | 512 | 段长基准 K |
| `window.min_segment_ratio` | 1 | 名义段长的最小倍率 |
| `window.max_segment_ratio` | 3 | 名义段长的最大倍率 |
| `window.min_segments` | 6 | 每条轨迹最少段数 |
| `window.max_segments` | 10 | 每条轨迹最多段数 |
| `window.content_reserve_ratio` | 1.5 | 各段独立的余量系数 α |
| `batch_split_counts` | 150 / 25 / 25 | 每批各划分最多分配的候选轨迹数，不超过剩余成品缺额 |
| `qa.max_answer_chars` | 128 | 可调正整数上限，生成、过滤、定稿及加载统一使用 |
| `qa.role_seed` | 20260928 | 角色分配与题目安排的随机种子 |
| `qa.max_supplement_rounds` | 3 | 首轮后最多补题轮数 |
| `qa.supplement_surplus` | 2 | 缺额段补题时额外生成的候选数 |
| `annotation` | `gpt-6-sol` / `medium` / 并发 4 | 模型服务地址、模型、推理强度、并发及请求预算 |
| `prompts_dir` | `configs/data_preparation/prompts/fineweb_factqa` | 包含 `generate.txt`、`verify.txt`、`document_review.txt` 的目录 |

名义段长取 `ceil(K × 最小倍率)` 至 `floor(K × 最大倍率)` 的整数，各段独立保存 `ceil(4 × lᵢ × α)` 个字符；示例为名义 512–1536、实际 3,072–9,216 字符。QA 固定不附续文，使用含余量的完整段生成问答，不裁回名义长度；分段不要求句子或词边界。

## 执行与命名

在项目根目录执行，配置中的相对路径也以此为基准；先确认 `annotation` 中的模型服务可用：

```bash
uv run --frozen python -m latent_working_memory.data_preparation.fineweb_factqa.campaign run \
  --config configs/data_preparation/fineweb-factqa/fineweb-factqa-k512-seg1to3x_train1000.json \
  --run-id 01-20261008
```

产物名为 `fineweb-factqa-k512-seg1to3x_train1000_01-20261008`，前缀由实际 K、倍率和 train 配额生成。`--output-root` 默认 `data`，`--artifacts-root` 默认 `artifacts/fineweb-factqa`，分别保存数据与运行记录。`--run-id` 默认执行时的上海日期，仅控制命名；恢复、查看进度和汇总须使用原 run-id、参数及输出根目录。

将命令中的 `run` 替换为 `report` 可只读查看进度，替换为 `finalize` 可重新汇总。campaign 自动保存主配置快照并派生批次配置、来源分配及共享请求缓存。单批阶段使用 `uv run --frozen python -m latent_working_memory.data_preparation.fineweb_factqa <prepare|annotate|finalize> --config <运行目录>/batch-configs/batch-NNN.json`。

再次构造时可复用配置、更换 `--run-id`，通过 `--previous-datasets DIR [DIR ...]` 显式列出全部待排除目录；默认空，不递归继承历史排除。FactQA 与 multisegment 共用 `used-sources.jsonl` 来源账本，可相互排除；已发布数据原有的 `preparation.json.used_sources` 也可读取。

## 构造流程

1. **读取与选源**：按 `source_batch_size` 无放回分批读取，过滤并按 ID、规范化 URL、正文和近重复关系去重。每簇选一篇代表，按 `split_counts` 归一化比例划分；新增来源与旧池及历史已用原文去重，既有窗口和划分保持不变。
2. **冻结窗口**：从每篇原文开头依次构造多条轨迹，按剩余字符预算抽取段数、段长并逐段计入余量。下一条从本条正文末尾开始，剩余不足最小窗口时停止；同篇全部窗口互不重叠且属于同一 split，以 `trajectory_id` 区分。
3. **生成与定位**：在每个完整段上生成问题、短答案、事实陈述及证据引文。程序检查字段、数量、答案长度与引文位置；证据须在所属段内唯一且连续，答案须位于证据内。
4. **两级核验**：`verify` 核验各段全部程序通过的候选；`document_review` 结合完整正文及累计局部通过候选，检查后文修正、歧义和事实重复。同一事实的改写归入同组，优先保留证据最早完整出现处的代表。
5. **补题与定稿**：重算逐段缺额，以“缺额＋余量”补题并重复两级核验，首轮后最多三轮。完整达标轨迹才进入成品；不足配额或文档级标注失败保留原因。定稿再次检查事实隔离、角色配额、证据区间与使用安排。
6. **补足与汇总**：每批各划分最多分配 `min(batch_split_counts, 剩余成品缺额)` 条候选轨迹。同篇窗口可分配到不同批次；已完成的划分停止分配，其余继续构造，最终精确达到 1000 / 100 / 100 条；来源耗尽仍有缺额则报错。合并前核对冻结窗口、轨迹唯一性、来源一致性和划分隔离，写出整轮元数据。

对于 N 段正文，QA 数量规则固定如下；任务题与 gate 题使用不同事实：

| 类型 | 首段 / 中间各段 / 末段 | 总数 |
|---|---|---:|
| 首轮候选上限 | 15 / 10 / 5 | 10N |
| 最终任务题 | 4 / 4 / 4 | 4N |
| 最终 gate 题 | 8 / 4 / 0 | 4N |

任务题在 train 中标为 `train`、在 dev/test 中标为 `evaluation`；gate 为独立题池。`usage` 固定每步的新题、旧题和 gate 题；dev/test 的任务题安排覆盖所有已读段。

## 输出

数据目录包含 `source-pool.json`、`used-sources.jsonl`、`train.jsonl`、`dev.jsonl`、`test.jsonl` 和 `preparation.json`。来源池保存索引及冻结段界，不复制正文；运行目录保存配置快照、候选、审查决定、失败记录和请求用量。

| JSONL 字段 | 含义 |
|---|---|
| `trajectory_id` / `document_id` / `dedup_cluster` / `split` | 轨迹、来源文档、去重簇标识及划分 |
| `source` / `window_char_span` | 原文 `file`、`row_group`、`row_index`；正文在原文中的字符区间 |
| `text` / `segments` | 完整正文及连续覆盖正文的 `segment_id`、`char_span` |
| `text_char_length` / `estimated_tokens` / `estimated_tokens_rule` | 正文字符数、估算 token 数及规则 `len(text) / 4`；估算包含段内余量 |
| `qas` / `usage` | 问答、事实组、角色、证据／答案区间及逐步使用安排 |

字符区间均左闭右开，按 Python 字符串索引计数。`window_char_span` 相对原文章；段界、证据和答案区间相对保存的正文 `text`。生成、核验及补题均沿用这些位置。

`preparation.json` 保存实际配置、prompts、run-id、排除记录和构造统计。`summary` 统计实际分配的构造任务；`source_statistics` 与 `source-pool.json.statistics` 统计整个冻结候选池。

| 统计字段 | 含义 |
|---|---|
| `frozen_trajectories` / `complete_trajectories` | 冻结候选／成品轨迹数 |
| `frozen_source_documents` / `complete_source_documents` | 对应的独立原文数，按 `document_id` 去重 |

已标注、失败及补题数量也以轨迹计数；同篇原文的多个窗口可跨批次，汇总原文数须重新去重，不能直接相加。

`used_sources_file` 指向来源账本，每行保存 `document_id`、`dedup_cluster`、`source`。账本按文档 ID 去重登记本轮实际分配的原文，**包括标注失败的来源**，不含仅扫描而未分配的候选；下一轮按整篇原文排除。

## 成品读取

现有 `fineweb-factqa-k512-seg1to3x_train1000_01-20261008` 已发布至 [ModelScope 数据仓库](https://modelscope.cn/datasets/percyWeeei/latent-working-memory/files)，采用上述文件结构，由历史数据迁移而来；保留原始正文、段界、QA、使用安排及划分，实际 train / dev / test 数量为 **1007 / 118 / 120**。构造参数和来源分配记录保留在元数据及来源池中。

训练读取 `preparation.json` 与三个 split 文件，按保存的字符区间分词，验证正文、QA、使用安排及来源隔离；构造快照用于追溯，无需原始 FineWeb Parquet。
