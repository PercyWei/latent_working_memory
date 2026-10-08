# 20260929_FineWeb 事实 QA 构造

创建时间：20260929 16:03:54 UTC+08:00

最后修订时间：20261008 17:39:11 UTC+08:00

从 FineWeb 原始 Parquet 构造多段正文、事实问答及逐段使用安排。与多段文本构造共用[分段逻辑](../segmentation.py)，先冻结字符区间，再生成、核验和补齐 QA；构造不加载 tokenizer。

## 配置

以 [k512 配置](../../../../configs/data_preparation/fineweb-factqa/fineweb-factqa-k512-seg1.5to2x_train1000.json) 为例：

| 参数 | 示例值 | 含义 |
|---|---:|---|
| `source_dir` | `data/raw/HuggingFaceFW-fineweb/sample-10BT` | FineWeb 原始 Parquet 目录 |
| `source_batch_size` | 100000 | 每批原文读取量，候选不足时继续读取 |
| `source_seed` | 20260907 | 来源顺序与去重簇划分的随机种子 |
| `selection_seed` | 20260928 | 候选排序、文档分段及窗口起点的随机种子 |
| `split_counts` | 1000 / 100 / 100 | train / dev / test 成品精确配额，也用于确定来源划分比例 |
| `window.capacity` | 512 | 段长基准 K |
| `window.min_segment_ratio` | 1.5 | 名义段长的最小倍率 |
| `window.max_segment_ratio` | 2 | 名义段长的最大倍率 |
| `window.min_segments` | 6 | 每条轨迹最少段数 |
| `window.max_segments` | 10 | 每条轨迹最多段数 |
| `window.content_reserve_ratio` | 1.5 | 各段独立的余量系数 α |
| `batch_split_counts` | 150 / 25 / 25 | 每批各划分最多分配的来源数，实际不超过剩余成品缺额 |
| `qa.max_answer_chars` | 128 | 可调正整数上限，生成、过滤、定稿及加载统一使用 |
| `qa.role_seed` | 20260928 | 角色分配与题目安排的随机种子 |
| `qa.max_supplement_rounds` | 3 | 首轮后最多补题轮数 |
| `qa.supplement_surplus` | 2 | 缺额段补题时额外生成的候选数 |
| `annotation` | `gpt-6-sol` / `medium` / 并发 4 | 模型服务地址、模型、推理强度、并发及请求预算 |
| `prompts_dir` | `configs/data_preparation/prompts/fineweb_factqa` | 包含 `generate.txt`、`verify.txt`、`document_review.txt` 的目录 |

名义段长取 `ceil(K × 最小倍率)` 至 `floor(K × 最大倍率)` 的整数，各段独立保存 `ceil(4 × lᵢ × α)` 个字符；示例为名义 768–1024、实际 4,608–6,144 字符。QA 固定不附续文，使用含余量的完整段生成问答，不裁回名义长度；分段不要求句子或词边界。

## 执行与命名

在项目根目录执行，配置中的相对路径也以此为基准；先确认 `annotation` 中的模型服务可用：

```bash
uv run --frozen python -m latent_working_memory.data_preparation.fineweb_factqa.campaign run \
  --config configs/data_preparation/fineweb-factqa/fineweb-factqa-k512-seg1.5to2x_train1000.json \
  --run-id 01-20261008
```

产物名为 `fineweb-factqa-k512-seg1.5to2x_train1000_01-20261008`，前缀由实际 K、倍率和 train 配额生成。`--output-root` 默认 `data`，`--artifacts-root` 默认 `artifacts/fineweb-factqa`，分别保存数据与运行记录。`--run-id` 默认执行时的上海日期，仅控制命名；恢复、查看进度和汇总须使用原 run-id、参数及输出根目录。

将命令中的 `run` 替换为 `report` 可只读查看进度，替换为 `finalize` 可重新汇总。campaign 自动保存主配置快照并派生批次配置、来源分配及共享请求缓存。单批阶段使用 `uv run --frozen python -m latent_working_memory.data_preparation.fineweb_factqa <prepare|annotate|finalize> --config <运行目录>/batch-configs/batch-NNN.json`。

再次构造时可复用配置、更换 `--run-id`，通过 `--previous-datasets DIR [DIR ...]` 显式列出全部待排除目录；默认空，不递归继承历史排除。FactQA 与 multisegment 共用 `used-sources.jsonl` 来源账本，可相互排除；已发布数据原有的 `preparation.json.used_sources` 也可读取。

## 构造流程

1. **读取与选源**：按 `source_batch_size` 无放回分批读取，过滤不合格及过短文档，按 ID、规范化 URL、正文和近重复关系去重。候选不足则扩池；新增候选与本轮旧池及显式指定的历史已用来源去重，既有来源、段界和划分保持不变。来源簇按 `split_counts` 归一化比例划分。
2. **生成与定位**：在每个完整段上生成问题、短答案、事实陈述及证据引文。程序检查字段、数量、答案长度与引文位置；证据须在所属段内唯一且连续，答案须位于证据内。
3. **两级核验**：`verify` 核验各段全部程序通过的候选；`document_review` 结合完整正文及累计局部通过候选，检查后文修正、歧义和事实重复。同一事实的改写归入同组，优先保留证据最早完整出现处的代表。
4. **补题与定稿**：重算逐段缺额，以“缺额＋余量”补题并重复两级核验，首轮后最多三轮。完整达标轨迹才进入成品；不足配额或文档级标注失败保留原因。定稿再次检查事实隔离、角色配额、证据区间与使用安排。
5. **补足与汇总**：每批各划分最多分配 `min(batch_split_counts, 剩余成品缺额)` 篇新来源。已完成的划分停止分配，其余继续构造，最终精确达到 1000 / 100 / 100 条；来源耗尽仍有缺额则报错。合并前检查来源、划分和重复，写出整轮元数据。

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

`preparation.json` 保存实际配置、渲染后的 prompts、run-id、排除记录、构造统计和失败原因，以 `used_sources_file` 引用来源账本。账本每行保存 `document_id`、`dedup_cluster`、`source`，登记本轮实际分配的所有文档，**包括标注失败的来源**，不含仅扫描而未分配的候选；下一轮按整篇原文排除。
