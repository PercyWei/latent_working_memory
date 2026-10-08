# 20261008_FineWeb 多段文本构造

创建时间：20261008 10:41:43 UTC+08:00

最后修订时间：20261008 22:56:37 UTC+08:00

从 FineWeb 原始 Parquet 构造分段正文及续文候选。与 FactQA 共用[分段逻辑](../segmentation.py)，保存固定字符区间；构造不加载 tokenizer，K 与倍率指定余量前的名义长度。

## 配置

以 [K512 配置](../../../../configs/data_preparation/fineweb-multisegment/fineweb-multisegment-k512-seg1to3x_train32k.json) 为例：

| 参数 | 示例值 | 含义 |
|---|---:|---|
| `source_dir` | `data/raw/HuggingFaceFW-fineweb/sample-10BT` | FineWeb 原始 Parquet 目录 |
| `source_batch_size` | 100000 | 每批原文读取量，配额不足继续读 |
| `source_seed` | 20260907 | 来源顺序与去重簇划分的随机种子 |
| `selection_seed` | 20260916 | 文档内段数与段长的随机种子 |
| `split_counts` | 32000 / 128 / 128 | train / dev / test 成品精确配额，也用于确定来源划分比例 |
| `window.capacity` | 512 | 段长基准 K |
| `window.min_segment_ratio` | 1 | 名义段长的最小倍率 |
| `window.max_segment_ratio` | 3 | 名义段长的最大倍率 |
| `window.min_segments` | 3 | 每条轨迹最少段数 |
| `window.max_segments` | 5 | 每条轨迹最多段数 |
| `window.continuation_tokens` | 512 | 续文候选的名义 token 数 Q；仅用于估算保存字符数 |
| `window.content_reserve_ratio` | 1.5 | 每段及续文各自的余量系数 α |

`window` 集中保存分段参数，`source_seed` 决定来源顺序与划分，`selection_seed` 决定文档内采样。只设置 `split_counts`，不另设划分比例。

各段名义长度在 K × 倍率范围内取整数；正文每段按 `ceil(4 × lᵢ × α)`、续文按 `ceil(4 × Q × α)` 换算为字符数，分别取整后求和。

候选续文的实际 token 数由训练 tokenizer 决定，不保证达到 Q。v3 的 LM 目标长度由训练配置 `training.lm_target_tokens` 独立指定，默认 512。

## 执行与命名

在项目根目录执行，配置中的相对路径也以此为基准：

```bash
uv run --frozen python -m latent_working_memory.data_preparation.fineweb_multisegment \
  --config configs/data_preparation/fineweb-multisegment/fineweb-multisegment-k512-seg1to3x_train32k.json \
  --output-root data --run-id 20261008
```

产物目录为 `data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/`。前缀由实际配置生成，`--run-id` 默认使用上海时区当天日期；它只用于命名，不改变采样，同名目录拒绝覆盖。

再次构造时可复用配置，更换 `--run-id`，并通过 `--previous-datasets DIR [DIR ...]` 排除已有数据。该参数默认空，优先读取所列目录的 `used-sources.jsonl`，并兼容旧发布数据内嵌的 `preparation.json.used_sources`；需排除的目录必须全部显式列出，不递归继承历史排除记录。

## 构造流程

1. **读取与过滤**：按 `source_seed` 无放回分批读取，保留能容纳最低分段要求及续文余量的文档，重复 ID 不再入池。
2. **去重与选源**：对累计候选按 ID、规范化 URL、正文及词 5-gram Jaccard 相似度 ≥ 0.9 的近重复关系分组，排除与已有数据匹配的整个簇。每簇最多选一篇，按簇划分 train/dev/test；配额按轨迹数计，全部达标后固定来源，来源耗尽则报告缺额。
3. **采样分段**：从每篇代表原文开头依次构造多条轨迹。按剩余字符预算扣除续文窗口，抽取段数及各段名义长度，并为后续段保留最低预算。正文长度为各段 `ceil(4 × lᵢ × α)` 之和；随机流由 `selection_seed` 和文档 ID 决定。
4. **截取窗口**：正文保存为 `text`，紧邻尾部保存为 `continuation`；下一条从本条续文末尾开始，完整窗口（含余量）互不重叠，剩余原文不足最小窗口时停止。同篇全部轨迹属于同一 split，最后一篇只取填满配额所需的窗口。FactQA 共用此规则，取 Q=0 并在完整段上生成 QA。

## 输出

目录包含 `train.jsonl`、`dev.jsonl`、`test.jsonl`、`used-sources.jsonl`、`preparation.json` 和按实际统计生成的 `README.md`。三个划分文件中每行是一条轨迹：

| 字段 | 含义 |
|---|---|
| `trajectory_id` / `document_id` / `dedup_cluster` / `split` | 轨迹、文档、去重簇标识及划分 |
| `source` / `window_char_span` | 原文 `file`、`row_group`、`row_index`；正文在原文中的字符区间 |
| `text` / `segments` | 分段正文；每段含 `segment_id` 和相对正文的 `char_span` |
| `text_char_length` / `estimated_tokens` / `estimated_tokens_rule` | 正文字符数、估算 token 数及规则 `len(text) / 4` |
| `continuation` | 紧接正文的连续尾部，仅含续文目标及其自身余量 |

以上共有字段与 FactQA 同名同结构。字符区间左闭右开，按 Python 字符串索引计数；各段连续覆盖 `text`。`estimated_tokens` 按含余量的实际正文字符数计算，与余量前的名义长度不同。读取时先切段再分别分词，实际 token 数与 tokenizer 有关；续文候选由使用端按训练目标长度处理。

`used-sources.jsonl` 按文档 ID 去重排序，逐条保存 `{document_id, dedup_cluster, source}`，每篇实际使用原文仅登记一次，后续构造按整篇原文排除。`preparation.json` 保存构造配置、`run_id`、实际创建时间、显式排除目录、排除来源数及独立原文数、轨迹数统计，通过 `used_sources_file` 指向该文件，不重复保存来源大列表。
