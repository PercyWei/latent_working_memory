# 20261008_FineWeb 多段文本构造

创建时间：20261008 10:41:43 UTC+08:00

最后修订时间：20261008 11:53:27 UTC+08:00

从 FineWeb 原始 Parquet 构造连续文本窗口，保存完整候选文本、分段计划及来源位置。构造不加载模型或 tokenizer，文中的 token 长度均为计划值。

## 配置

配置位于 `configs/data_preparation/fineweb-multisegment/`：

| 配置 | 每段长度 | 正文长度范围 |
|---|---:|---:|
| [K64](../../../../configs/data_preparation/fineweb-multisegment/fineweb-multisegment-k64-seg1to3x_train32k.json) | 64–192 tokens | 192–960 tokens |
| [K512](../../../../configs/data_preparation/fineweb-multisegment/fineweb-multisegment-k512-seg1to3x_train32k.json) | 512–1536 tokens | 1536–7680 tokens |

- **分段**：以 `capacity` 为 K，每段范围为 `ceil(K × min_segment_ratio)` 至 `floor(K × max_segment_ratio)`；两份配置均为 1–3 倍 K、每条 3–5 段。
- **规模**：`counts` 为 train 32000、dev 128、test 128；`source_batch_size=100000` 控制每批读取量，配额不足继续读取。
- **窗口余量**：`continuation_tokens=512`，正文与续文共用 `content_reserve_ratio=1.5`。

## 执行与命名

在项目根目录执行，配置中的相对路径也以此为基准：

```bash
uv run --frozen python -m latent_working_memory.data_preparation.fineweb_multisegment \
  --config configs/data_preparation/fineweb-multisegment/fineweb-multisegment-k512-seg1to3x_train32k.json \
  --output-root data --run-id 01-20261008
```

产物目录为 `data/fineweb-multisegment-k512-seg1to3x_train32k_01-20261008/`。前缀由实际配置生成，`--run-id` 默认使用上海时区当天日期；它只用于命名，不改变采样，同名目录拒绝覆盖。

再次构造时可复用配置，更换 `--run-id`，并通过 `--previous-datasets DIR [DIR ...]` 排除已有数据。该参数默认空，读取所列目录的 `preparation.json.used_sources`；需排除的目录必须全部显式列出，不递归继承历史排除记录。

## 构造流程

1. **读取与过滤**：按 `source_seed` 无放回分批读取，保留能容纳最低分段要求及续文余量的文档，重复 ID 不再入池。
2. **去重与选源**：对累计候选按 ID、规范化 URL、正文及词 5-gram Jaccard 相似度 ≥ 0.9 的近重复关系分组，排除与已有数据匹配的整个簇。每簇最多选一篇，按簇划分 train/dev/test；全部配额满足后固定来源，来源耗尽则报告缺额。
3. **采样分段**：每篇选中文档生成一条轨迹。在原文可容纳的范围内抽取段数与各段长度，再求和得到正文长度 `L`，不预设总长。分段随机流由 `seed` 和文档 ID 决定。
4. **截取窗口**：令 `Q = continuation_tokens`、`α = content_reserve_ratio`，按 `ceil(4 × (L + Q) × α)` 计算窗口字符数，随机选择原文起点并保存连续全文及累计写入切点。

## 输出

目录包含 `train.jsonl`、`dev.jsonl`、`test.jsonl`、`preparation.json` 和按实际统计生成的 `README.md`。每行 JSONL 是一条轨迹：

| 字段 | 含义 |
|---|---|
| `sample_id` / `document_id` / `dedup_cluster` | 样本、来源文档和去重簇标识 |
| `text` | 完整候选窗口，含正文、续文及字符余量 |
| `write_token_ends` | 分段计划的累计 token 终点 |
| `source` | 原始 `file`、`row_group`、`row_index` 及相对原文的左闭右开字符区间 `char_span` |

`preparation.json` 保存构造配置、`run_id`、实际创建时间、排除记录和统计；`used_sources` 仅登记本次实际生成轨迹的文档，供后续构造排除来源。
