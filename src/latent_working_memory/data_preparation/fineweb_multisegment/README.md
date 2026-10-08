# 20261008_FineWeb 多段文本构造

创建时间：20261008 10:41:43 UTC+08:00

最后修订时间：20261008 11:40:33 UTC+08:00

构造自包含的 FineWeb 连续文本窗口与多段 token 写入计划，供单次压缩、逐段写入、历史重构和续文预测共用。构建时不加载模型或 tokenizer；训练直接读取保存正文。

## 配置与命名

正式配置集中在 `configs/data_preparation/fineweb-multisegment/`，共用同一构造入口，通过 `--config` 指定文件。

| 配置 | 每段长度 | 正文自然范围 | 续文目标 |
|---|---:|---:|---:|
| [K64](../../../../configs/data_preparation/fineweb-multisegment/fineweb-multisegment-k64-seg1to3x_train32k.json) | 64–192 tokens | 192–960 tokens | 512 tokens |
| [K512](../../../../configs/data_preparation/fineweb-multisegment/fineweb-multisegment-k512-seg1to3x_train32k.json) | 512–1536 tokens | 1536–7680 tokens | 512 tokens |

两份配置均为 3–5 段，配额为 train 32000、dev 128、test 128。下表默认值以 K512 为例：

| 配置项 | 默认值 | 含义 |
|---|---|---|
| `capacity` | 512 | 构建基准 K，用于确定每段计划长度 |
| `min_segment_ratio` / `max_segment_ratio` | 1 / 3 | 每段长度为 `ceil(K × min)` 至 `floor(K × max)`，支持小数倍率 |
| `min_segments` / `max_segments` | 3 / 5 | 在原文可容纳的段数范围内等概率抽取 |
| `counts` | train 32000、dev 128、test 128 | 各划分的候选轨迹配额 |
| `continuation_tokens` | 512 | 加载后实际需要的紧邻续文长度 |
| `content_reserve_ratio` | 1.5 | 正文与续文共同使用的字符余量系数 |
| `source_batch_size` | 100000 | 每轮读取的原始文档数；配额不足继续读取，不设扫描总量上限 |

配置只保存采样参数；本次产物身份和来源排除由命令行指定：

| 命令行参数 | 默认值 | 含义 |
|---|---|---|
| `--output-root` | `data` | 生成数据集目录的父目录 |
| `--run-id` | 执行时上海日期 `YYYYMMDD` | 产物标识，例如 `01-20261008`；首字符为英文字母或数字，其余可用英文字母、数字、`.`、`-`、`_` |
| `--previous-datasets DIR [DIR ...]` | 空列表 | 显式列出本次全部需排除的数据集，读取各自 `preparation.json.used_sources` |

目录名为“实际配置生成的前缀 + `_` + run_id”。默认日期后缀例如：

```text
fineweb-multisegment-k512-seg1to3x_train32k_20261008
```

`k512` 保留构建基准，`seg1to3x` 表示每段 1–3 × K，`train32k` 表示目标训练轨迹数，也就是实际采用的训练来源文档数。正文长度由各段求和，默认自然范围为 1536–7680 tokens，没有独立的总长参数。规模标签是简写，精确参数与实际统计保存在元数据；同名输出目录已存在时拒绝覆盖。`run_id` 只标记产物，不改变 seed 或采样；仅更换标识不会得到另一份独立数据。

`preparation.json` 顶层记录本次 `run_id`、`previous_datasets` 及真实 `created_at`，`config` 仅记录采样参数。自定义标识中的日期不会修改实际创建时间。

## 执行

从项目根目录复用同一份配置构造两份数据，以下以 K512 为例；构造 K64 时将 `--config` 换为上表 K64 文件，产物前缀自动变为 `fineweb-multisegment-k64-seg1to3x_train32k`：

```bash
# 第一份：未指定历史排除
uv run --frozen python -m latent_working_memory.data_preparation.fineweb_multisegment \
  --config configs/data_preparation/fineweb-multisegment/fineweb-multisegment-k512-seg1to3x_train32k.json \
  --output-root data --run-id 01-20261008

# 第二份：显式排除第一份实际使用的来源
uv run --frozen python -m latent_working_memory.data_preparation.fineweb_multisegment \
  --config configs/data_preparation/fineweb-multisegment/fineweb-multisegment-k512-seg1to3x_train32k.json \
  --output-root data --run-id 02-20261008 \
  --previous-datasets data/fineweb-multisegment-k512-seg1to3x_train32k_01-20261008
```

上例分别生成后缀为 `_01-20261008` 与 `_02-20261008` 的目录；第二份的来源差异来自显式排除，不是 `run_id`。省略 `--run-id` 时使用执行当天的上海日期。

只生成多段数据，根目录直接保存 `train.jsonl`、`dev.jsonl`、`test.jsonl`、`preparation.json` 和按实际统计生成的 `README.md`。当前代码迁移没有执行全量构造；现行训练配置中的日期目录需先生成，或改成实际构造目录。

## 构造规则

1. **分批无放回读取。** 固定 `source_seed` 的 Parquet 读取流持续前进，每个原始位置只读一次；跨批相同文档 ID 不重复入池。基础过滤后，仅保留足以容纳最少段数、每段最低长度及续文余量的文档。
2. **累计去重与来源排除。** 每批合格文档加入累计池，按 ID、规范化 URL、规范化正文及词 5-gram Jaccard 近重复关系重新聚类。恢复 `--previous-datasets` 所列数据实际用过的原文，排除匹配到的整个当前簇；簇代表名称变化仍可排除。
3. **填充来源配额。** 每个未排除簇至多选一篇，按最终簇身份划分 train/dev/test。来源比例默认为 90%／5%／5%，实际产量由 `counts` 控制。任何划分不足就读取下一批并重新选择，所有配额满足后才固定来源并写出；来源耗尽则报告缺额，不循环复用文档凑数。
4. **先分段，再求正文长度。** 每篇最终选中的文档只构造一次。根据原文可用预算抽取段数 N，再依次抽取各段长度，抽取时为剩余段保留最低长度。打乱段长顺序以免预算约束集中在末段，正文文字顺序保持不变；最后求和得到 `L = sum(各段长度)`。
5. **截取完整窗口。** 令 Q 为 `continuation_tokens`、α 为 `content_reserve_ratio`，窗口字符数为 `ceil(4 × (L + Q) × α)`，原文可用正文预算为 `floor(原文字符数 / (4 × α)) − Q`。从可行起点随机截取连续全文，保存累计 token 切点；正反向预算使用同一精确有理数规则。

每篇文档的分段随机流由 `seed` 和文档 ID 决定。`source_batch_size` 只控制每轮读取量，不是累计池的内存上限；补读时保留先前合格候选并重新聚类，可处理新文档连接旧簇的情况。批大小可能改变停止时的候选集合，从而改变最终选中来源。

`used_sources` 只登记实际生成轨迹的文档及原始位置，扫描但未采用的文档不进入名单。`--previous-datasets` 不递归继承旧数据集排除过的历史；本次需要排除的所有旧数据都须显式逐项列出。例如第三份若需与前两份隔离，应同时传入第一份和第二份目录。

`source.file` 与配置路径按项目根目录解析。构建及跨数据集近重复排除需要原始 Parquet，训练与评估不需要。作为排除依据的旧数据必须提供 `used_sources`；历史 reconstruction 元数据没有该字段，不能直接加入该列表。

## 样本与训练边界

| 字段 | 内容 |
|---|---|
| `sample_id` / `document_id` / `dedup_cluster` | 样本、来源文档和来源簇身份 |
| `text` | 连续候选窗口全文，包含正文余量与末尾续文缓冲 |
| `write_token_ends` | 目标 token 序列中的累计写入终点 |
| `source` | `file`、`row_group`、`row_index`、`char_span`；字符区间相对原文章、左闭右开 |

段界是 token 长度计划，换 tokenizer 后对应的文字边界可能变化。对整个候选窗口一次分词，AE、续文及所有写入步骤共享同一连续 token 序列；不按段分别分词，不重复保存历史前缀或 token IDs。

- v2：验证构建参数中的段数与每段倍率，保留完整多段轨迹；warm-up 在同一正文的最后切点做单次压缩。
- v3 `multisegment_full`：取最终前缀，按训练 `max_input_tokens` 右裁剪；`multisegment_first_write` 取首段并按同样规则裁剪。
- 续文从**实际使用的终点**紧邻取 `continuation_tokens` 个 tokens。首段使用只要求该前缀及续文可用，不因后续完整轨迹过短而淘汰。
- `min_input_tokens` 用于过滤过短输入；裁剪后的长度、裁剪数量与过滤原因随运行记录。裁剪后的训练片段不再要求满足原始构造的每段最低倍率。

使用 K64 数据时须同时调整下游的长度选择配置：全文计划为 192–960 tokens，首段为 64–192 tokens。当前 v3 的 K512 预训练配置分别要求全文至少 1024、首段至少 768 tokens，直接更换数据路径会将 K64 样本全部过滤；若需保留完整计划范围，相应最低长度可设为 192 和 64。

默认正文自然范围不是训练必须使用的长度。采用 4096-token 输入上限的 v3 实验可在加载时裁剪。字符余量仅是 tokenizer 无关的估计，实际分词后不足正文或续文长度的样本仍需过滤。
