# 20260909_独立 AE/LM 数据构造实现（20:13:30 UTC+08:00）

创建时间：20260909 20:13:30 UTC+08:00

最后修订时间：20260909 20:13:30 UTC+08:00

数据准备代码位于 `src/latent_working_memory/data_preparation/`。主流程为：固定来源池 → 简单属性检查与去重划分 → semantic 构造、模型判定与配额补足 → 统计入选长度分布 → random 独立构造、模型判定与配额补足 → 契约检查与完成记录。质量抽查具有独立入口，在完整数据生成后执行。

## 1. 样本数量与长度口径

每个版本、每个 split 的 AE 与 LM 分别达到相同的最终入选数量。配方中的 `samples_per_task` 按 train/dev/test 排列，示例默认值为 `[100000, 2000, 2000]`：

| 版本 | train AE / LM | dev AE / LM | test AE / LM | 总样本数 |
|---|---|---|---|---|
| semantic | 100000 / 100000 | 2000 / 2000 | 2000 / 2000 | 208000 |
| random | 100000 / 100000 | 2000 / 2000 | 2000 / 2000 | 208000 |

配额统计发生在模型判定和样本去重之后。数据文件中的任务标识为 `ae` 和 `continuation`，后者对应 LM。训练器从固定数据中选择任务，训练目标权重由训练配置管理。

`length_bounds=[64,128,256,512,1024]` 定义实际输入 X 的 token 区间：1–64、65–128、129–256、257–512、513–1024。semantic 完成后，分别统计每个 split、每项任务的区间计数；random 按对应计数补足。区间内的具体长度和样本来源独立选择。LM 目标长度单独记录，供后续比较监督 token 量。

这些区间是数据构造的分布口径；`ExperimentConfig.input_length_bounds` 和访问权重控制训练采样，两者分别保存。

## 2. 主流程

1. **读取固定候选池。** `sources.py` 从已有 `HuggingFaceFW-fineweb/sample-10BT/` Parquet 中按 seed 混合读取，候选上限由 `max_documents` 控制。保留原文、ID、URL、抓取日期、dump 和文件路径。默认候选预算为 18000 篇，其可达产量由真实筛选结果确定。
2. **检查显式属性，固定来源划分。** `quality.py` 校验来源字段，按 `min_document_chars` 排除过短原文。`dedup.py` 按 ID、规范化 URL、全文和近重复关系归并文档；近重复使用词 5-gram Jaccard，相似度默认 0.9。每簇保留一个通过基础检查的代表，以簇标识和 seed 确定 split。两套数据共同使用根目录的来源登记表，同一文档及近重复簇的全部派生样本继承相同 split。
3. **构造 semantic 候选。** `fineweb.py` 保存句子和段落的原文位置，生成单句、段落、连续句子组及相邻段落组合。各粒度覆盖不同自然长度；离线主题索引可以提供额外连续句组。AE 写入并重建完整片段 S；LM 从 S 的内部句界选择切点，写入前缀 X，预测剩余后缀 Y。LM 的全部 Y 与 EOS 参与监督。构造时按实际 tokenizer 校验输入、目标和合法 memory 容量。
4. **模型判定最终样本，补足 semantic 配额。** `SampleScorer` 将 AE 的 X 或 LM 的 X/Y 发给 Qwen3.8-27B。公共标准判断可读性、内容连贯性和网页噪声；semantic 提示词同时核查句界完整性。模型返回 `keep / reject / uncertain` 和理由，`keep` 样本进入剩余任务配额。相同任务与 X/Y 去重，跨 split 的相同长输入被排除。候选不合格时继续处理后续候选和来源，直至配额完成或来源池耗尽。
5. **依据入选分布独立构造 random。** `truncation.py` 使用独立随机顺序选择来源，按对应 split、任务的区间缺额选择输入长度，再随机选择原文 token 起点。AE 重建该跨度；LM 从同一连续区域取得 X 与紧邻后缀 Y。原文跨度重新分词后按实际 X 长度入桶。random 提示词接受首尾及 X/Y 切点落在句子或词内部，按共同内容标准判定质量。每篇文档的候选尝试数由 `random_candidates_per_document` 控制，默认 64。
6. **补足 random 配额。** 模型保留且通过去重的候选进入对应任务和长度区间；各区间完成后停止接收。两套数据的来源可以重合，也可以不同，来源归属始终由共享登记表决定。random 保存实际 `granularity=random` 与句界截断统计。
7. **检查并写入完成记录。** `audit.py` 验证原文跨度、X/Y 连续性、来源与近重复簇的 split、模型保留判定和合法容量，汇总任务数量、长度分位数及边界统计。`comparison.json` 核对两套数据每个 split 的 AE/LM 配额与输入区间计数。检查通过后写入对应 `preparation.json`；semantic 可以先完成，random 在跨版本检查通过后完成。配额不足会报告任务缺额，random 还会报告长度区间缺额。

当前配置的输入上限为 1024 tokens，LM 目标上限为 256 tokens，父片段按 1024-token 预算构造。memory 容量根据实际输入长度、压缩率和完整读取预算确定，访问时由训练采样器选择。

## 3. 模型接口、缓存与阶段执行

评分服务通过 `review_base_url` 指定，默认 `http://127.0.0.1:8000/v1`，模型名为 `Qwen/Qwen3.8-27B`。客户端使用直连 HTTP、结构化 JSON 输出、temperature 0，并通过 `chat_template_kwargs.enable_thinking=false` 关闭 thinking。并发数、输出上限和请求超时由配方控制；服务容量在真实部署后实测调整。[vLLM 结构化输出接口](https://docs.vllm.ai/en/latest/features/structured_outputs/)、[Qwen3.8 部署说明](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)

缓存键包含模型、服务地址、提示词、输出协议及完整 X/Y。已完成请求逐条落盘，重复候选复用结果。HTTP 错误、截断响应或格式错误会使当前阶段失败；失败目录保留诊断文件。后续执行可复用评分缓存，使用新构造目录；已完成 semantic 后可以独立启动 random 阶段。

评分服务就绪后，完整构造入口为：

```bash
uv run python -m latent_working_memory.data_preparation \
  --config configs/v1/pretrain_a800.json \
  --recipe configs/data_preparation/fineweb.json \
  --dataset-dir data/raw/HuggingFaceFW-fineweb \
  --output-dir data/v1/fineweb-independent \
  --score-cache data/v1/quality-cache/sample-scores.jsonl
```

`--stage sources` 只建立来源池，使用 `--dataset-dir`；`--stage semantic` 和 `--stage random` 读取已有池，使用 `--score-cache`。同一来源池各阶段沿用原始来源预算、去重参数、seed 和 split 定义。完成的阶段保持原目录，失败阶段重建时先将该阶段目录移至诊断目录，再执行对应阶段。

```text
fineweb-independent/
  source-pool.json            来源池标识、划分配置与统计
  sources.jsonl              候选原文、文档簇、split、基础判定
  semantic/
    train.jsonl dev.jsonl test.jsonl
    documents.jsonl          实际入选样本的原文与来源登记
    sample-decisions.jsonl   已评分候选的判定、位置与配额处理
    audit.json               契约检查与长度统计
    preparation.json         完成记录、入选直方图与评分协议
  random/                    同样的文件布局
  comparison.json            两套数据的配额和区间分布比较
```

样本 provenance 保存原文字符跨度、边界类型、tokenizer、文档簇、输入内容键及质量判定的缓存键。完成记录通过 `source_pool_id` 关联共享来源，random 额外记录其参考的 semantic `preparation_id`。

## 4. 独立抽查

`inspection.py` 对已完成的某个版本单独抽样。随机面板均匀抽取最终样本，分层面板按任务、粒度、长度抽样，每篇文档至多一个样本。面板包含实际 X/Y、边界类型、来源和留空的复核字段。

```bash
uv run python -m latent_working_memory.data_preparation.inspection sample \
  --data-dir data/v1/fineweb-independent/semantic \
  --output-dir artifacts/v1/data-preparation/fineweb-independent-inspection \
  --split train --examples 200 --seed 20260909

uv run python -m latent_working_memory.data_preparation.inspection judge \
  --inspection-dir artifacts/v1/data-preparation/fineweb-independent-inspection \
  --recipe configs/data_preparation/fineweb.json \
  --score-cache data/v1/quality-cache/inspection-scores.jsonl
```

`judge` 使用独立抽查提示词与缓存协议重新判定，保留人工或已有复核结果，每批写回进度，并生成 `summary.json`。人工可填写 `judgment`、`review_reason`、`reviewer`，再运行同一模块的 `summarize` 子命令。汇总分别记录 pass、fail、uncertain 和未复核数量；存在存疑或未复核样本时，报告缺陷比例的上下界。

抽查与构造是两个执行链路，抽查读取固定数据并写入独立产物。模型抽查用于定位遗漏，人工复核可用于识别同一评分模型反复出现的判断偏差。检查发现系统性问题后，修订主流程配方并生成下一份数据。

## 5. 实现与验证状态

| 模块 | 职责 |
|---|---|
| `config.py / __main__.py` | 配额、区间、评分接口配置与阶段入口 |
| `sources.py / quality.py / dedup.py` | 来源读取、基础属性检查、去重与来源分组 |
| `segmentation.py / fineweb.py / truncation.py` | 原文边界、多粒度 semantic 与独立 random 样本 |
| `scoring.py` | 最终样本的模型判定、结构化协议与缓存 |
| `pipeline.py / audit.py` | 两阶段入选、配额补足、共同来源契约和分布比较 |
| `inspection.py` | 独立抽样、模型或人工复核、汇总 |

本地 v1 的 65 项测试通过，覆盖配额与入选后分布、样本去重、共享来源隔离、原文偏移、容量、HTTP 评分接口、错误处理、缓存和抽查恢复。完整命令链使用本地测试 HTTP 服务和微型 Parquet 验证；Ruff 检查通过。

Qwen3.8-27B 权重已于 20260909 17:34:31 UTC+08:00 下载到服务器 `/data/bywei/models/Qwen/Qwen3.8-27B`。本次完成代码与本地接口验证。评分服务部署、真实 Qwen 判定质量与吞吐验证、正式配额的数据生成是后续执行步骤，正式训练验证的数据入口为 semantic。
