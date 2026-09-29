# 20260928_FineWeb 事实 QA 数据构造流程

创建时间：20260928 16:27:05 UTC+08:00
最后修订时间：20260929 10:32:05 UTC+08:00

本文维护 FineWeb 事实 QA 的可复用流程：来源隔离、长度选择、生成与补题、诊断与裁决、数据格式、恢复及文件入口。批次参数、实际命令和结果在各自执行报告中维护；训练目标和整体实验安排见[实验方案](20260926_step1_damage_guided_capacity_experiment.md)。

**当前实现：固定八段、仅构造 train、按缺额补题。** 建议的 6–10 段尚未实现，不能只改配置启用。已完成批次和统计产物统一列在文末索引，不作为所有运行必须采用的参数。

## 1. 输入、来源隔离与配置

使用项目根目录 `.venv`，依赖由 `pyproject.toml` 与 `uv.lock` 管理。输入为本地 FineWeb `sample-10BT` 原始 Parquet；下载入口固定来源 revision，显式禁用代理并支持续传。已有原文可直接复用。

当前来源隔离以“旧候选池”作为保守排除单位：按 `source.data_seed` 重建旧遍历顺序，跳过 `old_pool_documents` 篇；在随后 `scan_documents` 篇中做基础过滤和来源簇去重，再排除与旧池匹配的整个新簇。匹配包括文档 ID、规范化 URL、规范化正文，以及词 5-gram 集合的 Jaccard 相似度。继承来源簇的 train/dev/test 划分，只保留 train；按 `selection_seed + document_id` 稳定排序，最多检查 `max_train_inspections` 篇，冻结最多 `window.count` 篇，每篇一个窗口。

本地没有旧 AE／LM 成品的完整来源名单，现行入口也不接收这类名单；不能把“排除旧候选池”写成“只排除实际 AE／LM 文档”。扫描后改变候选范围可能影响去重簇与 split；扩大预算须新建批次，不能覆盖已有冻结结果。

| 配置部分 | 职责与当前约束 |
|---|---|
| `source` | 原文位置、revision、来源与选样 seed、排除／扫描／检查预算、去重阈值、split 比例 |
| `window` | 冻结文档上限、段数和段长；当前 `segments` 必须为 8，随机段长由 `min_segment_chars / max_segment_chars` 限定 |
| `qa` | 各段首轮候选与 train/gate 配额、答案长度、分池与抽样 seed、复核题数、补题轮数及余量 |
| `annotation` | Responses 地址、模型、reasoning effort、超时、并发、网络尝试及输出上限；当前客户端最多 4 并发、每请求最多尝试 3 次 |
| `prompts` | `generate / verify / document_review / answer / subagent_review` 五份 prompt 的路径 |
| 输出位置 | `artifacts_dir` 保存运行记录；`dataset_dir` 保存正式数据；`cache_dir` 保存可跨批次复用的请求缓存 |

源配置描述一个具体批次。创建新运行时使用不同的产物和数据目录；`prepare` 保存完整配置与实际 prompt 快照，后续仅允许改变服务 `endpoint`。模型、prompt、配额或选样规则变化均需新批次。

## 2. 长度依据与随机切分

所有长度使用 `len(original_text)`，包含空格与换行；字符位置采用 Python 左闭右开区间。构造阶段不加载 tokenizer，英文 tokens 按字符÷4 粗估；实际训练再检查真实分词长度和模型上下文预算。原文超过窗口上限仍可截取，不应按上限整篇丢弃。

### 来源分布与区间选择依据

20260929 对本地 15 个 Parquet 全量统计：扫描 14,868,862 篇，排除旧池 ID 100,000 篇、进一步排除 URL 精确匹配 757 篇和规范化正文匹配 137 篇，剩余 **14,767,968 篇文档记录**。统计覆盖全部划分，未做近重复聚类或剩余记录相互去重，因此是长度候选规模，不是最终独立 train 文档数或 QA 产量。

| 分位数 | P25 | 中位数 | P75 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| 原文字符数 | 905 | 1,801 | 3,467 | 6,095 | 8,937 | 21,697 |

| 原文至少字符数 | 粗估 tokens | 候选记录数 | 占剩余来源比例 |
|---|---:|---:|---:|
| 12,288 | 3,072 | 413,339 | 2.799% |
| 18,432 | 4,608 | 198,114 | 1.342% |
| 24,576 | 6,144 | 118,312 | 0.801% |
| 30,720 | 7,680 | 78,192 | 0.529% |
| 40,960 | 10,240 | 44,684 | 0.303% |

完整分位数、直方图和精确频数见[长度统计产物](../../artifacts/v3/fineweb-length-distribution_20260929/)。这项普查采用旧池加精确匹配的排除口径；正式选样额外执行近重复来源簇排除，两者不能混作同一分母。

**建议区间为 6–10 段、每段 3,072–4,096 字符，总窗口 18,432–40,960 字符。** 相比 24,576 字符门槛，长度候选增加 79,802 篇（67.45%）。优先增加段数变化和候选扫描范围，保留已试验过的段长；更短的段能否支撑既定 QA 密度需另测。此建议仍待实现：需同步推广来源切分、配置配额、组装校验、更新安排、prompts 和分段统计。届时每篇 N 从 `[6, min(10, len(text)//3072)]` 抽样，QA 为 8N 题；段数不会自然等量分布，报告应分 N 统计。

### 当前可执行的八段算法

1. 原文不足 `8 × min_segment_chars` 时记为 `length_failed`；不受段落或句子边界限制。
2. 用 `selection_seed + document_id` 初始化随机数，逐一抽取八个段长。每次在允许范围内抽样，并为其余段预留最低字符数。
3. 打乱段长顺序，再随机选择可容纳总长的原文起点；按累计段长直接切分，保持原文连续且不插入字符。
4. 冻结来源位置、窗口和段界。相同文档与 seed 可复现；长度足够就能切分，不存在 `boundary_failed`。题量不足不能换文档或重新挑选窗口。

每段为 3,072–4,096 字符时，当前八段窗口为 24,576–32,768 字符。边缘的不完整词句不妨碍切分，但生成 QA 时必须找到完整、段内自足的证据。

## 3. 生成、过滤、补题与使用安排

| 段位 | 首轮候选上限 | train | gate | 最终不同事实数 |
|---|---:|---:|---:|---:|
| 首段 `seg0` | 15 | 4 | 8 | 12 |
| 中间六段，每段 | 10 | 4 | 4 | 8 |
| 末段 `seg7` | 5 | 4 | 0 | 4 |
| 合计 | 80 | 32 | 32 | 64 |

每轮依次执行 **生成 → 程序定位 → 独立局部核验 → 累积候选全文审查 → 事实去重 → 检查逐段配额**。

- 生成器只以当前段为证据，返回 `fact_statement / question / answer / evidence_quote`。问题应明确主体、事件与时间范围；答案采用自然英文短语，最多 128 字符。
- 程序要求证据是段内唯一出现的连续原文片段，答案为证据的连续子串；字符位置由代码定位。答案出现多次时记录首次位置，不因此合并不同事实。
- 局部核验独立判断支持性、指代与答案质量。全文审查检查后文修正、范围歧义及跨段同事实重复，不能借后文补足早期题缺失的证据。
- 同一事实组只留一道代表题，优先最早具有完整证据的段，再按候选加入顺序选择。按固定 seed 分配 train/gate，事实组不跨池；总题数足够不能抵消某一段不足。
- 仅向缺额段请求 `缺额 + supplement_surplus` 道新候选，默认余量 2；提供已有事实与拒绝原因以避重和纠错。旧题拒绝状态保留，修正尝试使用新的轮次 QA ID 并重新核验。
- `round_index=0` 为首轮，之后默认最多追加三轮；生成器可以少给或返回零题。达标即停止，预算耗尽后仍不足才记为配额失败。API／协议／写盘错误属于运行中断，不能当作数据失败。

QA ID 为 `trajectory_id:segN:roundR:qaM`。补题预算与网络重试预算分开，重启不重置轮数。

使用安排离线固定：首段使用四道新训练题；首次更新使用首段四道旧训练题与八道门控题；后续更新使用当前段四道新题、上一段两道旧训练题加更早两道、上一段四道门控题加更早四道。更早题按历史段轮转，不能用未来段证据，不按运行中的模型表现重新选题。

## 4. 固定诊断、外部复核与定稿

`diagnose` 在全部标注完成后，按文档和段位分散抽取最多 96 题，包括配额失败文档。每题分别独立执行“问题＋证据”和“仅问题”回答，记录 EM/F1，并生成两份固定复核面板。它**不自动执行面板语义复核或主审裁决**。

复核者使用冻结的 `subagent_review` prompt 与面板中的原文、候选和诊断回答，在独立上下文检查证据、时间、答案与同事实关系，不提供前序核验理由。可用独立模型请求或其他明确指定的复核方式；实际执行方、模型、分工、调用与用量写入批次记录，模型复核不能称作人工复核。

主审保留原始复核判断，逐项核对标记项及其关联原文，形成 `<artifacts_dir>/resolved-review.json`：

```json
{"decisions": [{"qa_id": "<固定面板中的 QA ID>", "accepted": true, "reason": "", "same_fact_with": [], "evidence_prediction_correct": true}]}
```

必须恰好覆盖面板且不重复。拒绝项须有理由且不得同时合并；合并链接应指向同文档当前仍保留的候选，先清理指向已淘汰旧题的链接。每个证据诊断回答须有布尔语义判断。判断题目合格与判断诊断模型答对是两件事，不能互相替代。

`finalize` 应用拒绝与事实合并，使用已有富余候选重排题池；若有缺额且仍有预算，继续相同补题循环。标注与终审共用配置指定的每篇补题预算（当前默认三轮），不重新获得预算。原始标注和诊断面板保留，终审新增题不属于原面板的抽样结论。所有段位满足配额后才写入正式数据。

仅问题答对不能直接认作泄漏，抽样正确率也不代表全量质量或实际压缩记忆效果。报告保留原面板结果，不因删题重算为更高通过率。

## 5. 命令、数据格式与恢复

以下命令均在项目根目录运行；占位符应替换为具体批次配置或目录。下载入口只接受代码中固定的 FineWeb revision；无需在每次运行时重新下载。

```bash
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa.download \
  --directory <原文目录> --endpoint <HTTPS来源站点> --revision <固定revision> --workers 3
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa.analyze_lengths \
  --config <构造配置.json> --output <长度分析目录>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa prepare --config <构造配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa annotate --config <构造配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa diagnose --config <构造配置.json>
# 独立复核并完成主审裁决，保存 resolved-review.json 后：
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa finalize --config <构造配置.json>
```

`annotate --limit <文档数>` 可分批检查，后续以同一配置继续。长度普查可按已完成分片续跑，改变比较门槛直接读取精确长度频数；正式构造的选样仍须走 `prepare`。

正式输出只有 `train.jsonl` 和 `preparation.json`。JSONL 每行一条完整轨迹，保存文档／轨迹 ID、来源簇与 Parquet 位置、窗口和段界、原文、QA 及使用安排。QA 保存问题、答案、目标事实、事实组、证据／答案字符区间和 train/gate 角色；证据从原文按区间还原，不重复保存证据全文，也不预存 token ID。准备信息保存实际配置、统计、诊断、用量、模型和软件 provenance。

| 运行记录 | 用途 |
|---|---|
| `config.json / prompts.json / selection.json` | 配置和 prompt 快照、冻结文本及来源筛选统计 |
| `documents/doc-NNN.json` | 累积候选、最新审查和每轮缺额／新增事实／净变化／用量；完整轮次后保存 |
| `diagnostic-panel.json / diagnostic-answers.json / diagnostics.json` | 固定面板、两种回答、逐题及汇总分数 |
| `review-panel-*.json / resolved-review.json` | 复核输入与主审裁决；原始判断和理由另存于本批次 |
| `reviewed-documents/ / final-documents/` | 继承原轮数的终审状态、最终配额与剩余候选 |
| `requests.jsonl` 与共用缓存 | 实际调用、轮次、失败、用量，以及请求体、原始响应、解析结果和尝试记录 |
| `annotate-summary.json` | 标注完成／未完成数、逐段及逐轮产出汇总 |

同批次恢复使用原配置、原目录和同一命令；从最后完整轮次恢复，轮内已完成请求由内容缓存复用。原始响应已保存而解析未完成时可离线恢复；本地写盘失败不触发远端重试。终审开始后裁决冻结，修改裁决不能继续复用旧终审状态。并发派发遇致命错误停止新增请求，已发出的调用结束后保留实际用量。

报告按[数据构造记录规范](../../docs/data_construction_reporting.md)区分来源筛选、逐段 QA 过滤、首轮／补题／终审达标率，以及未入库合格题与质量拒绝。分母、不同事实、抽样范围、额外用量和墙钟时间均应明确。

## 6. 当前文件与产物索引

源码与配置表达现有接口；运行目录保存某次执行的事实。下列产物路径是当前实例，后续新批次不沿用其输出身份。

| 类别 | 位置与职责 |
|---|---|
| 构造源码 | [fineweb_qa/](../../src/latent_working_memory/data_preparation/fineweb_qa/)：`__main__.py` 命令行；`pipeline.py` 阶段、轮次与恢复；`sources.py` 来源与随机切分 |
| 标注与题池 | 同目录 `annotation.py` 请求与证据定位；`assembly.py` 事实去重、角色及更新安排 |
| 诊断与定稿 | 同目录 `diagnostics.py` 面板与评分；`finalization.py` 应用主审裁决 |
| 下载与长度分析 | 同目录 `download.py` 无代理下载；`analyze_lengths.py` 池外原文长度普查 |
| 共享来源实现 | [pretrain/](../../src/latent_working_memory/data_preparation/pretrain/) 中的 `sources.py / dedup.py / quality.py / fineweb.py` |
| 构造配置实例 | [fineweb-factqa-8192-doc2k.json](../../configs/data_preparation/fineweb-factqa-8192-doc2k.json)：已完成八段小试的源配置；完整参数以本批次快照为准 |
| 五份 prompts | [fineweb_factqa/](../../configs/data_preparation/prompts/fineweb_factqa/) |
| 回归测试 | [tests/v1/](../../tests/v1/) 中八个 `test_fineweb_qa_*.py`，覆盖下载、来源、长度统计、标注、组装、诊断、终审和完整管线 |
| 文档分工 | 本文维护流程；[来源调研](20260927_step1_qa_data_survey.md)维护来源依据；[整体实验方案](20260926_step1_damage_guided_capacity_experiment.md)维护训练与比较设计；[记录规范](../../docs/data_construction_reporting.md)维护通用报告口径 |
| 原始 FineWeb | [sample-10BT/](../../data/raw/HuggingFaceFW-fineweb/sample-10BT/) |
| 已完成小试 | [执行报告](../../artifacts/v3/fineweb-qa-random-topup_20260928/20260928_fineweb_qa_random_topup_report.md)、[运行目录](../../artifacts/v3/fineweb-qa-random-topup_20260928/)：具体设置、命令、逐篇结果、复核和样例 |
| 正式 QA 数据 | [fineweb-factqa-random-topup-8192-doc2k_20260928/](../../data/fineweb-factqa-random-topup-8192-doc2k_20260928/)：`train.jsonl / preparation.json` |
| 请求缓存 | [fineweb-qa-requests/](../../artifacts/data_preparation/fineweb-qa-requests/) |
| 全量长度统计 | [fineweb-length-distribution_20260929/](../../artifacts/v3/fineweb-length-distribution_20260929/)：`config.json / summary.json / length-counts.json / files/ / run.log` |
| 同批候选门槛比较 | [20260929_length_threshold_comparison.json](../../artifacts/v3/fineweb-qa-random-topup_20260928/20260929_length_threshold_comparison.json)：固定 2,000 篇的 18,432／24,576 字符比较 |

代码验证命令：

```bash
.venv/bin/python -m pytest tests/v1/test_fineweb_qa_*.py -q
.venv/bin/python -m ruff check src/latent_working_memory/data_preparation/fineweb_qa tests/v1/test_fineweb_qa_*.py
.venv/bin/python -m ruff format --check src/latent_working_memory/data_preparation/fineweb_qa tests/v1/test_fineweb_qa_*.py
```
