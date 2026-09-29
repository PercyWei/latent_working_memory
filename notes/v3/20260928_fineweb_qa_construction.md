# 20260928_FineWeb 事实 QA 数据构造流程

创建时间：20260928 16:27:05 UTC+08:00
最后修订时间：20260929 11:49:42 UTC+08:00

本文维护可复用的来源隔离、长度协议、批次划分、QA 构造、质量复核与恢复接口。具体运行与产率保存在批次报告；训练目标和方法比较见[整体实验方案](20260926_step1_damage_guided_capacity_experiment.md)。

**当前实现：6–10 段、冻结来源池后分批、train/dev/test 三划分、每篇每段抽样与独立裁决。** 工程验证使用离线固定样例；新来源池和正式首批尚未执行，等待用户检查。旧八段小试的 13 条轨迹／832 题仅代表旧协议结果。

## 1. 先冻结来源池，再分批构造

来源池配置与批次配置分开，分别采用唯一格式，不自动兼容旧八段配置。

| 配置 | 职责 |
|---|---|
| [fineweb-factqa-source-pool.json](../../configs/data_preparation/fineweb-factqa-source-pool.json) | `source` 定义原文、旧池、扫描范围、去重与 split seed；`window` 定义段数与字符范围；`batch_counts` 固定每批各 split 的取样份额；`pool_dir` 保存索引 |
| [fineweb-factqa-6to10-batch000.json](../../configs/data_preparation/fineweb-factqa-6to10-batch000.json) | 引用 `source_pool_dir` 与 `batch_index`；定义 QA／补题、抽检、模型、六份 prompts、请求缓存、执行目录和正式输出目录 |

`prepare-pool` 一次性读取约定扫描范围：按原 seed 重建并排除旧候选池，对扫描原文做基础过滤与来源簇去重，排除与旧池匹配的整个新簇。匹配包含 ID、规范化 URL、规范化正文及词 5-gram 集合的 Jaccard 近重复；再按簇分配 train/dev/test，按固定 seed 排序并冻结随机窗口。

本地缺少 AE／LM 成品的完整来源名单，仍保守排除整个旧候选池，不能称为只排除实际进入 AE／LM 的文档。来源池 `source-pool.json` 保存 `pool_id`、实际配置、统计、来源位置、簇、split、窗口和段界，**不复制原文或 token IDs**。源配置不变时复用既有池；改变扫描范围、划分、段长或批次份额须使用新池，不在旧池上追加重聚类。

批次 b 对每个 split 取该 split 固定列表的 `[b × count, (b+1) × count)`。因此不同编号批次不重叠，跨批次不再计算去重簇或 split。尾批按实际剩余数量取样并报告短缺，某一 split 耗尽不妨碍其他 split 继续；全部耗尽时报错。`prepare` 读取这些位置的原文，核对文档 ID，保存该批 `selection.json`，记录池 ID、批次编号与各 split 取值范围。

每批使用独立 `artifacts_dir` 和 `dataset_dir`；下一批复制批次配置，递增 `batch_index` 并修改输出目录。重复相同编号属于同一来源切片，不应作为新增独立数据合并。不同来源池之间的重叠不由批次编号保证，不能未经核查拼接。

## 2. 长度依据与可变段数

字符数采用 `len(original_text)`，包含空格和换行，位置为 Python 左闭右开区间。构造不加载 tokenizer，tokens 按字符÷4 粗估；实际训练再核对真实长度。长于窗口上限的原文可截取，不按上限整篇丢弃。

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

现采用 **6–10 段、每段 3,072–4,096 字符，总窗口 18,432–40,960 字符**。保持单段长度以控制 QA 密度变化，通过段数扩大总长度范围。全量长度候选足以支持更大范围选样，不以缩短每段来追求固定小预算下的高通过率。

随机切分按以下顺序执行：

1. 原文不足 `min_segments × min_segment_chars` 时记为长度失败。
2. 以 `selection_seed + document_id` 初始化随机数，在 `[min_segments, min(max_segments, len(text)//min_segment_chars)]` 中抽取 N。
3. 依次抽取 N 个段长，每次为剩余段预留最低字符数；打乱段长，再随机选择可容纳总长的原文起点。
4. 按累计长度直接切分，无词句／段落边界要求，保持原文连续。QA 证据仍需完整位于所属段内，不能补用切断部分或未来段。

相同来源与 seed 可复现；长度足够就能分段，不存在 `boundary_failed`。短一些的原文可容纳的 N 较少，N=6…10 不会自动等量，统计须按段数分组。固定窗口后不得因 QA 失败换文档或重新挑窗口。

## 3. QA 配额、角色与时间顺序

配额由 `assembly.qa_quotas(N)` 统一计算，不在配置中保存固定长度数组。

| 段位 | 首轮候选上限 | 任务题 | gate 题 | 不同事实 |
|---|---:|---:|---:|---:|
| 首段 | 15 | 4 | 8 | 12 |
| 中间 N−2 段，每段 | 10 | 4 | 4 | 8 |
| 末段 | 5 | 4 | 0 | 4 |
| 合计 | 10N | 4N | 4N | 8N |

train 的任务题角色为 `train`；dev/test 为 `evaluation`，不标记为训练题；门控题统一为 `gate`。6／8／10 段分别输出 48／64／80 题。同一事实组只选一题，优先最早具有完整证据的段，再按候选顺序选代表；任务与门控按固定 seed 分池，不能共享事实。

`usage` 统一保存 `task_new_qa_ids / task_old_qa_ids / gate_qa_ids`：

- train：每段四道新题；首次更新使用首段四道旧题；之后旧题取上一段两道与更早两道，跨步轮转。
- dev/test：新题为当前段四道，旧题为全部已读历史段的任务题；最后一步覆盖全部 4N 道评估题。
- 三个 split 的门控安排相同：首段无门控，首次更新取首段八题，之后取上一段四题加更早四题。旧任务题与门控证据必须早于当前段。

## 4. 生成、过滤与有界补题

每轮执行 **生成 → 程序定位 → 独立局部核验 → 累积候选全文审查 → 事实去重 → 逐段配额检查**。

生成只以当前段为证据，返回 `fact_statement / question / answer / evidence_quote`。问题须明确主体与必要范围；答案是自然短语，最多 128 字符。证据须是段内唯一出现的连续原文片段，答案是其连续子串，偏移由代码定位。局部核验检查支持性与问题质量，全文审查检查后文修正、范围歧义和跨段同事实重复。

只向缺额段请求 `missing_total + supplement_surplus` 道新候选，默认余量 2；附已有事实与历史拒绝原因。被拒旧题不能恢复为通过，修正后的新尝试使用 `trajectory_id:segN:roundR:qaM`，再次完整核验。模型可返回不足数量或零题；全篇富余不能抵消局部短缺。

首轮为 round 0，默认最多补三轮，**标注与终审共用每篇累计预算**。达标立即停止；耗尽后仍不足记为配额失败。网络／协议／写盘错误记为运行中断并保存错误，不伪装成不合格数据。dev/test 使用冻结的同一构造规则，不能根据其结果调 prompt 或配额。

## 5. 随批次增长的诊断、复核与裁决

`review.qas_per_segment` 默认 1：每篇每段从去重后的候选中按固定 seed 抽取一题，题少则全取，无题则显式记录空段。覆盖全部冻结文档，包括配额失败的文档，**没有全批最多 96 题的上限**；200 篇 6–10 段文档最多形成 1,200–2,000 道初始抽检题。

1. `diagnose` 冻结面板，逐题执行“问题＋证据”和“仅问题”两种回答，保存原始预测及 EM/F1。按文档输出 `review-inputs/`，同时记录每篇段数、split、实际抽检数与空段。
2. `review` 对每篇使用独立请求和 `review.txt`，不提供前序生成／核验理由。检查证据、范围、时间、答案、事实重复及诊断回答语义正确性。
3. 任何拒绝、同事实链接、诊断错误或非空理由均进入 `adjudicate.txt` 的第二次独立请求。裁决者同时看到原文、复核提议与候选当前状态，可确认或推翻提议；过滤指向已淘汰候选的链接。
4. 代码严格检查面板覆盖、布尔判断、拒绝理由、同文档保留候选链接及被同时拒绝的目标。原始判断和裁决分别保存；无效裁决中止当前阶段并保留错误，不能静默跳过或发布。若模型反复返回非法裁决，可在原始复核与证据基础上人工填写该文档的 `reviews/doc-NNN/resolved.json`（含完整面板决定及理由），再运行 `review`；同一验证器会检查外部裁决，不能绕过配额或证据诊断判断。终审补题的对应路径在 `supplement-reviews/doc-NNN/round-R/resolved.json`。
5. 全部文档完成后汇总 `resolved-review.json`，终审前可修正合法的单文档裁决并重新汇总，终审开始后禁止改变裁决，按 split／段数汇总抽检、拒绝和证据回答语义判断；已完成文档在续跑时不重复调用。

拒绝项必须有理由且 `same_fact_with=[]`。模型复核与裁决是不同请求，但不代表统计独立，更不等于人工复核。仅问题答对不是泄漏证明；抽样质量不能推为全量正确率或训练收益。

`finalize` 应用裁决、重排富余候选并重算配额；若剩余预算允许，继续补题。**终审补题后新进入不同事实候选池的题全部补做双条件回答、独立复核和必要裁决**。这些结果单独存于 `supplement-reviews/`，不改初始固定面板的分母。复核再拒绝导致缺额时继续使用同一剩余轮数，耗尽即失败。中断时保存待复核题和已完成生成轮次，恢复不会再次消耗生成轮数。

## 6. 执行、保存与恢复

在项目根目录 `.venv` 执行，下列为待执行命令模板，不是运行记录：

```bash
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa prepare-pool --config <来源池配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa prepare --config <批次配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa annotate --config <批次配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa diagnose --config <批次配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa review --config <批次配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa finalize --config <批次配置.json>
```

下载与长度普查是独立入口 `fineweb_qa.download`、`fineweb_qa.analyze_lengths`；普查使用来源池配置的 `source`，更改比较门槛可直接读取精确长度频数。`annotate --limit <数量>` 支持同批分次处理，后续阶段要求该批全部标注完成。

| 保存位置 | 内容 |
|---|---|
| 来源池 `source-pool.json` | 池 ID、固定来源配置、各 split 索引、字符窗口、段数统计和批次份额；无正文副本 |
| 批次 `config.json / prompts.json / selection.json` | 实际批次配置、六份 prompt、池身份、取样范围及当前批次文本 |
| `documents/ / annotate-summary.json` | 原始候选、每轮决定与缺额、首轮和补题后的统计 |
| `diagnostic-*.json / diagnostics.json / review-coverage.json` | 固定初始面板、两种回答、评分与实际覆盖 |
| `review-inputs/ / reviews/ / resolved-review.json / review-summary.json` | 文档级输入、原始复核、独立裁决、最终决定和分组统计 |
| `reviewed-documents/ / supplement-reviews/ / final-documents/` | 终审状态、待复核的新题、补充诊断与裁决、最终配额 |
| `requests.jsonl` 与共用缓存 | 实际请求阶段／轮次、用量、重试、原始响应与解析结果 |
| 正式输出目录 | `train.jsonl / dev.jsonl / test.jsonl / preparation.json`，空划分仍写空文件；准备信息记录池 ID、批次范围及分 split／N 结果 |

每个 JSONL 行是一条完整轨迹，保存来源、原文与字符边界、QA 事实组、角色和使用安排。证据从原文按位置还原，不另存重复证据全文或 token IDs。候选不足的文档保留在运行记录，不写入正式划分文件；`preparation.json` 在所有文档成功完成定稿阶段后写入。

`prepare` 冻结批次配置和 prompts，后续仅允许改变服务 endpoint。改变模型或构造规则需新运行，不能修改旧批次；旧八段产物不自动迁移。请求内容缓存可跨批次复用，网络重试最多三次；致命错误停止新增派发，已发出的请求结束并记录用量。各完整轮次及复核结果持久化，恢复使用原配置、原目录和相同命令。

报告遵循[数据构造记录规范](../../docs/data_construction_reporting.md)，分开统计来源长度准入、各轮事实净变化、独立复核错误、补题用量和最终轨迹达标率。按 N 与 split 报告，运行中断与配额失败分别计数，不把未入库候选都视为质量错误。

## 7. 待检查配置与现有产物索引

待检查的来源池计划扫描旧池之后 200,000 篇，划分比例沿用 90%／5%／5%；每批份额为 train 150、dev 25、test 25，首批最多 200 篇。它们是**计划预算**，未承诺实际来源数；来源不足会在批次范围统计中显示。全量池准备的性能尚未实测，用户检查前不运行真实选样或模型标注。

| 内容 | 位置 |
|---|---|
| 核心代码 | [fineweb_qa/](../../src/latent_working_memory/data_preparation/fineweb_qa/)：`sources.py` 冻结池与批次；`assembly.py` 动态配额与角色；`pipeline.py` 六个构造阶段；`annotation.py` 六类模型请求 |
| 诊断、裁决与保存 | 同目录 `diagnostics.py` 抽样／输入／评分；`finalization.py` 应用裁决；`storage.py` 原子 JSON 写入 |
| 下载与统计 | 同目录 `download.py / analyze_lengths.py`；共享来源逻辑位于 [pretrain/](../../src/latent_working_memory/data_preparation/pretrain/) |
| 当前配置与 prompts | [来源池配置](../../configs/data_preparation/fineweb-factqa-source-pool.json)、[首批配置](../../configs/data_preparation/fineweb-factqa-6to10-batch000.json)、[六份 prompts](../../configs/data_preparation/prompts/fineweb_factqa/) |
| 待生成的新池／首批目录 | `data/fineweb-factqa-source-pool_20260929/`；`artifacts/v3/fineweb-qa-6to10_20260929/batch-000/`；`data/fineweb-factqa-6to10_20260929/batch-000/`；均尚未创建 |
| 八段历史设置 | [归档配置](../../configs/archive/data_preparation/fineweb-factqa-8192-doc2k.json)，复现旧代码使用提交 `68157cb` 和旧批次实际 prompt 快照 |
| 八段历史运行 | [小试报告](../../artifacts/v3/fineweb-qa-random-topup_20260928/20260928_fineweb_qa_random_topup_report.md)、[正式数据](../../data/fineweb-factqa-random-topup-8192-doc2k_20260928/)，保留原八段格式与结果 |
| 来源长度依据 | [全量统计](../../artifacts/v3/fineweb-length-distribution_20260929/)、[同批 2,000 篇比较](../../artifacts/v3/fineweb-qa-random-topup_20260928/20260929_length_threshold_comparison.json) |
| 原文与缓存 | [sample-10BT/](../../data/raw/HuggingFaceFW-fineweb/sample-10BT/)、[请求缓存](../../artifacts/data_preparation/fineweb-qa-requests/) |
| 验证与相关文档 | [tests/v1/](../../tests/v1/) 中 `test_fineweb_qa_*.py`；[整体实验](20260926_step1_damage_guided_capacity_experiment.md)、[历史来源调研](20260927_step1_qa_data_survey.md) |

本次工程验证通过 84 项离线测试，覆盖 6–10 段 × 三划分的配额与证据时间顺序、来源池复用与批次不重叠、按段抽样、裁决校验、断点恢复及共享补题预算耗尽；Ruff 检查通过。测试使用临时 Parquet 与固定响应，不代表真实模型质量或生产吞吐验证。复查命令：

```bash
.venv/bin/python -m pytest tests/v1/test_fineweb_qa_*.py -q
.venv/bin/python -m ruff check src/latent_working_memory/data_preparation/fineweb_qa tests/v1/test_fineweb_qa_*.py
.venv/bin/python -m ruff format --check src/latent_working_memory/data_preparation/fineweb_qa tests/v1/test_fineweb_qa_*.py
```
