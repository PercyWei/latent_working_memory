# 20260929_FineWeb 事实 QA 构造接口

创建时间：20260929 16:03:54 UTC+08:00
最后修订时间：20261008 11:39:39 UTC+08:00

本目录实现来源筛选、6–10 段随机切分、QA 生成、全量局部核验、全文审查、配额补题和统一交付。运行记录保存在 `artifacts_dir`，正式数据保存在 `dataset_dir`。

## 配置与入口

- [来源配置](../../../../configs/data_preparation/fineweb-factqa/fineweb-factqa-source-pool.json)：原始语料、来源排除、去重、划分、窗口和批次份额。
- [批次模板](../../../../configs/data_preparation/fineweb-factqa/fineweb-factqa-6to10-batch000.json)：QA 配额、模型、并发、重试及输出预算。
- [整轮配置](../../../../configs/data_preparation/fineweb-factqa/fineweb-factqa-train1000.json)：来源配置、批次模板、train 目标、根目录和显式排除的数据集。

在项目根目录运行：

```bash
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa.campaign run --config <本轮配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa.campaign report --config <本轮配置.json>
```

控制器依次执行 `prepare → annotate → finalize`。调试或恢复单个阶段时，使用 `latent_working_memory.data_preparation.fineweb_qa <阶段> --config <批次配置.json>`；`annotate --limit N` 可限定本次处理文档数。

## 来源、切分与配额

来源准备按固定规则排除 ae+lm 旧来源池，并按 ID、规范化 URL、全文和近重复关系去重。train/dev/test、候选顺序及窗口在模型请求前冻结；每轮批次从 batch-000 编号，按划分取不重叠切片。

每条轨迹随机 6–10 段，每段 3,072–4,096 字符，使用任意字符边界。字符区间为左闭右开；窗口位置相对原始文章，段落、证据及答案位置相对窗口正文。token 长度按字符数÷4估算。

N 段轨迹首轮生成候选的上限为 `[15, 10, ..., 10, 5]`，最终需要 4N 道任务题和 4N 道 gate 题。每段任务题为 4 道，gate 配额为 `[8, 4, ..., 4, 0]`。任务题与 gate 题使用不同事实；dev/test 的任务题标为 evaluation，使用安排覆盖全部已读前缀。

## 生成与全量检查

| 模型请求 | 输入与职责 | Prompt |
|---|---|---|
| `generate` | 当前段原文、已有事实及拒绝理由；生成问题、短答案、事实陈述和证据引文 | [generate.txt](../../../../configs/data_preparation/prompts/fineweb_factqa/generate.txt) |
| `verify` | 当前段及全部程序通过的候选；逐题检查证据支持、问题范围、答案自然性和限定条件 | [verify.txt](../../../../configs/data_preparation/prompts/fineweb_factqa/verify.txt) |
| `document_review` | 完整窗口、段边界及累计局部通过的候选；检查后文修正、歧义和跨段事实重复 | [document_review.txt](../../../../configs/data_preparation/prompts/fineweb_factqa/document_review.txt) |

生成结果先经过程序检查：字段完整、引文唯一且连续、答案位于引文内、字符位置合法，并满足数量及答案长度限制。每个有有效候选的段落调用一次局部核验，每篇文档每轮调用一次全文审查。

全文审查逐题返回接受／拒绝及事实组。同一事实的改写和反向问法归为一组；程序优先保留证据最早完整出现的段落中的代表，再按候选顺序选择。每道题的证据必须在所属段内独立支持答案。

每轮检查后重新计算逐段配额。缺额段按“缺额＋余量”补题，补题同样经过两级全量检查。首轮后最多补三轮；拒绝的旧候选保持拒绝，修正后的题使用新的候选 ID。配额齐全或补题预算耗尽时结束标注。

`finalize` 使用保存的候选与全文审查决定，重新组装并校验事实隔离、角色配额、证据位置和使用安排。完整达标轨迹写入批次数据，配额不足记为 `quota_shortfall`；定稿和整轮汇总均由本地程序完成。

## 请求、缓存与恢复

请求以模型、输入、prompt、结构化输出 schema 和输出预算的散列标识。`request.json` 保存请求；`attempt-N.json` 保存实际输出约束、状态、耗时和用量；完整合格响应进入 `parsed.json`，原始响应保留供追溯。同轮批次共用 `requests/`，新一轮使用独立目录。

每次决策必须完整覆盖候选 ID，且不重复；拒绝需要理由，接受的全文审查决定需要事实组。网络与响应错误共用最多三次尝试的预算，恢复时沿用已消耗次数。无效 JSON、错号、漏答和输出截断先归档，再重试同一请求；预算耗尽记为 `annotation_contract_failed`。

服务内容过滤记为 `content_filtered`，文档退出构造。已完成轮次、原始响应和真实用量保留，批次继续处理其他冻结文档。网络重试耗尽、未知服务状态、模型配置不符或写盘错误会停止运行。

每轮标注完成后保存文档状态。恢复使用原配置、目录和缓存；已完成文档及持久化失败记录直接复用。文件锁防止同轮重复启动，恢复前检查旧阶段进程是否仍存活。修改构造规则时使用新的运行目录。

## 产物与正式交付

```text
<artifacts_dir>/
  config.json、status.json、campaign.log
  batch-configs/batch-NNN.json
  batches/batch-NNN/
    config.json、prompts.json、selection.json
    documents/                     # 已完成轮次、候选及审查决定
    failed-documents/              # 文档级失败
    final-documents/               # 定稿结果与逐段缺额
    dataset/                       # 批次 train/dev/test 和 preparation
    prepare.log、annotate.log、finalize.log
  requests/                        # 本轮请求缓存
<dataset_dir>/
  source-pool.json                  # 本轮来源及显式排除依据
  train.jsonl
  dev.jsonl
  test.jsonl
  preparation.json                 # 整轮统计、来源记录与完成标记
```

`target_train_trajectories` 只统计本轮定稿 train。每批结束后计数，达到目标即停止派发新批次；最后一批允许产生余量。随后按批次顺序合并 JSONL，检查来源、划分、重复和数量，最后写出整轮 preparation 作为完成标记。重新汇总使用：

```bash
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa.campaign finalize --config <本轮配置.json>
```

preparation 保存实际参数和 prompts、逐阶段数量、补题记录、失败原因、用量、逐文档结果及批次索引。质量结论以模型核验和程序契约检查为依据。

`report` 使用本轮冻结配置，只读统计已定稿批次和在途请求。`completed_batches_usage` 是已定稿批次成本，`all_batches_usage` 包括在途用量，`dataset_published` 表示整轮 preparation 是否存在。统计覆盖各划分和失败样本，按请求及尝试编号去重；缓存命中不重复计入，缺失 usage 单列且不估补。交互助手和自动汇报用量不在构造 API 统计内。

## 新数据集的来源排除

每个 preparation 的 `used_sources` 只记录本轮已分配原文，包括构造失败文档。记录包含文档 ID、来源簇和 Parquet 位置，排除单位是整篇原文。

首次构造设置 `previous_datasets: []`；后续显式列出要排除的数据目录，例如 `["data/A", "data/B"]`。初始化合并这些目录各自的 `used_sources`，按文档 ID 去重。显式排除集合冻结在本轮 source-pool 的 `excluded_sources` 中，恢复只读取本轮快照。

规则一致且存在未带历史排除的完整来源快照时，直接筛选候选；其他情况重新扫描并应用本次明确指定的排除集合。扩大范围时另建来源配置增加 `scan_documents`，保持原始语料、划分和 ae+lm 排除规则一致。来源耗尽仍未达标时明确报错。

## 实现与验证

`campaign.py` 负责整轮调度与报告，`sources.py` 负责来源和窗口，`annotation.py` 负责请求与证据定位，`pipeline.py` 负责生成、核验、补题和批次定稿，`assembly.py` 负责事实组、配额与使用安排，`publication.py` 负责整轮交付，`storage.py` 负责原子写入。`download.py` 与 `analyze_lengths.py` 分别提供下载和长度分析工具。

```bash
.venv/bin/python -m pytest tests/v1/test_fineweb_qa_*.py -q
.venv/bin/python -m ruff check src/latent_working_memory/data_preparation/fineweb_qa tests/v1/test_fineweb_qa_*.py
```
