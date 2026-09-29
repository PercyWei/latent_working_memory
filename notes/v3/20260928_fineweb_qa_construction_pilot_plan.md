# 20260928_FineWeb 事实 QA 构造流程与小规模试验

创建时间：20260928 16:27:05 UTC+08:00
最后修订时间：20260928 21:16:25 UTC+08:00

**当前流程：随机字符切分，按段缺额补题，最多追加三轮。** 代码已完成离线验证；当前新批次尚未执行模型标注。旧小试采用语义边界、无补题：冻结 12 篇、正式 1 条轨迹和 64 道 QA，其原始数据与[报告](../../artifacts/v3/fineweb-qa-pilot_20260928/20260928_fineweb_qa_pilot_report.md)保留，不作为新流程产率。

关联：[实验方案](20260926_step1_damage_guided_capacity_experiment.md)、[来源范围](20260927_step1_qa_data_survey.md)、[正式配置](../../configs/data_preparation/fineweb-factqa-8192-doc2k.json)、[数据构造记录规范](../../docs/data_construction_reporting.md)。

## 固定规则

| 项目 | 当前设置 |
|---|---|
| 来源 | FineWeb `sample-10BT`，revision `9bb295ddab0e05d785b879661af7260fed5140fc`；原始 15 个 Parquet 已在本地 |
| 排除与划分 | 以来源 seed `20260907` 重建并排除旧 100,000 篇候选池及匹配簇；按既有来源簇规则划分 train/dev/test，仅使用 train |
| 预算与选样 | 旧池后扫描最多 3,000 篇，按 selection seed `20260928` 的稳定顺序检查最多 2,000 篇 train，冻结最多 24 篇，每篇一个窗口 |
| 文本规格 | 连续八段；每段随机 3,072–4,096 字符，总长自然落在 24,576–32,768 字符，约 6–8k tokens |
| 标注模型 | 本地 Responses `gpt-6-sol`、`medium`；最多 4 个在途请求，单请求临时错误最多尝试 3 次 |
| QA 配额 | 每篇 64 个不同事实，训练题 32、门控题 32；各段必须分别达标 |
| 补题 | 首轮后最多 3 轮，每个缺额段请求“缺额＋2”道候选；标注与终审共用预算 |
| 诊断 | 固定抽取最多 96 题，双条件回答和独立上下文的模型复核；不记为人工复核 |

字符采用 Python 字符位置，包含空格与换行，区间左闭右开。构造按 `len(text)/4` 估算 tokens，不加载 tokenizer；训练再用实际 tokenizer。`doc2k` 表示检查预算，不代表最终文档数。

## 1. 来源筛选与随机字符切分

1. 复用旧来源遍历、基础过滤、来源簇去重和 split 规则，保留 train 候选，按 seed 与文档 ID 排序。
2. 原文不足 `8 × min_segment_chars` 时记为 `length_failed`。否则以 seed 与文档 ID 初始化独立随机数。
3. 逐一随机抽取段长，每次为尚未抽取的段预留最低长度。例如剩余字符预算为 `B`、剩余段数为 `r` 时，在 `[min_segment_chars, min(max_segment_chars, B − (r−1) × min_segment_chars)]` 内抽取整数。
4. 打乱八个段长，再在原文中均匀随机选择可容纳总长度的起点，按累计长度直接切分。无需段落、句子、词或空格边界；无回溯、无失败重试。
5. 冻结文本、原文位置与段边界。原文长度足够即能成功，取消 `boundary_failed`；恰好达到最低长度时，八段自然都是最低长度。

同一文档与 seed 可重复得到相同窗口；不同文档独立采样，段长与位置具有多样性，不要求数值绝不重复。QA 生成忽略边缘的不完整片段，证据仍需完整位于所属段内。生成或补题失败时不更换原文或窗口。

`selection.json` 保存来源及筛选统计，额外记录 `train_uninspected`；预算未覆盖的文档与淘汰文档分别计数。

## 2. 生成、过滤与补题

| 段位 | 首轮候选上限 | train | gate | 所需不同事实 |
|---|---:|---:|---:|---:|
| seg0 | 15 | 4 | 8 | 12 |
| seg1–seg6，每段 | 10 | 4 | 4 | 8 |
| seg7 | 5 | 4 | 0 | 4 |
| 合计 | 80 | 32 | 32 | 64 |

每轮依次执行：**生成 → 程序定位 → 独立局部核验 → 累积候选全文审查 → 事实去重 → 检查逐段配额**。

- 生成从当前段的明确事实出发，返回 `fact_statement / question / answer / evidence_quote`。补题额外提供整篇已保留事实及该段之前的拒绝原因，用于避重与纠错；这些反馈不能充当证据。
- 程序要求证据是段内唯一出现的连续原文片段，答案是证据内连续片段且不超过 128 字符。偏移由代码定位；答案多次出现时取第一次。
- 局部核验独立判断证据支持、问题范围和答案质量。全文审查检查后文修正、指代歧义及跨段同事实重复；每轮审查累积局部通过的候选，保留逐题决策。
- 每个事实组只保留一个代表，优先证据所在段最早的题，再按候选加入顺序选择。各段的 train/gate 池按固定 seed 分配，事实组不能跨池；全篇富余不能抵消某一段的短缺。
- `round_index=0` 为首轮；只对未达标段追加 `缺额 + supplement_surplus` 道候选，默认余量 2。完成过滤后重新计算缺额，达标立即停止，最多执行 `round_index=1,2,3`。
- 每轮允许返回不足数量甚至零题。被拒的旧 QA 不恢复为通过；修正问题或证据应生成带新轮次 ID 的新候选，并重新接受完整核验。

QA ID 采用 `trajectory_id:segN:roundR:qaM`，保留跨轮唯一性。`max_supplement_rounds=3` 与网络重试 `max_attempts=3` 是不同预算，均不会因重启而重置。

## 3. 固定诊断、裁决与定稿

标注结束后，从全文通过且去重的候选中按文档及段位分散抽取最多 96 题，包括轨迹配额失败的文档。每题独立执行“问题＋证据”和“仅问题”两种回答，记录 EM/F1。两份复核面板由不同上下文的 subagent 分工检查，主 agent 对标记项裁决，保存 `resolved-review.json`。

`finalize` 先应用拒绝与同事实合并，用已有富余候选重新分配题池。若仍缺额且尚有补题预算，则按同一循环补题；首轮及标注补题的轮次继承，不再获得三轮。终审已拒的旧题不会复活，已裁决的同事实关系在后续核验中继续生效。预算耗尽后仍缺额才记为配额失败；请求、协议或写盘错误属于运行中断，保留状态续跑，不伪装成数据不合格。

终审后补题照常经过程序、局部和全文检查。**原 96 题面板及其诊断结果保持固定；终审后新增题不属于这份抽样复核，其质量不能用原面板结果代替。** 正式数据只写出全部段位达标的轨迹，每篇选 64 题；未使用的富余题与失败轨迹中的合格候选仍保留在运行记录中。

沿用八步使用安排：首次写入使用当前段的四道训练题；首次更新使用首段四道旧训练题与八道门控题；后续更新按固定安排轮转旧段训练与门控题，禁止使用未来段证据。

## 4. 状态、恢复与统计

`prepare` 冻结配置与五份 prompt。后续阶段要求一致，仅服务 `endpoint` 可变；改选样、模型、prompt 或补题规则须使用新批次。当前配置已切换到新的 `random-topup` 目录，旧批次不做格式迁移或兼容读取。

| 记录 | 内容 |
|---|---|
| `documents/doc-NNN.json` | 标注状态：累积候选、最新审查和组装结果、每轮生成／核验／审查／配额及用量；每个完整轮次后保存 |
| `reviewed-documents/doc-NNN.json` | 定稿阶段继承的状态、固定裁决及剩余补题过程；保留原 `documents/` 供诊断追溯 |
| `final-documents/doc-NNN.json` | 最终配额、短缺段、候选 ID 及实际补题轮数 |
| `requests.jsonl` | 阶段、文档、轮次、实际请求、缓存、重试和 usage；请求缓存还保存请求体与原始响应 |
| `annotate-summary.json`、`preparation.json` | 实际完成／未完成文档、首轮与最终达标数、逐轮统计、各段处理链和总体用量 |

中断后从最后一个完整轮次恢复，轮内已完成请求通过内容缓存复用；未完成标注不能进入 `diagnose`。终审开始后裁决冻结，避免续跑时把新裁决套在旧补题状态上。每轮的 `new_eligible_facts` 是本轮新增候选中被选为事实代表的数量，`net_eligible_change` 是整个事实池净变化，两者不混用。

报告分开给出：首轮达标数／冻结数、补题后达标数／冻结数、终审后达标数／冻结数；每轮参与文档、生成数、新增事实、净变化与额外用量。缓存命中不计作新生成；请求耗时之和与墙钟时间分别记录。

24 篇下，首轮最多 408 个生成／核验／全文审查请求，每个补题轮最多再增加 408 个；加上 192 次诊断回答，总上限为 1,824 个逻辑请求，不含 subagent 工作量。实际会在达标后提前停止。各请求输出上限为生成 8,192、局部核验 4,096、全文审查 32,768、诊断回答 2,048 tokens；全文审查上限覆盖累积候选增长。

## 5. 代码、配置与数据位置

以下相对路径均以项目根目录为基准。

| 内容 | 位置与职责 |
|---|---|
| 实现目录 | [fineweb_qa/](../../src/latent_working_memory/data_preparation/fineweb_qa/) |
| `download.py` | 固定 FineWeb 来源的无代理下载与续传；已有本地原文可直接复用 |
| `sources.py` | 旧来源排除、去重、划分、稳定选样和随机字符窗口 |
| `annotation.py` | Responses 请求、结构化输出、缓存、重试、用量和证据定位 |
| `assembly.py` | 事实代表选择、逐段配额、角色分池与八步使用安排 |
| `diagnostics.py` | 固定面板抽样、双条件评分、subagent 复核材料 |
| `finalization.py` | 应用裁决与同事实合并，重新计算候选池和配额 |
| `pipeline.py`、`__main__.py` | 阶段调度、三轮补题、恢复、统计与命令行入口 |
| 配置 | [fineweb-factqa-8192-doc2k.json](../../configs/data_preparation/fineweb-factqa-8192-doc2k.json) |
| Prompts | [fineweb_factqa/](../../configs/data_preparation/prompts/fineweb_factqa/)：`generate / verify / document_review / answer / subagent_review` |
| 测试 | [tests/v1/](../../tests/v1/) 中的七个 `test_fineweb_qa_*.py` |
| 原始来源 | `data/raw/HuggingFaceFW-fineweb/sample-10BT/`，15 个已下载的 Parquet |
| 新批次运行目录 | `artifacts/v3/fineweb-qa-random-topup_20260928/`；尚未执行 |
| 新批次正式输出 | `data/fineweb-factqa-random-topup-8192-doc2k_20260928/`；执行后生成 `train.jsonl` 与 `preparation.json` |
| 共用请求缓存 | `artifacts/data_preparation/fineweb-qa-requests/` |
| 旧批次记录与数据 | `artifacts/v3/fineweb-qa-pilot_20260928/`、`data/fineweb-factqa-8192-doc2k_20260928/`；保留 1 条轨迹、64 题及失败材料 |

正式 JSONL 每行一条完整轨迹，保存来源、原文字符范围、文本、八段位置、64 个 QA 和使用安排。证据由字符区间还原，不保存 token ID 副本。`preparation.json` 保存实际配置、来源统计、模型与软件信息、诊断和补题汇总。数据与执行产物保存在本机，Git 提交不包含原始 Parquet、请求缓存或生成数据。

## 6. 执行与验证

在根目录 `.venv` 运行。原始 FineWeb 已下载，当前可从 `prepare` 开始：

```bash
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa prepare \
  --config configs/data_preparation/fineweb-factqa-8192-doc2k.json
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa annotate \
  --config configs/data_preparation/fineweb-factqa-8192-doc2k.json --limit 4
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa annotate \
  --config configs/data_preparation/fineweb-factqa-8192-doc2k.json
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa diagnose \
  --config configs/data_preparation/fineweb-factqa-8192-doc2k.json
# 完成两份面板的模型复核与主 agent 裁决，保存 resolved-review.json 后：
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa finalize \
  --config configs/data_preparation/fineweb-factqa-8192-doc2k.json
```

`finalize` 可能调用标注服务进行剩余预算内的补题。反复运行已完成阶段会复用已保存结果；中断后沿用同一命令恢复。仅检查代码时使用：

```bash
.venv/bin/python -m pytest tests/v1/test_fineweb_qa_*.py -q
.venv/bin/python -m ruff check src/latent_working_memory/data_preparation/fineweb_qa tests/v1/test_fineweb_qa_*.py
.venv/bin/python -m ruff format --check src/latent_working_memory/data_preparation/fineweb_qa tests/v1/test_fineweb_qa_*.py
```

当前验证覆盖随机切分的长度、连续性、复现与多样性，按缺额补题、余量、跨轮去重、拒绝反馈、零题与预算耗尽、中断续跑、终审共用预算及固定诊断面板。离线通过不表示新流程的真实构造产率已得到验证。

验证时间：20260928 21:16:25 UTC+08:00。60 项 FineWeb QA 专项测试及 Ruff 检查、格式检查通过；用旧批次的 12 篇已保存的 FineWeb 窗口文本验证随机切分，长度、连续性与复现均通过，得到 12 种不同段长组合。该检查未生成新 QA，也未调用标注服务。
