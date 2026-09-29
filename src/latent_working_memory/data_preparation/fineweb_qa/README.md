# 20260929_FineWeb 事实 QA 构造接口

创建时间：20260929 16:03:54 UTC+08:00
最后修订时间：20260929 16:03:54 UTC+08:00

本目录实现来源池冻结、分批选样、6–10 段 QA 构造、诊断、复核及定稿。具体运行与结果保存于配置指定的 `artifacts_dir`，共享输出保存于 `dataset_dir`。

## 配置与阶段

- [来源池配置](../../../../configs/data_preparation/fineweb-factqa-source-pool.json)：来源排除、去重、扫描范围、split、字符窗口及每批各 split 份额。池只保存来源位置与切分计划。
- [批次配置](../../../../configs/data_preparation/fineweb-factqa-6to10-batch000.json)：池路径、批次编号、QA 和补题规则、六类请求的模型及 prompts、并发、缓存和输出位置。
- [prompts](../../../../configs/data_preparation/prompts/fineweb_factqa/)：`generate / verify / document_review / answer / review / adjudicate`。

在项目根目录依次运行；构造入口与模型请求阶段不是一一对应关系：

```bash
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa prepare-pool --config <来源池配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa prepare --config <批次配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa annotate --config <批次配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa diagnose --config <批次配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa review --config <批次配置.json>
.venv/bin/python -m latent_working_memory.data_preparation.fineweb_qa finalize --config <批次配置.json>
```

来源池统一冻结簇、train/dev/test 和窗口；不同批次按编号取不重叠切片。每篇随机 6–10 段，每段 3,072–4,096 字符，允许任意字符边界，完整证据必须位于对应段内。N 段轨迹需 4N 道任务题和 4N 道 gate 题；同一事实不跨角色重复。dev/test 的任务题角色为 evaluation，使用安排覆盖全部已读前缀。

首轮后最多补三轮，标注与终审共用预算。逐段配额不足才补题；耗尽仍不足记为 `quota_shortfall`。默认每篇每段抽一题，无全批抽检上限；先做有证据／仅问题诊断，再独立复核，存疑项另行裁决。终审补题新增的合格候选全部复核。

## 内容过滤与运行错误

服务明确返回 `status=incomplete` 且 `incomplete_details.reason=content_filter` 时，客户端抛出 `ContentFilteredError`，不解析或使用被截断的 QA，不重复请求同一被过滤内容。该事件归为**文档级构造失败**，原因统一为 `content_filtered`；批次继续处理其他冻结文档，不更换失败文档或重抽窗口。

这一规则适用于全部六类请求，以及终审补题。`failed-documents/doc-NNN.json` 保存文档索引、轨迹 ID、首次失败阶段、请求 ID、原始响应路径和时间。已完成轮次保留供审计，失败文档的所有 QA 均不写入正式数据。已发出的并发请求可能完成，但不再给该文档派发新请求。

原始请求和响应仍保存在共用缓存。续跑遇到旧过滤响应时，从缓存识别同一文档失败，不再次访问模型；后续续跑直接跳过已有失败记录。请求日志单独标记 `failure_reason`，保留服务实际返回的 usage；缓存复用不重复计网络用量。

其他异常仍按原规则处理：临时网络错误按既有预算重试；重试耗尽、其他未完成原因、非法 JSON／schema、模型契约或写盘错误会中止批次。不会将未知异常归为内容过滤，也不会以失败文档标记掩盖运行故障。

## 保存、统计与恢复

| 位置 | 内容与口径 |
|---|---|
| `config.json / prompts.json / selection.json` | 冻结配置、提示词、池身份与本批输入；只有 endpoint 可变 |
| `documents/ / annotate-summary.json` | 已完成轮次、候选及配额；`content_filtered_documents` 与未完成文档分列 |
| `failed-documents/` | 持久化的内容过滤失败记录，跨阶段及续跑复用 |
| `diagnostic-panel.json / diagnostic-answers.json / diagnostics.json` | 面板一经生成保持不变；诊断过滤造成的排除单列 `excluded_panel_qa_ids`，EM/F1 只以完成双条件诊断的保留面板计分 |
| `review-coverage.json / review-summary.json` | 逐文档覆盖；被过滤文档及排除题数单列，不能计为复核通过或普通拒绝 |
| `reviews/ / resolved-review.json` | 保留文档的复核和裁决；普通删题与合并仍重算配额 |
| `reviewed-documents/ / supplement-reviews/ / final-documents/` | 终审状态、补题复核、逐文档最终结果；终审过滤后即使原先配额达标也不入库 |
| `requests.jsonl` 与缓存 | 网络、缓存、过滤及运行失败记录；`content_filtered_calls` 为调用记录数，文档失败数按文档去重 |
| `train.jsonl / dev.jsonl / test.jsonl / preparation.json` | 最终轨迹；所有冻结文档保留在分母中，内容过滤与配额不足分开统计，失败明细附于 preparation |

恢复前确认同批旧进程已经停止，随后使用原配置和目录运行中断阶段，再执行后续阶段。修复代码后记录实际提交及恢复命令，保留此前日志。内容过滤记录无需人工改成 accepted，也无需删除缓存。若所有文档均失败，正式划分文件为空，准备信息仍记录零成功及完整失败分母。

模型复核不是人工正确率，诊断答对也不证明训练后的记忆收益。

## 代码与离线验证

`sources.py` 处理冻结池与选样；`assembly.py` 处理配额、事实隔离和使用安排；`annotation.py` 处理请求、缓存、错误分类与证据定位；`pipeline.py` 处理阶段调度和文档失败；`diagnostics.py / finalization.py` 处理评分与裁决。

```bash
.venv/bin/python -m pytest tests/v1/test_fineweb_qa_*.py -q
.venv/bin/python -m ruff check src/latent_working_memory/data_preparation/fineweb_qa tests/v1/test_fineweb_qa_*.py
```
