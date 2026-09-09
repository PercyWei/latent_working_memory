# 20260909_v1 产物整理记录（14:52:14 UTC+08:00）

创建时间：20260909 14:52:14 UTC+08:00

最后修订时间：20260909 14:52:14 UTC+08:00

本次整理范围为本机项目的 `artifacts/v1/`。本地目录保存从服务器同步的实验报告、日志和辅助文件；服务器的运行目录、数据与 checkpoint 延续原实验路径。历史脚本、`invocation.json`、`provenance.json` 和评估记录中的路径表达当时的服务器环境。

## 1. 本地目录

| 分类 | 子目录 | 用途 |
|---|---|---|
| `data-preparation/` | [fineweb-pilot-20260908](../../artifacts/v1/data-preparation/fineweb-pilot-20260908/) | 512 篇文档的数据准备统计 |
| `validation/` | [pretrain-local-validation](../../artifacts/v1/validation/pretrain-local-validation/) | 初期本地测试报告 |
| `validation/` | [pretrain-server-validation](../../artifacts/v1/validation/pretrain-server-validation/) | 服务器测试、变长 batch 压力测试和 GPU 检查日志 |
| `validation/` | [pretrain-smoke-20260908](../../artifacts/v1/validation/pretrain-smoke-20260908/) | 两步训练与保存恢复验证 |
| `validation/` | [evaluation-smoke-20260909](../../artifacts/v1/validation/evaluation-smoke-20260909/) | 新增评估指标的真实模型验证 |
| `experiments/` | [pretrain-overfit-20260908](../../artifacts/v1/experiments/pretrain-overfit-20260908/) | 固定 16 样本的 100 步拟合实验 |
| `experiments/` | [pretrain-generalization-20260908](../../artifacts/v1/experiments/pretrain-generalization-20260908/) | 扩大数据实验的质量审查、调度脚本、节点汇总和模型选择 |
| `experiments/` | [pretrain-generalization-lr1e-4-20260908](../../artifacts/v1/experiments/pretrain-generalization-lr1e-4-20260908/) | 学习率 1e-4 的训练验证报告 |
| `experiments/` | [pretrain-generalization-lr3e-5-20260908](../../artifacts/v1/experiments/pretrain-generalization-lr3e-5-20260908/) | 学习率 3e-5 的训练验证报告 |
| `evaluations/` | [pretrain-generalization-test-20260909](../../artifacts/v1/evaluations/pretrain-generalization-test-20260909/) | 选定 checkpoint 的独立 test 结果 |
| `legacy/` | [fineweb-download-20260908](../../artifacts/v1/legacy/fineweb-download-20260908/) | 早期下载、环境准备及 pilot 启动脚本与日志 |
| `legacy/` | [fineweb-layout-swanlab-20260908](../../artifacts/v1/legacy/fineweb-layout-swanlab-20260908/) | 数据路径迁移、历史 SwanLab 导入及验收记录 |

原有 12 个目录分别移入上述分类，目录名保持原名。扩大数据实验的审查与调度文件作为同一实验记录共同保存。运行目录内继续保存逐步验证结果；`evaluations/` 用于独立评估任务。当前独立 test 文件沿用历史名称 `dev-step-002000.json` 和 `.jsonl`，实际划分由 `invocation.json` 中的 `--split test` 确定。

## 2. 清理内容

| 原位置（相对整理前的 `artifacts/v1/`） | 处理依据 | 文件数 |
|---|---|---:|
| `fineweb-layout-swanlab-20260908/backups/` | 11 份元数据按已记录的迁移规则转换后，与当前对应文件内容一致；迁移脚本和验收报告保留 | 11 |
| `pretrain-overfit-20260908/superseded-generation/` | 这些生成结果使用修复前的位置编号；修复后的固定面板与周期验证结果保留 | 14 |

上述两处共 25 个文件、约 1.24 MiB，已移入本机废纸篓 `/Users/percyw/.Trash/latent-working-memory-artifacts-20260909-145035/`，分别保存在 `metadata-backups/` 和 `superseded-generation/` 中，可按原位置恢复。所有其余产物在目录移动前后的文件内容一致，文档中的本地链接随目录更新。

历史实验脚本用于追溯已完成的实验，脚本中的绝对路径对应原服务器布局。当前数据准备入口为 `python -m latent_working_memory.data_preparation`。本次数据准备重构的验证产物继续位于 [artifacts/data-preparation-refactor-20260909](../../artifacts/data-preparation-refactor-20260909/)，重构前源码快照保存在 [artifacts/refactor-data-preparation-before-20260909](../../artifacts/refactor-data-preparation-before-20260909/)。
