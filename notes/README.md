# 20260912_研究文档索引（15:42:35 UTC+08:00）

创建时间：20260910 11:36:22 UTC+08:00
最后修订时间：20260912 15:42:35 UTC+08:00

| 内容 | 文档 |
|---|---|
| 当前预训练数据策略、实现与构建记录 | [预训练数据构建](v1/20260910_pretraining_data_construction.md) |
| 预训练数据类型对比：训练、完整评估与产物路径 | [实验记录](v1/20260911_pretraining_boundary_comparison_restart.md) |
| 方法结构与后续训练阶段 | [框架 v1](20260907_growing_latent_working_memory_framework_v1.md) |
| 数据与任务路线 | [真实数据训练路线](20260907_real_data_training_route_review.md) |
| 文献背景与创新边界 | [方向查重](streaming_mutable_latent_memory_prior_art_2026_09.md) |
| 机制诊断设计 | [whole-prefix／streaming gap](experiment_designs/20260903_whole_prefix_streaming_gap_experiment.md) |
| ICAE 复现 | [PwC 检查计划](experiment_designs/20260904_icae_v1_reproduction_validation_plan.md) |
| C-DIC 实验结果 | [训练](reproduction_results/20260904_c_dic_training_reproduction_record.md)、[初步评估](reproduction_results/20260905_c_dic_evaluation_reproduction_record.md)、[评估对齐](reproduction_results/20260906_c_dic_evaluation_alignment_record.md) |
| C-DIC 操作与实现 | [复现 README](../reproductions/cdic/README.md) |

`v1/` 保存预训练数据构建、实验计划与运行记录。旧数据构建说明已合并，旧方案正文可从 Git 历史查阅。

实验产物按 `artifacts/v1/<实验系列>/{train,eval,compare,plan}/` 组织。当前系列位于 `artifacts/v1/pretrain-data-comparison-2048-20260911/`；数据准备、工程验证和历史辅助文件分别保存在 `data-preparation/`、`validation/` 和 `legacy/`。历史日志保留执行时的原始路径。
