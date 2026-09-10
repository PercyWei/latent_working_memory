# 20260910_研究文档索引（16:33:29 UTC+08:00）

创建时间：20260910 11:36:22 UTC+08:00
最后修订时间：20260910 16:33:29 UTC+08:00

| 内容 | 文档 |
|---|---|
| 当前预训练数据策略、实现与构建记录 | [预训练数据构建](v1/20260910_pretraining_data_construction.md) |
| 方法结构与后续训练阶段 | [框架 v1](20260907_growing_latent_working_memory_framework_v1.md) |
| 数据与任务路线 | [真实数据训练路线](20260907_real_data_training_route_review.md) |
| 文献背景与创新边界 | [方向查重](streaming_mutable_latent_memory_prior_art_2026_09.md) |
| 机制诊断设计 | [whole-prefix／streaming gap](experiment_designs/20260903_whole_prefix_streaming_gap_experiment.md) |
| ICAE 复现 | [PwC 检查计划](experiment_designs/20260904_icae_v1_reproduction_validation_plan.md) |
| C-DIC 实验结果 | [训练](reproduction_results/20260904_c_dic_training_reproduction_record.md)、[初步评估](reproduction_results/20260905_c_dic_evaluation_reproduction_record.md)、[评估对齐](reproduction_results/20260906_c_dic_evaluation_alignment_record.md) |
| C-DIC 操作与实现 | [复现 README](../reproductions/cdic/README.md) |

`v1/` 仅维护当前预训练数据构建文档。旧数据构建说明已合并，旧方案正文可从 Git 历史查阅；实验数据与运行产物保持原位置。

本地 `artifacts/v1/` 按 `data-preparation/`、`validation/`、`experiments/`、`evaluations/` 和 `legacy/` 分类。历史实验正文中的服务器绝对路径对应当时运行环境，本地链接指向同步后的分类目录。
