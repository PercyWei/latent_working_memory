# 历史配置

创建时间：20260914 11:43:52 UTC+08:00
最后修订时间：20260929 16:03:54 UTC+08:00

统一保存早期试跑、冒烟验证与已由正式实验替代的配置。下列配置保留其原参数；历史产物中的原配置路径仍按当时执行记录解释。

| 配置 | 用途 |
|---|---|
| `pilot.json`、`pilot_a800.json` | 早期 pilot 与环境验证 |
| `pretrain_a800.json` | 早期预训练试跑 |
| `pretrain_boundary_comparison_a800.json` | 早期边界对比设置 |
| `generalization_lr1e4_a800.json`、`generalization_lr3e5_a800.json` | 早期学习率探索 |

FineWeb 八段小试配置保存在 [data_preparation/fineweb-factqa-8192-doc2k.json](data_preparation/fineweb-factqa-8192-doc2k.json)，对应代码提交 `68157cb`；其实际 prompts 与结果以旧运行目录快照为准，不能直接供当前六阶段管线续跑。现行来源池与批次配置见 [FineWeb 构造流程](../../src/latent_working_memory/data_preparation/fineweb_qa/README.md)。

当前正式实验使用各自的配置，不以本目录为默认配置来源。这里不保存实验产物或 checkpoint。
