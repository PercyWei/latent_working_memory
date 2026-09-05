# 20260905 C-DIC Table 1 复现计划（20260905 11:16:59 CST）

创建时间：20260905 11:16:59 CST（UTC+08:00）

最后修订时间：20260905 11:16:59 CST（UTC+08:00）

状态：MSC C-DIC seed 42 评估入口和两 episode pilot 已完成；全量运行前需收敛协议差异

## 目标

复现论文 Table 1 中 C-DIC 在 MSC 和 REALTALK 上的 PPL、BLEU、ROUGE-L、ROUGE-1 和 ROUGE-2。第一阶段只验证已训练 seed 42 checkpoint 的 MSC 行；确认协议和结果合理后，再扩展到完整 MSC、REALTALK 和必要 baseline。

论文 Table 1 的 C-DIC 结果：

| 数据集 | PPL | BLEU | R-L | R-1 | R-2 |
|---|---:|---:|---:|---:|---:|
| MSC | 8.431 | 0.023 | 0.160 | 0.205 | 0.037 |
| REALTALK | 9.789 | 0.035 | 0.134 | 0.176 | 0.030 |

## 已确认协议

- backbone：Llama-2-Chat-7B，generator frozen；
- compression tokens：128；
- retrieval threshold：`0.8`；
- recency decay：`0.05`；
- 训练：MSC 2 epochs、AdamW、learning rate `2e-4`；
- Table 1 使用 teacher forcing；
- MSC 涉及 sessions 2–5；
- 指标：PPL、BLEU-4、ROUGE-L/1/2；
- 论文 MSC inference batch size：8。

## 当前实现选择

- 读取 MSC session 5 test，每个 record 包含 sessions 1–5；
- session 1 仅使用 gold response 构建 memory，sessions 2–5 生成并计分；
- 每轮生成后仍使用 gold response 压缩和 write-back，保持 teacher-forced history；
- greedy decoding，最多生成 128 tokens；
- 当前按 episode 顺序、batch size 1 运行，支持 episode sharding 和按 turn 续跑；
- BLEU 使用 lowercase regex tokenization 的 corpus BLEU-4，无 smoothing；
- ROUGE 同时保存 macro F1 和 recall，Table 1 暂以 F1 对照。

## 未公开或待核对项

- Table 1 使用 validation、test 或跨 session split 聚合；
- 是否只计每个 session 或 conversation 的最后一个 turn；
- instruction memory 的具体文本和初始化方式；
- query、response 与 speaker 的序列化模板；
- generation length、EOS 外的停止规则；
- BLEU smoothing、tokenizer，以及 ROUGE stemming 和 aggregation；
- batch size 8 的具体 memory batching 实现。

上述差异足以显著改变绝对指标。未收敛前，结果标记为 protocol approximation，不作为论文复现成功或失败的最终判断。

## 执行阶段

1. 两 episode end-to-end pilot，验证 checkpoint、teacher forcing、generation、指标和断点续跑；
2. 对 prompt serialization、instruction initialization 和计分 turn 范围做小样本对照；
3. 选择最接近论文描述且无数据泄漏的协议；
4. 使用 GPU 0 和 1 分片运行 501 个 session 5 test episodes；
5. 合并 predictions，报告原始计数、token denominator、指标实现和论文差值；
6. 再实现 ICAE incremental、ICAE one-shot 等优先 baseline，最后扩展至 REALTALK。

## 停止条件

若 pilot 的 PPL 或生成质量与论文相差明显，先排查协议和实现，不直接消耗数小时运行全量数据。任何调整必须使用独立配置和 artifact 目录，不覆盖已有结果。

## 来源

- C-DIC 论文：<https://arxiv.org/abs/2606.12411v1>
- MSC 官方项目：<https://parl.ai/projects/msc/>
