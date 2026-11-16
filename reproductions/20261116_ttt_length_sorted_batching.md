# 20261116_LaCT 与 E²-TTT 的长度排序与混长组批

创建时间：20261116 10:47:03 Asia/Shanghai

最后修订时间：20261116 10:47:03 Asia/Shanghai

## 当前行为

两个正式评估入口均支持长度排序、相邻组批和动态左 padding，无需新增参数。`--batch-size` 表示每卡样本数上限，尾批保留实际数量。

- LaCT：每个 RULER 长度档位内，跨六项任务按实际 token 数稳定降序，相邻样本组批。保留 `--batch-size-by-length`，并在 padding 前逐条检查原 128-token 预算；生成从补齐后的批次宽度截取，结果分别记录真实 `input_tokens` 和物理 `batch_input_width`。
- E²-TTT：`LengthSortedHFLM` 保留生成参数和 BOS 状态分组，复用原生 HFLM 的长度排序、相邻组批、左 padding、停止处理、缓存和结果顺序还原。RULER 不截断；SQuAD／SWDE 仍按原规则左截断到 2,000 tokens，生成预算仍为 48。
- 原始模式保留 B1；本地优化模式沿用已验证的 padding 状态调度。TTT 仍逐样本执行，模型公式、更新时序、精度和样本隔离没有变化，未修改两者上游文件或本地 TTT 计算文件。

运行命令见 [LaCT README](lact/README.md#环境与运行命令) 和 [E²-TTT README](e2_ttt/README.md#环境与评估命令)。LaCT 续评只重排未完成样本，每个历史来源保留其组批方式；E²-TTT 沿用相同设置和策略续评，新策略的评估使用新目录，不将旧等长计划标为混长计划。

## 验证

服务器 `zcy-memory-v4.2`，物理 GPU 0，L40S 48GB。沿用各方法固定环境，实际运行代码为 `36ac4d0a`，当时服务器 Git 状态为 clean。下面仅验证评估入口，不改变正式 benchmark 的规模。

| 检查 | 范围 | 结果 |
|---|---|---|
| LaCT 组批与 padding | 真实 4K／8K 输入；跳过一条已有结果后共 9 条；各档位 batch 上限分别为 2／3 | 排序、上限、尾批、ID 唯一覆盖、mask 长度及真实 token 还原通过 |
| E²-TTT 原生 harness 的 CPU 检查 | 混长输入、生成参数／BOS 分组、截断、尾批、cache hook 和结果还原 | 通过 |
| LaCT Momentum／Muon PostNorm／Muon PreNorm | 每模型三条完整 8K RULER prompt，B3，最多生成 128 tokens | 三模型全部生成、计分及记录成功；同目录续评无重复记录或重新生成 |
| E²-TTT MLP／SwiGLU | 每模型 RULER 8K 两条、SQuAD completion 六条、SWDE 三条，B3，原生生成与评分 | 两模型各 11 条全部完成；续评各命中 11 条缓存，重新生成 0 条，无重复预测 |

LaCT 的三条 GPU 输入按 8,064、8,035、7,435 tokens 排列，物理宽度为 8,064；逐条 `input_tokens`、token IDs 解码、原生评分和生成长度均已核对。E²-TTT 小样本通过内部 worker 的第 0／500 个行分片选取，保留原始数据行、prompt、生成预算和计分；正式命令仍评估 README 中的全部样本。

这是长度组织和评估接入验证，不是新的 B1 数值一致性实验，也不是吞吐或显存上限测量。既有跨 batch 浮点差异及 Muon 限制见 [padding 推理](20261115_ttt_padding_inference.md)和[第二个 token 的数值定位](20261116_lact_muon_postnorm_second_token_trace.md)。此前等长 B16／B8 的资源测量不能直接视为本轮混长上限。

## 产物

本地与服务器记录位于 `artifacts/reproductions/sorted-batching_20261116/`：`plan.json`、`cpu_batching.json`、`lact_checks.json`、`e2_checks.json` 记录计划和检查结果；各模型目录保存原生计划、逐样本预测、评分、续评状态及缓存，未保存模型状态或 tensor dump。

E²-TTT 首次运行因服务器缺少 SQuAD completion／SWDE parquet 停在数据加载阶段；从官方 `hazyresearch/based-squad`、`hazyresearch/based-swde` 下载原文件到规范目录并同步后，复用原计划继续运行并通过上述检查。没有重写 benchmark 数据。
