# 20261110_E²-TTT 批量推理

创建时间：20261110 20:46:26 Asia/Shanghai

最后修订时间：20261110 21:56:52 Asia/Shanghai

## 实现边界

`original` 保留 `e2_ttt/upstream/` 官方路径，batch size 仅为 1；`optimized` 使用独立的本地实现。评估按原截断后的真实 token 长度、生成参数及 BOS 状态分组，不使用 padding，不复制尾批；编码、停止规则、输出预算与评分继续复用原 HFLM／task。

本地实现保留真实 batch 维度及各样本独立的 attention／TTT 状态；缓存解码先展开归一化参数，再构造 target；Momentum 尾部裁剪按每条样本的 heads 分别确定。更新公式、chunk 大小、更新前读出、归一化和门控顺序保持原规则。适配仅作用于当前模型实例，不修改全局注册或上游文件，不改变 `state_dict` 键与缓存 tuple 布局。支持范围及来源见 [UPSTREAM.md](e2_ttt/UPSTREAM.md)。

## 实际验证范围

环境为 `zcy-memory` 物理 GPU 0、A800 80GB、PyTorch 2.8.0＋CUDA 12.8，使用独立 `ttt-e2_ttt` 环境。代码通过 GitHub 同步后从服务器仓库运行。

- 4K：MLP／SwiGLU 1.3B，原始 B1 对照优化 B1／2／4／8。
- 16K：MLP／SwiGLU 1.3B，原始 B1 对照优化 B1／8。
- 每个 case 使用 RULER S-NIAH-1／2 各 8 条同长真实输入、SQuAD completion 与 SWDE 各 8 条，共 32 条。recall 保留 2000-token 输入上限、48-token 输出预算与原停止规则；RULER 为 128-token 输出预算。
- 固定续串覆盖 511／512／513 及 1023／1024／1025 两处 chunk 边界；固定 B4 和目标行，置换或替换其余样本检查隔离。

## RULER 耗时与显存

下表每行比较同一组 16 条 RULER 输入。generation 为模型生成调用合计；harness wall 另含输入编码、等长分组／排序及输出文本处理，均排除模型载入、选样、预热与评分。倍数以原始 B1 为参照；显存为两项任务中的最大 CUDA allocated，非 reserved。

| 模型／长度 | 原始 B1 generation／harness（秒） | 优化 B8 generation／harness（秒） | 加速倍数 generation／harness | 峰值 allocated B1 → B8（GiB） |
|---|---:|---:|---:|---:|
| MLP 4K | 113.17／113.40 | 16.76／17.07 | 6.75×／6.64× | 4.73 → 13.19 |
| SwiGLU 4K | 113.77／114.01 | 17.36／17.67 | 6.55×／6.45× | 4.92 → 14.30 |
| MLP 16K | 135.40／136.40 | 26.00／27.53 | 5.21×／4.95× | 9.96 → 38.09 |
| SwiGLU 16K | 139.42／140.41 | 27.65／29.17 | 5.04×／4.81× | 10.11 → 38.94 |

这是小样本实测，不是完整 benchmark 耗时估计。RULER 实际为 B8；recall 的 B8 上限在这组输入中形成 B2＋B6／B6＋B2，不能按所有输入均达到 B8 外推。

## 生成、评分与状态检查

四个 case 的优化 B1 均与原始 B1 在全部 32 条输入上生成 IDs 和评分相同。IDs 比较截取至每行按原规则首次出现的 EOS 或停止串，包含终止 token；未经截取的 `raw_generated_ids` 另存。下表报告优化 B8 相对原始 B1 的“IDs 不同条数／原生评分不同条数”：

| 模型／长度 | RULER（16 条） | SQuAD completion（8 条） | SWDE（8 条） |
|---|---:|---:|---:|
| MLP 4K | 3／0 | 2／1 | 0／0 |
| SwiGLU 4K | 2／0 | 3／0 | 1／0 |
| MLP 16K | 2／1 | 2／1 | 0／0 |
| SwiGLU 16K | 4／1 | 3／0 | 1／0 |

MLP 的 SQuAD `/708` 在两轮均由 1 分变为 0 分，16K RULER S-NIAH-2 的 `/20` 同样由 1 变为 0；SwiGLU 16K 的 S-NIAH-2 `/3` 则由 0 变为 1。这里的分数变化仅描述这些样本，不构成完整 benchmark 质量结论。

每个 case 的 8／8 组结构与缓存进度检查、4／4 组固定 B4 隔离检查通过。隔离采用 `atol=rtol=0`：MLP 每轮 3744 个缓存张量及 12 次 logits 对照、SwiGLU 4320 个缓存张量及 12 次 logits 对照均逐项相等。跨 batch 的状态 allclose 并未全部通过；不能据此声称不同 batch 的数值逐项一致。

## 差异样本诊断

MLP SQuAD `/708` 的输入长 172 tokens；虽设置 B8 上限，发生差异的实际批次为 B6。首次生成分歧位于从 0 起计的位置 19，固定相同的此前 19 个 token 后，24 层缓存进度均为 191，尚未到达 TTT chunk 512，没有发生记忆写入。

原始 B1 与优化 B1 的记录完全相同：`the`（272）与 `Thomas`（6083）的 logits 均为 8.6875，argmax 选择 272。实际 B6 中二者为 8.6875 与 8.75，argmax 改为 6083。第一层 attention norm 输出仍相同，Q 投影已出现最大绝对差 0.03125；prefill logits 与分歧步 logits 的最大绝对差分别为 0.09375、0.125。

该例已观测到 BF16 批量计算在 Q 投影处产生差异并改变并列候选的选择，不能归因于 TTT chunk 写入。

MLP 16K S-NIAH-2 `/20` 的首个分歧位于位置 2：原始 B1／优化 B1 中 `3`（28770）和 `.`（28723）的 logits 为 11.1875／11.0625；B8 中均变为 11.0625，argmax 改选 `.`。第一层 attention norm 和 Q 投影记录相同，但 prefill／分歧步 logits 最大绝对差分别为 0.5546875／0.6015625，24 层缓存进度一致。

后一例确认了临界 logit 的选择变化，尚未定位最早产生差异的算子；不能把所有差异都归因于 Q 投影，或将两例推广为完整 benchmark 质量结论。

## 正式评估入口检查

两个模型均通过 `eval.sh` 以 `optimized`、batch 上限 8 运行划分为 500 份后的第 0 号分片：RULER 4K 两项各 1 条、SQuAD 6 条、SWDE 3 条，共 11 条。首次执行和 `--resume` 均完成；各自 `entry_check.json` 的 `metrics_equal`、`predictions_equal` 均为 `true`，确认续评汇总和逐题结果保持一致。

这是正式入口的少量分片及缓存续评检查，不是每模型 10,095 条 benchmark 的正式结果，也未验证实际中途杀进程后的恢复。

## 来源与产物

结果根目录为 `artifacts/reproductions/e2_ttt/batch-validation_20261110/`，保存 JSON／JSONL 记录及入口续评使用的 SQLite 响应缓存，无 tensor dump 或 profiler trace：

- `mlp-4k-b1-2-4-8/`、`swiglu-4k-b1-2-4-8/`：clean `09791ca3`。
- `mlp-16k-b1-8/`、`swiglu-16k-b1-8/`、`mlp-4k-b8-squad-708-diagnosis/`、`mlp-16k-b8-ruler-20-diagnosis/`、`entry-mlp/`、`entry-swiglu/`：clean `194fcf95`。

验证入口为 [validate_batch.py](e2_ttt/scripts/validate_batch.py)，单题诊断入口为 [diagnose_batch_difference.py](e2_ttt/scripts/diagnose_batch_difference.py)。各目录的 `plan.json` 保存输入、设置与来源，`summary.json` 保存汇总；验证目录另有逐样本预测与状态／计时记录。
