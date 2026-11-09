# 20261109_LaCT 增量推理缓存

创建时间：20261109 15:20:00 Asia/Shanghai

最后修订时间：20261109 20:39:51 Asia/Shanghai

本轮目标是在保留迁入 LaCT 原始路径作对照的前提下，缓存历史计算，减少每个生成 token 重算完整前缀的开销。评估入口与命令见 [LaCT README](lact/README.md#本地-ruler-评估)，源码和本地代码的职责见 [UPSTREAM.md](lact/UPSTREAM.md#代码职责)。本记录不代表已完成正式 RULER 评估。

## 实现过程与边界

1. 在 `lact/incremental/` 单独实现缓存，没有修改迁入的 LaCT 模型、配置、TTT 更新函数或 flame trainer；`original` 继续调用迁入模型的原始完整前缀路径。
2. 每层保存近期 attention KV、总位置数、fast weights、PreNorm 累积状态、初始行范数、Muon 前 Momentum，以及待写 chunk 的 K/V、学习率和 Momentum 系数。RoPE 按总位置偏移，attention 保留原窗口。
3. 对齐完整 chunk 的切片布局，并将范数、读出与更新计算组织为按原公式执行的编译函数。保留 FP32 参数、BF16 autocast 和 checkpoint 的状态精度设置，未调整更新公式、chunk 大小或生成规则。

保持 chunk 内先读后写，满尾段在下一 token 到来时才提交。非空前缀长度为 \(L\)、chunk 大小 \(C=2048\) 时，提交次数为 \(\lfloor(L-1)/C\rfloor\)，待写长度为 \(L-C\lfloor(L-1)/C\rfloor\)。当前仅支持 **batch size 1、无 padding、非 fused TTT 推理**；独立生成从新缓存开始。

本轮计算实现的验证修订为 **`8df67e81`**，由当时的代码同步记录确认。历史运行尚未自动采集 Git 状态，原始 `plan.json` 保留当时内容，不补填未经采集的状态；后续运行来源记录见 [README](lact/README.md#运行来源与结果)。

## 验证记录

在 `zcy-memory` 物理 GPU 0（A800-80GB、PyTorch 2.6.0 / CUDA 12.4）检查 chunk 边界、真实样本的 prefill 及四步固定续接、独立 128-token greedy 生成和 A→B→A 缓存隔离。Momentum 的两条路径在同一进程重新运行，未复用旧原始生成或计时。三组 logits 有限性、所有层的缓存进度及缓存隔离均通过。

最终产物保留在 `artifacts/reproductions/lact/incremental-validation_20261109/` 的以下三个目录，表中链接指向各自汇总：

| checkpoint／产物目录 | 真实样本范围 | teacher-forced argmax 相同 | 完整生成 IDs 相同 | 最大 logits 绝对差 | 本轮判定 |
|---|---|---:|---:|---:|---|
| Momentum / PreNorm：[momentum-paired-final-full](../artifacts/reproductions/lact/incremental-validation_20261109/momentum-paired-final-full/summary.json) | 四长度，各两条 | 55 / 55 | 8 / 8 | 0.25 | 通过本轮检查 |
| Muon / PreNorm：[muon-prenorm-final](../artifacts/reproductions/lact/incremental-validation_20261109/muon-prenorm-final/summary.json) | 4K，两条 | 24 / 25 | 1 / 2 | 1.0546875 | 未通过 |
| Muon / PostNorm：[muon-postnorm-final](../artifacts/reproductions/lact/incremental-validation_20261109/muon-postnorm-final/summary.json) | 4K，两条 | 25 / 25 | 2 / 2 | 0.4375 | 通过本轮检查 |

Muon / PreNorm 在 4K 多查询样本的生成位置 86 首次分歧，另在合成续接至 4098 tokens 时出现 argmax 差异；分数相同不能替代生成 IDs 一致。Muon / PostNorm 的结论仅适用于本轮 4K 样本。

过程中，与旧冻结参考比较曾得到 Momentum 7 / 8 条生成一致。16K 多查询样本的原始路径在不同执行上下文中，于从 0 起计的位置 57 从 token 1 变为 13，原因尚未确定。后续默认编译设置下，该样本的 120 个 prefill 中间张量和最终 logits 逐位相同，续接相同 57 个 token 后两条路径均选择 13，但 logits 仍有最大 0.0625 的差异；另一轮 eager 诊断中，该位置仍出现 argmax 分歧。这些观测不能证明所有失败都只来自浮点误差，也不能推广为任意输入或执行上下文下等价。

## 耗时记录

以下为 Momentum 同进程配对验证的均值，每长度两条真实样本，每条均生成完整 128 tokens；预热单独记录，计时前后同步 CUDA。

| 长度档位 | 原始生成（秒／样本） | 增量生成（秒／样本） | 加速比 |
|---|---:|---:|---:|
| 4K | 15.36 | 9.39 | 1.64 |
| 8K | 24.75 | 9.51 | 2.60 |
| 16K | 48.53 | 9.62 | 5.05 |
| 32K | 97.08 | 9.93 | 9.78 |

按四长度、六任务、每任务每长度 500 条，共 12000 条外推，原始生成约 **154.77 小时**，增量约 **32.04 小时**，约加速 **4.83 倍**。这是小样本估算，不是正式完整评估耗时；不含加载、预热、分词和结果写入。峰值分配显存约 **6.60 GiB**，当前 B=1 的耗时不能按空余显存直接换算成批量吞吐。

默认评估仍为 `original`；正式 12000 条 benchmark 尚未启动，Muon / PreNorm 的一致性问题仍待解决。完整命令、环境和输出格式统一维护在 [README](lact/README.md)。
