# 20261109_LaCT 增量推理缓存

创建时间：20261109 15:20:00 Asia/Shanghai

最后修订时间：20261109 15:20:00 Asia/Shanghai

## 实施与验证计划

用户要求补齐 LaCT 增量推理缓存，保留原始完整前缀实现作为对照，保持更新和读出逻辑，验证生成结果一致性，再实测速率并估算完整 12000 条 RULER 评估时间。

- 原始 `upstream/` 保持不变。在方法目录新增增量实现，并通过评估参数显式选择 `original` 或 `incremental`。
- 每层缓存 sliding-window KV、绝对位置、有效 fast weights、PreNorm 未归一化累积量与初始行范数、Muon 前 Momentum、尚未写入 chunk 的 K/V／学习率／Momentum 系数。
- 保持原先先读后更新和末 chunk 延迟写入时序：前缀长度为 \(L\) 时，已提交更新次数为 \(\lfloor(L-1)/2048\rfloor\)。不改变 chunk size、更新公式、归一化、精度或生成预算。
- 当前范围为非 fused TTT、batch size 1、无 padding 的公开 checkpoint 推理。原始路径仍可独立运行；不为未使用的 beam search、训练或 packed inputs 预置接口。
- 先用小输入覆盖 2048、4096 边界，比较 teacher-forced 最后位置 logits、增量 cache 进度和独立 greedy 生成结果；再使用真实 RULER 的 4K／8K／16K／32K，每长度至少两个样本，检查完整 128-token 预算下的 token IDs 一致性，并验证跨样本缓存隔离。
- 计时区分预热和稳定阶段，CUDA 同步后记录原始／增量耗时、生成长度、显存峰值；各长度按 3000 条外推完整评估。Momentum checkpoint 做完整验证，两个 Muon 变体追加代表性验证。
- 按用户已授权的 `zcy-memory` GPU 0 顺序执行。代码先发布 GitHub，服务器无法连接时经 Gitee 同步，从同步后的项目仓库运行。运行计划与结果保存到服务器 `artifacts/reproductions/lact/incremental-validation_20261109/`。

## 当前状态

正在实现。尚未声称原始与增量结果一致，也未启动正式全量 benchmark。
