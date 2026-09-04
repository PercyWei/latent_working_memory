# C-DIC 假设清单（20260904 17:08:27 CST）

创建时间：20260904 16:19:08 CST（UTC+08:00）

最后修订时间：20260904 21:01:28 CST（UTC+08:00）

状态：R1 核心机制与 inference adapter 已通过 GPU smoke test

论文明确了 C-DIC 的总体算法，但部分实现细节未公开。首次 MSC pilot 前需固定主实验选择；后续修改必须记录时间，并使用独立结果标签。

| 项目 | 论文依据 | 当前选择 | 状态 |
|---|---|---|---|
| 阈值等号 | 公式使用 `> tau`；Algorithm 1 使用 `>= tau`，write-back 仅在得分 `< tau` 时插入 | 等于阈值时视为 on-topic，执行 retrieve 和 replace | 依据 Algorithm 1 固定 |
| Pooling `psi` | 论文仅举例 mean 或 CLS | 首版 ICAE adapter 使用 `mean` | 需要敏感性实验 |
| Support 顺序 | Retrieved states 表示为无序集合 | 按衰减后得分降序排列，同分时保持 memory 写入顺序 | 实现假设 |
| 相似度同分 | 未说明 | 选择 memory 中更早的 state 作为 argmax | 实现假设 |
| Recency 重置 | `Delta t` 表示距上次 retrieval 的 turn 数 | 所有 supports 均重置，包括 off-topic fallback；新写入 state 从当前 turn 开始计时 | 实现假设 |
| State identity | 论文描述可修订的 thread states | Replace 时保留 `thread_id`，分配新 `state_id` 并增加 revision | 审计设计 |
| Query representation | 使用 `psi(f_comp(q_t, C))` | 使用相同 compression-token bank 压缩 query | 工程链路已通过 GPU smoke test |
| Instruction 初始化 | Figure 3 表示先压缩 instruction；Algorithm 1 从空 memory 开始 | 当前 core engine 从空 memory 开始；后续将 instruction seeding 作为显式选项 | 尚未确定 |
| 对话序列化 | 未说明 | 初始 smoke 使用 `<s>[INST] query [/INST] response </s>`，并保持可配置 | 实现假设，需在 MSC 验证 |
| Generator prompt 边界 | 论文写为 `[R_t; Emb(q_t)]`；ICAE v1 inference 使用 FT markers | 初始 adapter 使用 ICAE FT markers 包围 query | 实现假设，需要消融 |
| EOS 处理 | C-DIC 未说明；ICAE v1 定义 `model.eos_id=1` | 使用 `model.eos_id`，并拒绝扩展 ICAE token IDs | 已由 ICAE 单样本 smoke test 验证 |
| Retrieval budget | 默认方法不限制 memory bank；附录测试 `B_ret=6` | 默认 `None`，可选 bounded retrieval | 论文默认设置 |
| 训练 response | 训练公式使用 gold `r_t`；inference 使用生成的 `r_hat_t` | 训练使用 gold response，closed-loop inference 使用 generated response | 论文约束 |
| LoRA 细节 | 仅说明训练 compressor 与 compression tokens | 加载前从 ICAE checkpoint 推断 rank 和参数形状 | 已确认 rank 为 128，strict load 通过 |

不得通过 `strict=False`、静默截断或隐式 fallback 掩盖配置不一致。
