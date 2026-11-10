# 20261110_E²-TTT 批量推理

创建时间：20261110 20:46:26 Asia/Shanghai

最后修订时间：20261110 20:46:26 Asia/Shanghai

## 实现与验证计划

保留 `e2_ttt/upstream/` 的官方实现，默认 `original`、batch size 1。本地 `optimized/` 单独处理等长批次，修正缓存解码的归一化参数展开，并按样本确定 Momentum 尾部裁剪长度；更新公式、chunk 大小和先读后写顺序沿用原实现。评估继续使用原 HFLM 的输入编码、截断、停止与计分逻辑。

本次使用 `zcy-memory` 的物理 GPU 0（A800 80GB）和独立 `ttt-e2_ttt` 环境，分别验证公开 MLP／SwiGLU 1.3B 模型。代码先经 GitHub 同步，再从服务器项目仓库运行。每次验证先保存计划、输入和代码来源。

验证覆盖：

- RULER 的 4K、16K 真实等长样本，原始 B1 对照优化 B1／2／4／8／16；记录完整生成 IDs、原生评分、吞吐和峰值显存。
- SQuAD completion／SWDE 的真实样本，保持原 2,000-token 输入上限、48-token 输出预算和停止规则。
- 在相同固定续串下跨越 TTT chunk 边界，对照原始状态和优化状态的结构、进度与数值。
- 固定 batch 形状及目标样本，仅置换或替换其他样本，检查目标行的 logits 与缓存是否保持独立。

跨 batch 形状允许浮点误差引起少量生成差异，独立报告 IDs、评分与数值差异；状态结构、更新进度与样本隔离必须通过。保存精简 JSON／JSONL，不保存 tensor dump 或 profiler trace。结果位于 `artifacts/reproductions/e2_ttt/batch-validation_20261110/`。

## 结果

待完成 GPU 验证后记录。
