# 20261020_v4 探索实验

创建时间：20261020 16:26:40 Asia/Shanghai（UTC+08:00）

最后修订时间：20261020 16:26:40 Asia/Shanghai（UTC+08:00）

每条探索路径独立存放实验定义、评估入口和展示代码，复用 v4 的模型、checkpoint 和公共 benchmark 协议。执行产物保存在 `artifacts/v4/exploration/`。

| 路径 | 目的 |
|---|---|
| [memory_failure](memory_failure/README.md) | 诊断 LoCoMo 从完整上下文约 42% F1 降至正常 TTT 约 4.8% 的原因；先比较完整上下文、关闭读取、固定初始记忆和正常 TTT |
