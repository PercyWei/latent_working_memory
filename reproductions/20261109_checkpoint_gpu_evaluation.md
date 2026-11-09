# 20261109_TTT checkpoint GPU 验证与评估

创建时间：20261109 11:46:00 Asia/Shanghai

最后修订时间：20261109 11:47:21 Asia/Shanghai

## 本轮计划

用户已授权使用 `zcy-memory` 当前空闲的物理 GPU 0。所有 GPU 命令显式设置 `CUDA_VISIBLE_DEVICES=0`，各方法顺序运行，沿用已安装的独立环境和原始模型代码；不新增测试。

先验证模型完整加载、原始 GPU 算子与真实样本生成，再启动正式评估。运行验证使用真实评估样本，单独保存输出和资源记录，不计入正式 benchmark 分数。正式评估不缩减已约定的任务、长度或样本数。

| 方法 | 首批 checkpoint | 正式评估范围 |
|---|---|---|
| LaCT | Momentum 760M | 六项 RULER，4K／8K／16K／32K，每任务每长度 500 条，共 12000 条 |
| E²-TTT | MLP 1.3B、SwiGLU 1.3B | 两项 RULER，512–16K，共 6000 条；完整 SQuAD completion 2984 条、SWDE 1111 条；每个 checkpoint 独立评估 |
| In-Place TTT | 第三方 Qwen3-4B 32K TTT 及 baseline | 十三项 RULER，4K–256K，每任务每长度 100 条，共 9100 条；两个 checkpoint 分别评估 |

正式运行使用各自 `eval.sh`，保留原有 prompt、生成预算、截断规则和评分实现。LaCT 保留原 `use_cache=False` 完整前缀生成；In-Place TTT 超过 32K 的输入标注为长度外推。公开 checkpoint 与原论文模型的差异沿用各方法 README 的说明。

运行验证保存于服务器项目 `artifacts/reproductions/gpu-validation_20261109/`；正式结果保存于 `artifacts/reproductions/<method>/eval/<checkpoint>_20261109/`。记录计划、逐样本输出、实际耗时、GPU 显存及成功或失败状态；只有完成完整范围才汇总正式指标。

## 执行状态

GPU 0 为 A800-SXM4-80GB，开始前显存占用 0 MiB、利用率 0%。模型和约定的数据目录均已就位，正在验证运行路径。代码改动经 GitHub 发布后同步服务器；服务器无法直连 GitHub 时使用 Gitee。
