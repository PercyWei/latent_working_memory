# 20261109_In-Place TTT 推理优化

创建时间：20261109 21:16:13 Asia/Shanghai

最后修订时间：20261109 21:42:19 Asia/Shanghai

## 执行计划

目标是在保留官方代码、TTT 更新与读出逻辑及完整评估范围的前提下，降低长输入推理的显存和耗时。原始路径作为对照，本地优化单独放入 `in_place_ttt/optimized/`；默认仍为 `original`，通过显式模式选择优化路径。沿用现有独立环境，不修改上游源码或 site-packages。

1. 用同一 checkpoint、同一进程的原始／优化路径分别记录 CPU 分词、完整生成、prefill 和 decode 耗时、显存峰值及 TTT 尾段的底层存储；另对 4K 输入做一次轻量 attention 内核诊断，避免大型 trace。
2. 先独立验证尾段缓存：仅复制实际非空尾段，避免视图保留整段前缀；保留原 KV、适配权重、样本隔离和立即提交完整 chunk 的行为。
3. 再验证分块 MLP：默认以 16384 tokens 组织 gate/up 计算，TTT 仍按 checkpoint 的 1024-token chunk 顺序读出和更新，保留每个 chunk 独立卷积和 projection 方向。比较生成结果、状态数值、显存与速度；无收益或未通过的设置不作为默认等价优化。
4. 根据 prefill/decode 实测占比，决定是否继续实施 KV 预分配；不提前引入批量化或改动 attention 范围。现有 SDPA 与可用 FA2 的比较仅在诊断结果支持时进行。

## 验证范围与判定

- `zcy-memory` 单张物理 GPU 0，显式 `CUDA_VISIBLE_DEVICES=0`；开始时该卡空闲，后续各实验顺序执行。
- 两个现有 Qwen3-4B checkpoint 分别验证。先做 4K／32K 预检查，再覆盖完整七个长度档位（4K–256K），每长度选择两个真实任务样本，生成预算保持 64，沿用 EOS、base prompt 和原生评分。
- 用 1022／1024／2046-token 合成前缀及四步固定续接覆盖 TTT chunk 边界。比较 logits 有限性、argmax、完整 greedy token IDs、TTT 尾段及适配权重差异，并验证跨样本缓存隔离。MLP 分块不要求浮点逐位相同，但分歧必须完整记录，不能以任务分数相同替代 token 一致。
- 比较只加载一份模型，先释放上一条路径的 KV 后再运行下一条，避免两套长 KV 占满显存。原始与优化结果不混入正式 benchmark 分数。
- 新运行在 plan／execution 中保存本地 Git commit、dirty 状态、推理模式、实际模型／缓存／MLP 实现、上游 revision 与运行配置。代码先发布 GitHub，服务器不能连接时通过 Gitee 同步，从服务器项目仓库执行。
- 产物统一放入 `artifacts/reproductions/in_place_ttt/inference-optimization_20261109/`，保留最终验证和必要计时摘要；本文件最终仅保留简要过程、结果与限制。

## 阶段结果

- 代码修订 `f94e2231` 的 TTT 4K／32K 预检查完成：四个真实样本的 tail／chunked 两组输入 IDs、完整生成 IDs、文本和原生分数均与原始路径一致；四种合成前缀共 60 次边界检查通过。尾段优化的 TTT 状态逐位相同，分块 MLP 在长合成前缀中存在状态浮点差异，argmax 仍相同。
- 原始 attention 的 4K prefill／单步 decode 均实际调用 PyTorch FlashAttention，BF16、32 query heads／8 KV heads、无显式 mask，并使用原生 GQA。
- 32K 原始／尾段／分块路径的平均耗时为 6.27／6.22／6.27 秒；峰值分配显存约为 15.70／14.82／14.46 GiB。此阶段主要体现显存收益，尚未观察到明显提速。

完整七长度的 TTT／baseline 配对验证正在顺序运行；未启动正式 9100 条评估。
