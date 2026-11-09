# 20261109_LaCT 增量推理缓存

创建时间：20261109 15:20:00 Asia/Shanghai

最后修订时间：20261109 17:05:13 Asia/Shanghai

## 实现与原逻辑保持

本地增量实现位于 `lact/incremental/`，与未改动的官方 `lact/upstream/` 并存，加载同一 checkpoint。原始路径逐 token 重算完整前缀；增量路径复用历史计算。结论按 checkpoint 和验证范围报告：Momentum 通过本轮四长度代表性检查，Muon / PostNorm 通过本轮 4K 检查，Muon / PreNorm 尚未通过。评估默认仍为 `original`。

每层缓存包括：

- sliding-window attention 的近期 KV 与已处理位置数，用于保持原位置编码和窗口范围。
- 有效 fast weights，以及 PreNorm 的未归一化累积状态 `w*_main` 和初始行范数。
- 上一次 **Muon 变换前**的 Momentum 更新量；当前更新先合入 Momentum，再执行 Muon。
- 尚未提交 chunk 的 K/V、逐位置学习率和 Momentum 系数，以及已处理 token 数、更新次数和待写长度。

chunk 内先使用已有状态读出；完整 chunk 在下一 token 到来时才写入。对于非空前缀长度 \(L\) 和 \(C=2048\)：

\[
N_{\mathrm{updates}}=\left\lfloor\frac{L-1}{2048}\right\rfloor,
\qquad
L_{\mathrm{pending}}=L-2048N_{\mathrm{updates}}.
\]

因此，\(L=2048\) 时尚未提交更新，\(L=2049\) 时提交一次；不会提前写入当前尾段。PreNorm 保留未归一化的累积状态，归一化后的权重用于读出；PostNorm 则更新后按原规则归一化。参数以 FP32 加载，计算使用 BF16 autocast，状态精度遵循 checkpoint 的 `fp32_states`，不改 chunk 大小、更新公式或生成预算。

当前接口仅支持 **batch size 1、无 padding、非 fused TTT 的推理**。每次独立生成从新缓存开始。

## 运行入口

在服务器项目根目录激活 LaCT 原生环境后，显式选择原始路径：

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/lact/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --inference-mode original \
  --output-dir artifacts/reproductions/lact/eval/momentum-rope-prenorm-ruler-original
```

增量路径使用独立输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/lact/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --inference-mode incremental \
  --output-dir artifacts/reproductions/lact/eval/momentum-rope-prenorm-ruler-incremental
```

两者使用同一份六任务 RULER，4K／8K／16K／32K，每任务每长度 500 条，共 12000 条。以上为正式评估入口，本轮验证输出单独保存在服务器 `artifacts/reproductions/lact/incremental-validation_20261109/`，不计入正式 benchmark。

## 验证标准

本轮使用 `lact/scripts/validate_incremental.py` 检查 Momentum checkpoint：

1. 合成前缀长 2046、2048、4094 tokens，各续接四个固定 token，覆盖两处 chunk 边界；真实样本为四个长度各两种任务，共八条。每例比较 prefill 及随后四步的最后位置 logits，共 55 次，记录最大绝对／相对差、有限性和 argmax 是否一致。
2. 每步检查所有层的 attention 与 TTT 已处理长度均为 \(L\)，提交次数及待写长度满足上述公式。
3. 两条路径独立进行 greedy 生成，预算 128 tokens，保留原 EOS 停止规则，严格比较完整 token IDs。任务得分或解码文本相同不能替代此项。
4. 处理其他样本后再次生成首样本，检查缓存隔离。全部条件满足才判定本轮通过；失败保存记录并以非零状态退出。

`--reference-output` 可以复用该脚本此前完整保存的原始生成及计时。脚本先核对 checkpoint、配置、数据、运行环境和逐样本身份；缺失或不匹配直接报错。合成与真实样本的双路径 teacher-forced 检查、增量 warmup、完整生成和缓存隔离仍重新执行，复用来源写入结果。

## 当前验证结果

版本 **`8df67e81`** 在 `zcy-memory` 的物理 GPU 0（A800-80GB、PyTorch 2.6.0 / CUDA 12.4）完成以下验证。Momentum 当前基准为 **`momentum-paired-final-full`**：两条路径在同一进程重新运行，未复用旧原始生成或计时。所有真实样本的两条路径均生成完整 128 tokens；各轮 logits 有限性、所有层的缓存进度与跨样本隔离均通过。

| checkpoint | 真实样本范围 | teacher-forced argmax 相同 | 完整生成 token IDs 相同 | 本轮判定 |
|---|---|---:|---:|---|
| Momentum / PreNorm | 四个长度，各两条 | 55 / 55 | 8 / 8 | 通过本轮代表性检查 |
| Muon / PreNorm | 4K，两条 | 24 / 25 | 1 / 2 | 未通过 |
| Muon / PostNorm | 4K，两条 | 25 / 25 | 2 / 2 | 通过本轮代表性检查 |

Muon / PreNorm 的 `niah_multiquery` 在生成位置 86 首次分歧，两者分数均为 75%；另在合成前缀续接至 4098 tokens 时出现一次 argmax 差异。三种 checkpoint 的最大 logits 绝对差分别为 0.25、1.0546875、0.4375。Muon / PostNorm 的通过结论仅适用于本轮 4K 检查。

此前 `momentum-final-full` 与冻结的旧 `momentum-full` 原始结果比较，仅有 7 / 8 完整生成相同；这一失败记录仍保留，不能与本轮同进程对照混合。逐项比较旧、新两轮的**原始路径**结果，16K `niah_multiquery` 本身也在从 0 起计的位置 57 从 token 1 变为 13，其余七条完整 128-token IDs 相同。变化发生在不同执行上下文中，具体原因尚未确定，不能将旧参考对比的失败一概归为无害的浮点差异。

因此，当前结果支持 Momentum 与 Muon / PostNorm 在各自本轮样本范围内通过检查，不能推广为所有 checkpoint、输入或执行上下文下均等价。默认保持 `original`，未启动正式 12000 条 benchmark。当前结果产物为 `momentum-paired-final-full`、`muon-prenorm-final`、`muon-postnorm-final`；旧失败结果与诊断日志也保留在同一产物目录。

## 数值诊断

当前版本的 `momentum-final-norm-diagnose` 在默认编译设置下比较同一条 15614-token 前缀：24 层各五处、共 120 个中间张量以及最终 logits 均逐位相同。按相同的 57 个生成 token 继续计算时，两条路径的 argmax 均为 token 13；先前完整生成记录的原始 token 则为 1。因此，不能仅凭旧参考中的分歧判断缓存逻辑有误，仍需区分执行上下文及浮点计算的影响。

单独的 eager 诊断使用版本 `693e90fa`，记录为 `momentum-final-eager-diagnose`，两条路径同时关闭编译。在同一条 **15614-token** 完整前缀上，24 层各五处、共 120 个中间张量以及最终 logits 均逐位相同。随后按相同的 57 个生成 token 增量读取，与原始完整前缀计算相比，logits 最大绝对差为 **0.0625**：原始 argmax 为 token 13，前两名差 0.0625；增量路径中 token 1 与 13 均为 13.75，argmax 选择较小索引 1。24 层缓存进度均正确。

这说明该样本在 eager 设置下没有观测到 prefill 差异，而解码矩阵形状不同仍可能产生影响 argmax 的数值差异。该诊断支持继续核查浮点计算路径，不能证明所有分歧都仅来自数值误差，也不替代默认执行设置的一致性验证。

## 耗时记录

预热与稳定阶段分别记录，两条路径在计时前后同步 CUDA，并保存实际生成长度及显存峰值。完整评估按各长度 3000 条计算：

\[
\widehat T=\sum_{\ell\in\{4096,8192,16384,32768\}}3000\,\bar t_\ell.
\]

以下为 `momentum-paired-final-full`（`8df67e81`）的分长度均值，每长度两条真实样本、每条完整生成 128 tokens。两条路径的耗时均在本轮同一进程重新测量。

| 输入长度档位 | 原始生成（秒／样本） | 增量生成（秒／样本） | 加速比 |
|---|---:|---:|---:|
| 4K | 15.36 | 9.39 | 1.64 |
| 8K | 24.75 | 9.51 | 2.60 |
| 16K | 48.53 | 9.62 | 5.05 |
| 32K | 97.08 | 9.93 | 9.78 |

按 12000 条外推，原始生成约 **154.77 小时**，增量生成约 **32.04 小时**，估计加速 **4.83 倍**，仍超过半天的目标。本轮生成峰值分配显存为 **7,090,698,240 bytes（约 6.60 GiB）**。

两个 Muon checkpoint 的 4K 均值分别为 PreNorm **19.03 → 8.09 秒**、PostNorm **18.87 → 7.95 秒**；没有据此推断它们的完整四长度评估耗时。

这些是代表性样本上的计时，**不是正式完整评估耗时，也不能据此声明任意输入上的生成等价**。外推不包含模型加载、预热、分词和结果写入，实际耗时会受样本及运行波动影响。
