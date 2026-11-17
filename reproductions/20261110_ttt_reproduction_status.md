# 20261110_TTT 第一阶段复现状态

创建时间：20261110 11:30:45 Asia/Shanghai

最后修订时间：20261117 01:10:58 Asia/Shanghai

## 当前状态

LaCT、E²-TTT、In-Place TTT 的源码迁入、独立环境、数据接入及此前 checkpoint 运行检查已完成，均保留各自 `eval.sh`。三个方法默认使用 `original`，保留上游 B1 路径；显式选择 `optimized` 时按长度排序后相邻组批，左 padding 到批内最长输入，单样本、等长和混长输入使用同一状态路径。In-Place TTT 的本轮批量实现仅完成本地代码审查，尚未测试。

| 方法 | 正式评估范围 | 当前进度 |
|---|---|---|
| LaCT 760M | 六项 RULER，4K／8K／16K／32K，每任务每长度 500 条，共 12,000 条 | Momentum 已完成；两个 Muon 按需评估 |
| E²-TTT MLP／SwiGLU 1.3B | 每模型 RULER 6,000 条、SQuAD completion 2,984 条、SWDE 1,111 条，共 10,095 条 | 已完成运行检查，完整评估尚未记录 |
| In-Place TTT 32K TTT／baseline | 每模型十三项 RULER，4K–256K 七长度，每任务每长度 100 条，共 9,100 条 | 此前 B1 实现已通过运行检查；混长 batch 实现尚未测试，完整评估尚未记录 |

LaCT Momentum 的已有完整运行六任务、四长度宏平均为 **39.46%**，累计执行约 3.47 小时，[离线报告](lact/README.md#查看结果)保留逐任务分数和论文比较。公开权重实际为两头，是作者重训产物；它不是原论文表格对应的同一次训练结果。两个 Muon 公开权重实际为四头。

LaCT 使用作者公开重训权重，E²-TTT 使用作者官方权重；In-Place TTT 使用第三方 ProLong 32K／6,000 steps 权重，超过 32K 的结果属于长度外推。当前 checkpoint 评估不需要重新训练。

运行命令与参数以各方法 README 为准，数据准备见[数据准备](20261108_ttt_data_preparation.md)，依赖见[评估环境](20261109_environment_setup.md)。每卡独立加载模型、评估互斥样本并按原规则汇总；本轮没有重新测量混长吞吐、batch 上限或多卡性能。

20261116 在 `zcy-memory-v4.2` 的物理 GPU 0（L40S 48GB）复核当时的 `optimized` 模式：LaCT 三个模型各三条完整 8K 输入，E²-TTT 两个模型各 11 条、覆盖三个 benchmark，生成、原生计分与同目录续评均通过。实际代码为 `662b1dc1`，服务器 Git 状态为 clean；临时验证产物已清理。

20261116 以 CPU 验证 `849a19ad` 的共同父目录布局：两种方法的默认命名、显式 ID 续评及目录碰撞检查通过；E²-TTT 在原生环境中验证了子集续评不修改未选 benchmark、各 benchmark 的部分生成缓存独立恢复，以及整个运行的完成状态。原生评分调用与模型代码保持不变；先前的全局 ID、重复 prompt 独立计分和跨 GPU／batch 续评检查结论仍适用。本次未加载 checkpoint 或使用 GPU，临时验证产物已清理。

## 实现与验证边界

### LaCT

`lact/optimized/` 缓存窗口 KV、fast weights、范数、Momentum 和待提交 chunk，保留官方非 fused TTT 公式、先读后写与延迟提交规则。真实前缀长为 $L>0$、chunk 为 $C$ 时，提交次数仍为 $\lfloor(L-1)/C\rfloor$。padding 不计入位置、KV 或更新进度，模型参数保持 float32，计算使用 BF16 autocast。

三个 checkpoint 均已完成混长小样本生成、原生计分和续评检查；chunk 边界、padding 忽略及固定 batch 形状下的样本隔离已有验证。不同 batch 形状仍可能改变浮点舍入、生成和逐题评分。Muon 与官方完整前缀路径的对照覆盖有限，不能声称任意输入逐 token 相同，也不能把已定位的 batch 数值差异归为 padding 状态错误。

20261116 修复 decode 的动态布尔索引和逐样本 TTT 调用：HF 内部新增全有效 mask 使用可校验的缓存元数据，外部 mask 仍按实际内容解析；TTT 保持独立权重、dtype、Momentum 和真实进度，分组执行读出与 chunk 提交，使用固定 pending 缓冲。原读出／更新函数与上游源码未改。实现与验证代码为 `88db3efb`、`09b57930`、`b64fb853`。

在同一张 L40S 48GB 上，Muon PostNorm、相同 16 条真实 4K 输入、greedy 最多生成 128 tokens，预热后各测两次：

| 实现 | 每批耗时 | 吞吐 | 峰值 allocated | 峰值 reserved |
|---|---:|---:|---:|---:|
| 修复前 | 44.33／44.42 秒 | 0.361／0.360 samples/s | 22.76 GiB | 25.11 GiB |
| 修复后 | 10.86／10.96 秒 | 1.473／1.460 samples/s | 22.75 GiB | 24.01 GiB |

该小样本对照提速约 4.1 倍；16 条逐题评分全部一致，13 条完整生成 token 序列一致，另外 3 条存在生成差异。三个公开 checkpoint 在 2048／4096 chunk 边界附近的 18 条混长短生成均与旧实现一致。以上不是完整 benchmark 的分数或耗时。

旧 `TTTState` 与新状态的真实 CUDA 对照覆盖四种 Momentum／Muon、PreNorm／PostNorm 组合，共 7,776 项检查，使用 `atol=0.001`、`rtol=0.01`；读写进度、冻结行、padding 与样本隔离另作严格检查。分片与整段调用的 Muon 浮点差异可由旧实现复现，新旧对应调用及差异对照通过。原地修改 mask、新 mask、全零 decode 行和 inference tensor 检查通过。实际 decode 的 CPU 操作轨迹中 `aten::nonzero`、`aten::_local_scalar_dense` 均为 0；B16 每步 TTT 读出从逐样本 384 次减少为 24 次，混长异步提交也按原边界执行。临时脚本、trace 与输出不作为正式评估产物。


### E²-TTT

`e2_ttt/optimized/` 保留官方 SWA、closed-form Momentum、TTT chunk 与读写顺序，支持公开 MLP／SwiGLU checkpoint 的非 fused、无 short convolution 配置。各样本独立维护真实位置、有效 KV 与 TTT tuple；保留固定 FLA 版本的原生 RoPE 行为，不在接入中修正上游时序。

两个 checkpoint 均已完成 RULER、SQuAD completion、SWDE 的小样本生成、原生计分和缓存续评；状态结构、真实更新进度、padding 忽略及样本隔离已有验证。跨 batch 不保证 logits、生成 IDs 或逐题评分相同。旧等长测量不能视为当前混长实现的速度或显存上限。

### In-Place TTT

`in_place_ttt/optimized/` 支持 TTT 和 baseline 的单样本、等长和混长左 padding 推理。各样本独立维护真实位置、完整有效 attention KV 与 TTT 状态，按真实 chunk 进度分组执行；padding 不参与读写或更新。保留原始 TTT 公式、1,024-token chunk 的先读后更新规则和逐 chunk 的卷积零填充边界，满 chunk 在当次读出后更新。默认使用 `original`；`--mlp-chunk-size` 默认 `0`，正值启用额外 MLP 分块，TTT 模型要求其为 TTT chunk 的整数倍。

单卡和多卡均经 `eval.sh` 的共同启动器运行，每卡一个进程；数据路径在代码中固定。`--batch-size` 默认 `1`，`--batch-size-by-length` 覆盖指定长度；按真实 token 数降序排列后相邻组批。各轮执行按全局样本 ID 分配未完成样本，保留原 EOS、64-token 生成预算和 OpenCompass 评分协议，续评允许调整 GPU 数量、顺序与 batch 设置。

attention 沿用原生 SDPA，按真实 query／key 长度分组；TTT 按读出长度、是否更新及适配权重状态分组，不要求混长样本同步提交。HF 原停止条件保留，结束行冻结状态，生成 IDs 按各行首次结束位置截取。批量路径支持 `default`、`linear`、`yarn`、`llama3` RoPE，不支持 `dynamic`、`longrope`。

20261117 的混长 batch、批量 TTT 及评估接入仅完成本地代码审查，未运行测试、GPU 验证或性能测量。当前没有本轮 batch 上限、吞吐或全量耗时的实测结论。

以下一致性与资源数据来自此前 B1 实现，不能视为本轮混长 batch 的验证或显存测量：

- 两个 checkpoint 各覆盖七长度、每长度两条，共 14 条真实样本；原始／优化路径的完整生成 IDs、文本及原生评分一致。TTT 真实样本最终状态逐位一致，缓存隔离检查通过。
- 仅尾段压紧在所测边界状态逐位一致；MLP 分块在 17,406-token 合成输入出现 logits／内部状态差异，最大 logits 绝对差为 0.25，不能保证任意输入的生成一致。
- 此前独立 FA2／cuDNN 后端对照未通过完整生成一致性；cuDNN 还更慢，均未采用；当时保留 SDPA，实际调用 PyTorch FlashAttention。

| 256K 路径 | TTT 峰值显存 | baseline 峰值显存 |
|---|---:|---:|
| 原始 | 70.72 GiB | 62.87 GiB |
| 仅压紧尾段 | 63.28 GiB | 不适用 |
| 压紧尾段并分块 MLP | 56.84 GiB | 56.43 GiB |

此前 B1 的 TTT 256K 平均耗时约 127.82 → 127.38 秒，没有实质提速。128K／256K 原 attention kernel 分别占所测真实 GPU 活动时间约 80.3%／89.4%；该单样本内核对照不能用于估计本轮批量推理速度。入口默认启用 `expandable_segments:True`，用于缓解已遇到的显存碎片问题。

## 结果记录

三个方法的评估保存在 `<output-root>/eval/<output-id>/<benchmark>/`，默认 root 为 `artifacts/reproductions/<method>/`，默认 ID 为 `<checkpoint_name>_<YYYYMMDD-HHMM>`（上海时间）；续评须显式指定原 ID。产物保留运行计划、代码来源、逐样本预测、执行状态和汇总。LaCT Momentum 的已有正式结果保留；优化诊断、选参和临时验证产物不作为 benchmark 分数。

三个方法续评均按全局样本 ID 补齐未完成记录，允许调整 GPU 数量、顺序和 batch 设置；`original` 仍限 B1。E²-TTT 对尚未写出评分的样本复用原生生成缓存，再按原 task 评分。数据路径在代码中固定，目录见各自 README。源码边界见各方法 `UPSTREAM.md`。
