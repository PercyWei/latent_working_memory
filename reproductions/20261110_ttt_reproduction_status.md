# 20261110_TTT 第一阶段复现状态与优化结论

创建时间：20261110 11:30:45 Asia/Shanghai

最后修订时间：20261110 16:13:07 Asia/Shanghai

## 当前状态

LaCT、E²-TTT、In-Place TTT 的源码迁入、独立环境、本地数据接入及 checkpoint 运行检查已完成。第一阶段剩余工作是完成下表的正式评估；目前没有完整 benchmark 分数，不能报告论文指标已复现。

各方法保留官方源码和独立 `eval.sh`，不接入统一训练框架。运行使用 `zcy-memory` 的单张 A800，显式设置 `CUDA_VISIBLE_DEVICES=0`；完整命令在各方法 README 中维护。数据下载、构造与共用范围见[数据准备](20261108_ttt_data_preparation.md)，依赖与安装命令见[评估环境](20261109_environment_setup.md)。

评估参数测试的测量、当前设置与完成时间估计统一见[运行参数测试](20261110_ttt_evaluation_tuning.md)。LaCT 推荐单进程 batch size 16；E²-TTT 与 In-Place TTT 保持每进程 batch size 1，独立进程数另行指定。LaCT 的 4K 补测及整体估时仍在进行。

| checkpoint | 正式评估范围 | 当前使用方式 |
|---|---|---|
| LaCT Momentum 760M | 六项 RULER，4K／8K／16K／32K，每任务每长度 500 条，共 12,000 条 | [增量推理、单进程 batch size 16](lact/README.md#环境与运行命令)，按精确 token 数分组且无 padding；原始路径保留作对照 |
| E²-TTT MLP／SwiGLU 1.3B | 每模型：两项 RULER，512–16K，共 6,000 条；SQuAD completion 2,984 条；SWDE 1,111 条 | [官方实现，最多 4／6 个独立进程](e2_ttt/README.md#环境与评估命令)，每模型共 10,095 条 |
| In-Place TTT 32K TTT／baseline | 每模型：十三项 RULER，4K–256K 七个长度，每任务每长度 100 条，共 9,100 条 | [完整 attention、单进程](in_place_ttt/README.md#ruler-评估)；TTT 仅压紧尾部缓存，baseline 使用原始路径 |

LaCT 使用作者公开的重训权重；E²-TTT 使用作者官方 checkpoint；In-Place TTT 使用第三方 ProLong 32K／6,000 steps 权重，不对应论文原模型，其超过 32K 的结果属于长度外推。三个方法均不需要为本轮 checkpoint 评估重新训练。

## 已完成的验证与优化

### LaCT

独立的 `lact/incremental/` 缓存窗口 KV、fast weights、归一化累计状态与待写 chunk，保持原始先读后写及尾段延迟更新；原模型和更新函数未修改。非空前缀长为 \(L\)、chunk 大小为 2,048 时，更新次数仍为 \(\lfloor(L-1)/2048\rfloor\)。当前支持等长、无 padding、非 fused TTT 批次；每条样本独立保存 KV 和 fast weights，同一 RULER 长度内跨六任务按精确输入 token 数分组。

- Momentum：四长度各两条、共 8 条 128-token 生成与原始路径 IDs 一致；55 次 teacher-forced argmax、缓存进度与跨样本隔离检查通过。logits 仍有数值差异，不保证任意输入或不同执行上下文逐位等价。
- Muon／PostNorm：仅通过 4K 两条生成检查；Muon／PreNorm 的 4K 生成仅 1／2 一致，尚不适用增量评估。
- 此前单进程的 Momentum 32K 单样本均值从 97.08 秒降至 9.93 秒；单进程峰值分配显存约 6.60 GiB。
- 当前推荐单进程 B16。32K S-NIAH-2 的 16 条同长度真实输入吞吐为 0.8266 样本／秒，全卡峰值 69,289 MiB（约 67.7 GiB）。此前 B1×8 独立进程在另一组 8 条输入上为 0.3513 样本／秒、68.8 GiB；样本量与输入不同，不作精确配对加速比。
- B2／4／8／16 在同一组 16 条 32K 输入中，各有 15 条完整生成 IDs 与 B1 相同，逐条评分均为 16／16 相同；边界检查结构、有限值与更新进度为 45／45 通过，teacher-forced argmax 为 208／210 相同。不同 batch 形状仍有数值差异，当前标准允许这类差异，要求公式、因果时序和样本隔离不变。
- 固定 B4 和 focus 槽位，仅置换其他三条样本，prefill 加四步 decode 共五次比较中，focus 的 logits、KV 与全部 TTT tensor 字段逐字节一致。该验证直接检查同形状执行下的样本隔离，不把跨 batch 形状的数值差误判为样本混合。

### In-Place TTT

`in_place_ttt/optimized/` 独立实现 TTT 尾段存储压紧与 MLP 分块；attention KV 与 fast weights 不裁剪，保留 1,024-token chunk 的原读出／更新顺序。默认仍为 `original`；`optimized --mlp-chunk-size 0` 仅压紧尾段，`16384` 同时启用 MLP 分块。

- 两个 checkpoint 各覆盖七长度、每长度两条，共 14 条真实样本；原始／优化路径的完整生成 IDs、文本及原生评分一致。TTT 真实样本最终状态逐位一致，缓存隔离检查通过。
- 仅尾段压紧在所测边界状态逐位一致；MLP 分块在 17,406-token 合成输入出现 logits／内部状态差异，最大 logits 绝对差为 0.25，不能保证任意输入的生成一致。
- 独立 FA2／cuDNN 后端对照未通过完整生成一致性；cuDNN 还更慢，均未采用。正式路径保留 SDPA，实际调用 PyTorch FlashAttention。

| 256K 路径 | TTT 峰值显存 | baseline 峰值显存 |
|---|---:|---:|
| 原始 | 70.72 GiB | 62.87 GiB |
| 仅压紧尾段 | 63.28 GiB | 不适用 |
| 压紧尾段并分块 MLP | 56.84 GiB | 56.43 GiB |

TTT 256K 平均耗时约 127.82 → 127.38 秒，没有实质提速。128K／256K 原 attention kernel 分别占所测真实 GPU 活动时间约 80.3%／89.4%，完整 attention 的提速探索暂告一段落。入口默认启用 `expandable_segments:True`，用于缓解已遇到的显存碎片问题。

### E²-TTT

使用官方增量推理、512-token SWA 和 TTT chunk；未修改模型计算。独立进程并发调参与正式运行设置见[运行参数测试](20261110_ttt_evaluation_tuning.md)。

- 初始 GPU 检查：每模型 SQuAD／SWDE 各两条、RULER 六长度各两条，共 16 条，生成与原生评分流程可执行。
- 专项诊断：每模型 6 条真实输入、2 条边界输入，带诊断与原始生成的完整 IDs 全部一致。
- 16K prefill 后，attention KV 与 TTT 缓冲平均持有 5.75 GiB 超额底层存储，首个 decode 后释放；这是存储冗余，不是已实现的峰值节省。
- 127 次普通 decode 约 7.1–7.3 秒，普通单步约 55–57 ms；存在大量细粒度 CUDA 活动。同步回读集中在 prefill／完整 chunk 更新。若后续优化，先考虑缓存压紧与普通读出编译，再处理更新同步。

## 代码来源

当前评估参数和时间估计集中维护在[运行参数测试](20261110_ttt_evaluation_tuning.md)，不按空余显存或进程数线性换算耗时。

| 验证内容 | 实际计算代码 commit | 当时 Git 状态 |
|---|---|---|
| LaCT 增量配对 | `8df67e81` | 历史运行未自动采集，不补填 |
| LaCT 原生批量推理 32K 调参 | `a40a51f8` | clean |
| LaCT 固定槽位、同形状样本隔离 | `e76aa214` | clean |
| In-Place TTT 显存优化配对 | `f94e2231` | clean |
| In-Place TTT 最终 prefill profiling | `14796403` | clean |
| E²-TTT 阶段／缓存诊断 | `48d8ee43` | clean |
| E²-TTT 算子诊断 | `89a7929c` | clean |

必要结论集中保存在本记录。正式评估产物统一写入 `artifacts/reproductions/<method>/eval/<run>/`，保留计划、代码来源、逐样本预测、执行状态和完整汇总。小样本验证不计入正式 benchmark；所有入口均支持用相同设置与输出目录追加 `--resume`。
