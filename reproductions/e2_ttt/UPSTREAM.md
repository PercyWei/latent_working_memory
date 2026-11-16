# 20261108_E²-TTT 上游来源

创建时间：20261108 17:02:06 Asia/Shanghai

最后修订时间：20261116 10:23:53 Asia/Shanghai

- 官方仓库：<https://github.com/zeyun-zhong/E2-TTT>。
- 固定 revision：`0d1da370c7e3cea1642ba3af0477b7601849a1be`。
- 论文：<https://arxiv.org/abs/2608.21308>。
- 迁入方式：对固定 revision 执行 `git archive HEAD`，解包到 `upstream/`，保留官方模型、flame 与随附评估框架；不迁入 `.git`、模型、数据缓存或运行结果。
- `upstream/` 的已跟踪文件原样保留，未修改模型、TTT 更新函数、训练框架或评估逻辑；本地批量实现单独位于 `optimized/`。

## 许可证与依赖来源

主项目为 MIT，见 [upstream/LICENSE](upstream/LICENSE)。随附 flame 保留 [training/LICENSE](upstream/training/LICENSE)，lm-evaluation-harness 保留 [LICENSE.md](upstream/lm-evaluation-harness/LICENSE.md) 与其他来源声明。

官方记录的 harness revision 为 `c9772b90a8ee95b0df1ba76651443a0534c96aad`，包含四处作者修改，见 [training/README.md](upstream/training/README.md) 的 “Vendored lm-evaluation-harness”。这些修改属于官方实现。

## 本地代码边界

| 文件 | 职责 |
|---|---|
| `eval.sh`、`eval_parallel.sh`、`scripts/evaluate_parallel.py` | 选择推理模式、分配评估样本并汇总原始逐题指标 |
| `scripts/evaluate.py` | 加载 checkpoint 并选择原始或批量推理，使用随附 evaluator 计分；原生 CachingLM 支持续评 |
| `scripts/batch_harness.py` | 保留生成参数及 BOS 分组，复用原 HFLM 的长度降序排列、相邻组批、左 padding、生成、停止与结果还原；尾批使用实际数量 |
| `optimized/` | 本地 MLP／SwiGLU 批量推理，支持等长输入和标准左 padding，保持各样本的 attention 与 TTT 状态独立 |
| `scripts/local_tasks.py` | 本地数据接入及互斥行分片，保留原 prompt、生成参数及计分；RULER 拼接一次 answer_prefix 并严格检查 token 预算 |
| `scripts/train_340m_gpu1.sh` | 适配物理 GPU 1，保持官方全局 batch、训练步数及其余参数，调用原 training/train.sh |
| `requirements-eval.txt` | 固定独立评估环境依赖 |

SQuAD／SWDE 沿用 HFLM 的 2048 总预算、48 输出预算和左截断；RULER 包含 128 输出预算且不截断。Bootstrap 为 0，仅省略标准误估计。原 task、harness 及模型源码不变，这些接入文件不构成统一训练框架。

`original` 使用迁入模型的原始推理路径，仅允许 batch size 1；`optimized` 对加载后的同一模型实例应用本地实现。两种模式共用 checkpoint、prompt、精度、greedy 生成预算及原 task 评分，使用不同输出目录。运行记录分别保存本地 Git commit、未提交状态、模式与实现路径；续评保留首次计划来源，并记录当次执行来源。

## 本地批量实现

| 本地文件 | 上游来源与范围 |
|---|---|
| `optimized/mlp.py` | 从 `upstream/e2_ttt/ops/e2_ttt_mlp/chunk_momentum.py` 复制 closed-form chunk 更新、prefill 和缓存解码函数 |
| `optimized/swiglu.py` | 从 `upstream/e2_ttt/ops/e2_ttt_swiglu/chunk_momentum.py` 复制对应的三个函数 |
| `optimized/tail.py` | 提取上述函数的 momentum 尾部裁剪与累加；按单一样本的 heads 确定裁剪范围，保留原 batch size 1 的规则 |
| `optimized/model.py` | 两种模型的批量 backbone 与层接入；保留无 short convolution 的等长路径，并分派左 padding 路径 |
| `optimized/padding.py` | 按真实位置计算变长 attention；逐样本保存原 TTT tuple、有效 KV 和真实长度，复用原读出与更新函数 |

归一化、梯度与时间权重等公共函数继续导入上游实现。缓存解码先将归一化参数扩展至 `batch × heads` 再构造 target；其余更新公式、chunk 边界、更新前读出、门控与归一化顺序保持原规则。当前范围限定为公开 MLP／SwiGLU 的 closed-form Momentum、无 short convolution、非 fused TTT 配置；其他配置明确拒绝。

适配只替换当前实例的 backbone 和层类，不修改模型注册表或全局函数，不新增、删除或重排参数，`state_dict` 键保持不变。等长路径的缓存 tuple 不变；padding 路径逐样本保存相同的 TTT tuple，并单独记录真实长度。代表性验证结果与数值差异见[批量推理记录](../20261110_e2_ttt_batch_inference.md)。当前命令见 [README.md](README.md)，整体进度见[复现状态](../20261110_ttt_reproduction_status.md)。

常规 `optimized` 评估允许相邻长度混合组批，复用上述 padding 路径；原 prompt、截断、生成预算、停止、评分与 CachingLM 不变。左 padding 的计算边界、验证命令和结果见[padding 推理记录](../20261115_ttt_padding_inference.md)。续评同时核对组批策略，策略不同的运行使用独立输出目录。
