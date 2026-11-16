# 20261108_E²-TTT 上游来源

创建时间：20261108 17:02:06 Asia/Shanghai

最后修订时间：20261116 14:58:56 Asia/Shanghai

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
| `eval.sh`、`scripts/evaluate_parallel.py` | 按全局样本 ID 分配单卡／双卡任务；固定评估协议与 GPU／batch 设置分开记录，产物按共同 output-id 与 benchmark 子目录组织 |
| `scripts/evaluate.py` | 每个 worker 加载一次 checkpoint 后依次处理多个 benchmark，分别保存原始结果和缓存；保留随附 evaluator 和 CachingLM |
| `scripts/evaluation_state.py` | 固定数据路径与评估协议，收集历史原始逐题记录，复制原生生成缓存并按原始样本顺序汇总 |
| `scripts/batch_harness.py` | 保留生成参数及 BOS 分组，复用原 HFLM 的长度降序排列、相邻组批、左 padding、生成、停止与结果还原；尾批使用实际数量 |
| `optimized/` | 本地 MLP／SwiGLU 批量推理，统一处理单样本、等长和混长输入，保持各样本的 attention 与 TTT 状态独立 |
| `scripts/local_tasks.py` | 固定本地数据接入及原始行 ID 选择，保留原 prompt、生成参数及计分；RULER 拼接一次 answer_prefix 并严格检查 token 预算 |
| `scripts/train_340m_gpu1.sh` | 适配物理 GPU 1，保持官方全局 batch、训练步数及其余参数，调用原 training/train.sh |
| `requirements-eval.txt` | 固定独立评估环境依赖 |

SQuAD／SWDE 沿用 HFLM 的 2048 总预算、48 输出预算和左截断；RULER 包含 128 输出预算且不截断。Bootstrap 为 0，仅省略标准误估计。原 task、harness 及模型源码不变，这些接入文件不构成统一训练框架。

`original` 为默认模式，使用迁入模型的原始推理路径，仅允许 batch size 1；`optimized` 对加载后的同一模型实例应用本地实现。两种模式共用 checkpoint、prompt、精度、greedy 生成预算及原 task 评分，使用不同输出目录。运行记录分别保存本地 Git commit、未提交状态、模式与实现路径；续评保留首次计划来源，并记录当次执行来源。

续评按全局样本 ID 复用已完成记录，仅重新分配未完成样本，允许更换 GPU 数量、顺序和 batch size。尚未写出评分的生成结果通过原生 SQLite 缓存恢复，缓存 key 和响应原样保留；每次启动使用独立 worker 目录和数据库。输出位于 `<output-root>/<output-id>/<benchmark>/`；父目录保留总协议和启动记录，各 benchmark 独立保存计划、记录、缓存与汇总。续评须显式指定原 ID，也可只选择原计划内的部分 benchmark。数据路径固定在 `scripts/evaluation_state.py`，具体目录见 README。

## 本地批量实现

| 本地文件 | 上游来源与范围 |
|---|---|
| `optimized/mlp.py` | 从 `upstream/e2_ttt/ops/e2_ttt_mlp/chunk_momentum.py` 复制 closed-form chunk 更新、prefill 和缓存解码函数 |
| `optimized/swiglu.py` | 从 `upstream/e2_ttt/ops/e2_ttt_swiglu/chunk_momentum.py` 复制对应的三个函数 |
| `optimized/tail.py` | 提取上述函数的 momentum 尾部裁剪与累加；按单一样本的 heads 确定裁剪范围，保留原 batch size 1 的规则 |
| `optimized/model.py` | 两种模型的批量 backbone 接入；所有 batch 统一使用真实位置和逐样本状态调度 |
| `optimized/padding.py` | 按真实位置计算变长 attention；逐样本保存原 TTT tuple、有效 KV 和真实长度，复用原读出与更新函数 |

归一化、梯度与时间权重等公共函数继续导入上游实现。缓存解码先将归一化参数扩展至 `batch × heads` 再构造 target；其余更新公式、chunk 边界、更新前读出、门控与归一化顺序保持原规则。当前范围限定为公开 MLP／SwiGLU 的 closed-form Momentum、无 short convolution、非 fused TTT 配置；其他配置明确拒绝。

适配只替换当前模型实例的 backbone 类，并标记层的 MLP／SwiGLU 类型；不修改模型注册表、原层类或全局函数，不新增、删除或重排参数，`state_dict` 键保持不变。各样本保存独立 TTT tuple、有效 KV 和真实长度；单样本及等长输入也使用同一状态路径。

正式评估按长度排序后相邻组批，生成、停止、截断、评分与 CachingLM 保留原行为。验证范围见[复现状态](../20261110_ttt_reproduction_status.md)，运行方式见 [README](README.md)。
