# 20261108_E²-TTT 上游来源

创建时间：20261108 17:02:06 Asia/Shanghai

最后修订时间：20261117 11:42:23 Asia/Shanghai

## 来源与迁入范围

- 官方仓库：[zeyun-zhong/E2-TTT](https://github.com/zeyun-zhong/E2-TTT)，revision `0d1da370c7e3cea1642ba3af0477b7601849a1be`；[论文](https://arxiv.org/abs/2608.21308)。
- 对固定 revision 执行 `git archive HEAD`，将官方模型、flame 和随附评估框架迁入 `source/e2_ttt/`。其中已跟踪源码原样保留，本地适配与优化位于目录外。
- 主项目采用 [MIT](source/e2_ttt/LICENSE)；flame 和 harness 分别保留 [training/LICENSE](source/e2_ttt/training/LICENSE)、[LICENSE.md](source/e2_ttt/lm-evaluation-harness/LICENSE.md) 及来源声明。

随附 harness 基于 `c9772b90a8ee95b0df1ba76651443a0534c96aad`，包含四处作者修改，本地沿用这份上游实现。修改清单见 [Vendored lm-evaluation-harness](source/e2_ttt/training/README.md)。

## 代码职责

| 文件 | 本地职责 |
|---|---|
| `optimized/model.py`、`optimized/padding.py` | 接入 MLP／SwiGLU 批量推理，处理有效位置、变长 attention 和逐样本缓存 |
| `optimized/mlp.py`、`optimized/swiglu.py`、`optimized/tail.py` | 基于 `source/e2_ttt/e2_ttt/ops/e2_ttt_{mlp,swiglu}/chunk_momentum.py` 保留 closed-form 更新、prefill 与缓存解码，适配归一化参数和逐样本尾部裁剪 |
| `scripts/eval.sh`、`evaluation/evaluate_parallel.py`、`evaluation/batching.py` | 单卡／多卡调度、样本分配和按 RULER 长度设置 batch |
| `evaluation/evaluate.py` | 每个 worker 加载一次模型，依次执行分配到的多个 benchmark |
| `evaluation/batch_harness.py`、`evaluation/local_tasks.py` | harness 组批、本地数据与全局样本 ID 接入 |
| `evaluation/evaluation_state.py` | 固定评估协议、续评、原生缓存合并、来源记录与结果汇总 |
| `../evaluation_runtime.py` | 三方法共用的进程清理、时间、严格 JSON、基础 Git 来源和总体进度 |
| `scripts/train_340m_gpu1.sh`、`requirements-eval.txt` | 单卡备用训练入口与独立评估环境依赖 |

## 推理实现改进

上游已实现 closed-form Momentum 和局部 attention。本地改动集中于批量推理接入，保留更新公式、chunk 边界、更新前读出、门控和归一化顺序。

- **实例适配**：`original`（默认）使用上游 batch size 1 路径；`optimized` 替换当前模型实例的 backbone 类并标记层类型，保留参数与 `state_dict` 键。支持范围为公开 MLP／SwiGLU checkpoint 的 closed-form Momentum、无 short convolution、非 fused TTT 配置。
- **混长输入**：单样本、等长及左 padding 混长输入使用同一状态路径。投影、FFN 和变长 attention 批量执行，位置计算沿用固定 FLA 版本的原生 RoPE 规则。
- **状态隔离**：TTT 按样本执行，分别保存原生状态 tuple、有效 KV 和真实长度；padding 位置跳过 attention 与 TTT 读写。
- **归一化维度**：本地缓存解码函数先将归一化参数扩展至 `batch × heads`，再构造重建 target；归一化、梯度和时间权重等公共计算继续调用上游函数。
- **Momentum 尾部裁剪**：按每个样本自身的 heads 确定裁剪范围，保留上游 batch size 1 的裁剪与累加语义。

## 评估流程改进

- **本地数据接入**：RULER 读取 JSONL，`input + answer_prefix` 拼接一次；SQuAD／SWDE 读取 validation Parquet。保留原 task 评分、HFLM 生成和文本处理，生成预算见下表。
- **批量生成接入**：按生成参数和 BOS 状态分组，复用 HFLM 的长度排序、相邻组批、左 padding、停止及结果还原。尾批保留实际数量，RULER 可按评估长度设置 batch 上限。
- **模型复用**：每张可见 GPU 启动一个 worker，一次加载模型后依次处理多个 benchmark。
- **按样本续评**：按全局 ID 收集历史记录、重新分配未完成样本。支持调整 GPU 数量、顺序和 batch，也可选择原计划内的 benchmark 子集。
- **原生生成缓存**：各 benchmark 使用原生 `CachingLM` SQLite 缓存。每次启动建立独立 worker 目录和数据库，复制历史缓存的原生 key／响应，用于恢复尚未保存评分的生成结果。
- **协议与来源**：固定模型、数据、生成／评分设置和软件版本，逐轮记录 GPU 与 batch。目录锁和存活 worker 检查协调启动，记录保留 Git commit、未提交状态、模式及实现路径。
- **产物组织**：结果保存到 `<output-root>/eval/<output-id>/<benchmark>/`，按原始样本顺序合并逐题记录。各 benchmark 独立汇总，父目录记录整体完成状态。
- **总体进度**：只读各 worker 已评分 JSONL 的完整行累计完成数，随各任务评分记录的写出成批更新。

| 评估 | 沿用的生成与文本处理 |
|---|---|
| RULER | greedy 最多生成 128 tokens；检查总长度预算，保留完整输入 |
| SQuAD／SWDE | prompt 和 target 执行 `.strip()`，遇换行停止；总预算 2048、生成预算 48，prompt 按原规则左截断至 2000 tokens |

Bootstrap 设为 `0`，省略标准误估计；任务评分沿用原 evaluator。

具体命令、参数和产物清单见 [README](README.md)，验证范围、数值差异与性能记录见[复现状态](../20261110_ttt_reproduction_status.md)。备用训练入口调用原 `source/e2_ttt/training/train.sh`，使用物理 GPU 1，保留官方 340M recipe 的全局 batch 256 与训练步数；训练设置见[官方训练说明](source/e2_ttt/training/README.md)。
