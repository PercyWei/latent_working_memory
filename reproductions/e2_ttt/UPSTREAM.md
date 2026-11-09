# 20261108_E²-TTT 上游来源

创建时间：20261108 17:02:06 Asia/Shanghai

最后修订时间：20261109 11:54:14 Asia/Shanghai

- 官方仓库：<https://github.com/zeyun-zhong/E2-TTT>
- 固定 revision：`0d1da370c7e3cea1642ba3af0477b7601849a1be`（`master`）。
- 论文：<https://arxiv.org/abs/2608.21308>；本次方法及目标数字对照已下载的原文全文。
- 迁入方式：对固定 revision 执行 `git archive HEAD`，解包到 `upstream/`，保留完整官方源码与随附评估框架；不迁入 `.git`、模型、数据缓存或运行结果。
- `upstream/` 中的已跟踪文件原样保留，未修改模型、标量核、训练框架或评估逻辑。

## 许可证与依赖来源

主项目为 MIT，见 [upstream/LICENSE](upstream/LICENSE)。随附 flame 保留 [training/LICENSE](upstream/training/LICENSE)，随附 lm-evaluation-harness 保留 [LICENSE.md](upstream/lm-evaluation-harness/LICENSE.md) 与其他来源声明。

官方记录的 harness revision 为 `c9772b90a8ee95b0df1ba76651443a0534c96aad`，并包含四处作者修改，详见 [training/README.md](upstream/training/README.md#vendored-lm-evaluation-harness)。这些修改属于官方 E²-TTT 实现，本次保留。

## 本地文件

- `README.md`：具体论文目标、依赖、数据及执行入口。
- `eval.sh`：在项目根目录直接调用方法环境的单进程 Python，默认物理 GPU 0，仅接受单卡 `CUDA_VISIBLE_DEVICES=0` 或 `1`。
- `scripts/evaluate.py`：加载一次原 E²-TTT 模型，使用随附 HFLM 与 `evaluator.evaluate` 的显式 task 字典运行本地 benchmark，记录计划、逐样本及按任务／长度汇总的结果；`--resume` 核对原计划并通过原生 `CachingLM` 响应缓存重新构建完整结果，两个 HFLM wrapper 分别使用原生 cache hook。
- `scripts/local_tasks.py`：只替换数据接入。SQuAD／SWDE 继承原 task 的 prompt、生成参数及计分；绕开原 task 忽略 config kwargs 的初始化，读取本地完整 validation Parquet。RULER 读取既有 JSONL，拼接一次 answer_prefix，调用原 common_utils 计分，检查实际 token 预算，超限报错而不截断。原 task 和 harness 源码未修改。

本地 SQuAD／SWDE 采用原 HFLM 的 2048 总预算，预留 48 输出，prompt 上限 2000 并左截断。RULER 各档长度包含完整 prompt 与 128 输出预算。Bootstrap 设为 0，只省略标准误估计，不改变准确率计分。
- `scripts/train_340m_gpu1.sh`：仅适配本轮物理 GPU 1 的分配；单卡、梯度累积 64，保持官方 global batch 256、28,672 steps 和其余训练参数，通过 `bash -e` 调用官方 `train.sh`，使训练失败时停止，不继续执行 checkpoint 转换。
- `.gitignore`：允许上游模型源码、flame 配置及 harness 文档／模型源码被 Git 跟踪；覆盖项目根目录对 `models/`、`docs/` 产物的泛化忽略。
- `20261108_phase1_record.md`：本地已执行检查及尚未完成的运行。

这些外围文件不构成统一训练框架，也不替代官方模型或评估实现。
