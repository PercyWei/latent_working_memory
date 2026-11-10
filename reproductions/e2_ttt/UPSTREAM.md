# 20261108_E²-TTT 上游来源

创建时间：20261108 17:02:06 Asia/Shanghai

最后修订时间：20261110 11:34:26 Asia/Shanghai

- 官方仓库：<https://github.com/zeyun-zhong/E2-TTT>。
- 固定 revision：`0d1da370c7e3cea1642ba3af0477b7601849a1be`。
- 论文：<https://arxiv.org/abs/2608.21308>。
- 迁入方式：对固定 revision 执行 `git archive HEAD`，解包到 `upstream/`，保留官方模型、flame 与随附评估框架；不迁入 `.git`、模型、数据缓存或运行结果。
- `upstream/` 的已跟踪文件原样保留，未修改模型、标量核、训练框架或评估逻辑；本地推理优化未实施。

## 许可证与依赖来源

主项目为 MIT，见 [upstream/LICENSE](upstream/LICENSE)。随附 flame 保留 [training/LICENSE](upstream/training/LICENSE)，lm-evaluation-harness 保留 [LICENSE.md](upstream/lm-evaluation-harness/LICENSE.md) 与其他来源声明。

官方记录的 harness revision 为 `c9772b90a8ee95b0df1ba76651443a0534c96aad`，包含四处作者修改，见 [training/README.md](upstream/training/README.md#vendored-lm-evaluation-harness)。这些修改属于官方实现。

## 本地代码边界

| 文件 | 职责 |
|---|---|
| `eval.sh` | 从仓库根目录调用方法环境 Python，限定单进程、单卡 0 或 1 |
| `scripts/evaluate.py` | 加载一次原模型，使用随附 HFLM／evaluator 评估本地数据，记录结果；原生 CachingLM 支持续评 |
| `scripts/local_tasks.py` | 仅替换数据接入，保留原 prompt、生成参数及计分；RULER 拼接一次 answer_prefix 并严格检查 token 预算 |
| `scripts/train_340m_gpu1.sh` | 适配物理 GPU 1，保持官方全局 batch、训练步数及其余参数，调用原 training/train.sh |
| `requirements-eval.txt` | 固定独立评估环境依赖 |

SQuAD／SWDE 沿用 HFLM 的 2048 总预算、48 输出预算和左截断；RULER 包含 128 输出预算且不截断。Bootstrap 为 0，仅省略标准误估计。原 task、harness 及模型源码不变，这些接入文件不构成统一训练框架。

当前命令和协议见 [README.md](README.md)，迁移验证与优化分析结论集中记录于[复现状态](../20261110_ttt_reproduction_status.md)。
