# 论文复现

本目录保存论文方法的独立复现。训练和评估入口、依赖环境、源码来源及结果分别在各方法目录维护。

| 方法 | 入口 | 范围 |
|---|---|---|
| ICAE v1 | [icae](icae/README.md) | Llama-2-7B-Chat 压缩器及公开权重评估 |
| C-DIC | [cdic](cdic/README.md) | 多轮记忆、MSC 训练与评估 |
| LaCT | [lact](lact/README.md) | 760M 模型的 RULER 评估 |
| E²-TTT | [e2_ttt](e2_ttt/README.md) | 1.3B 模型的 RULER、SQuAD completion、SWDE 评估 |
| In-Place TTT | [in_place_ttt](in_place_ttt/README.md) | Qwen3-4B 32K TTT／baseline 的 RULER 评估 |

## 根目录文件

| 文件 | 职责与适用范围 |
|---|---|
| `README.md` | 方法入口、目录约定与公共文档索引 |
| [评估数据准备](20261108_ttt_data_preparation.md) | 三种 TTT 方法的数据规模、下载和构造命令 |
| [评估环境](20261109_environment_setup.md) | 三种 TTT 方法的依赖版本、环境安装及安装记录 |
| [复现状态](20261110_ttt_reproduction_status.md) | 正式评估进度、小样本验证结论、数值差异、资源测量及历史 B1 耗时外推 |
| [setup_eval_env.sh](setup_eval_env.sh) | 安装 LaCT、E²-TTT 或 In-Place TTT 的独立评估环境 |
| [evaluation_runtime.py](evaluation_runtime.py) | 三方法共用的进程清理、时间、严格 JSON、基础 Git 来源、worker 进度快照与统一展示 |

运行命令和当前参数由各方法 README 维护，来源与实现改进由各方法 `UPSTREAM.md` 维护；公共文档按上表分工。

## TTT 模型评估

三个方法已完成代码迁入和当前范围的小样本验证，正式评估进度与验证范围见[复现状态](20261110_ttt_reproduction_status.md)。评估使用各自的独立环境，命令均从项目根目录执行。

三个方法采用相同的外层目录分工：

| 位置 | 职责 |
|---|---|
| `source/<repo>/` | 按来源仓库保存迁入源码，保留仓库内部布局 |
| `optimized/` | 本地推理优化 |
| `evaluation/` | 各方法的 Python 评估模块，包含 `evaluate_parallel.py` 启动器、加载、续评、结果处理及报告模板 |
| `scripts/` | Shell 评估入口 `eval.sh` 和实际使用的训练启动脚本 |
| `configs/` | 实际使用的本地配置，按需设置 |
| `requirements-eval.txt` | 方法独立评估环境的依赖 |
| `README.md`、`UPSTREAM.md` | 使用方式、源码来源及本地改进 |

LaCT 的来源目录为 `source/lact/`、`source/flame/`；E²-TTT 为 `source/e2_ttt/`，In-Place TTT 为 `source/in_place_ttt/`。各方法的评估入口如下：

| 方法 | 入口 | 默认评估范围 | 完整命令与环境 |
|---|---|---|---|
| LaCT | [lact/scripts/eval.sh](lact/scripts/eval.sh) | 六项 RULER，4K／8K／16K／32K | [LaCT 说明](lact/README.md) |
| E²-TTT | [e2_ttt/scripts/eval.sh](e2_ttt/scripts/eval.sh) | 两项 RULER，512–16K；SQuAD completion、SWDE | [E²-TTT 说明](e2_ttt/README.md) |
| In-Place TTT | [in_place_ttt/scripts/eval.sh](in_place_ttt/scripts/eval.sh) | 十三项 RULER，4K–256K | [In-Place TTT 说明](in_place_ttt/README.md) |

三个方法默认使用 `original`，保留 B1 路径；`optimized` 支持单样本、等长及混长左 padding batch，通过 `--batch-size` 和 `--batch-size-by-length` 设置每卡上限。各样本的 attention／TTT 状态独立，验证范围见[复现状态](20261110_ttt_reproduction_status.md)。

三个方法的新运行均使用 `--output-root`／`--output-id`，各 benchmark 分别保存在 `<output-root>/eval/<output-id>/<benchmark>/`；续评须指定原 ID 并追加 `--resume`，GPU 数量、顺序及 batch 设置可调整。模型目录为 `~/rivermind-data/models/reproductions/<method>/`，数据位于 `data/`，结果位于 `artifacts/reproductions/<method>/`。

## 进度展示

三个 TTT 方法使用同一展示方式：

- 启动时显示一次方法、checkpoint、推理模式和物理 GPU。
- 总体完成数统计本次选择的 benchmark／长度范围，以已评分 JSONL 为准，显示总量、百分比、历史完成数与本轮完成数，以及阶段和耗时。
- 每个物理 GPU 一行，显示当前 benchmark、RULER 长度档位、阶段、生成数、完成数、当前 batch 上限、近期处理速度和当前 benchmark／长度档位的生成剩余时间；不估算跨长度的总剩余时间。`B` 表示当前上限，尾批可小于该值。

生成数按本轮待评请求逐条统计，包含原生缓存复用，重复 prompt 不合并计数。E²-TTT 的生成数随原生 batch 实时更新，完成数在原生任务评分并写出后成批增加；两者暂时不同是正常现象。处理速度包含缓存复用，不等同于模型推理吞吐。

阶段变化及时刷新；常规刷新间隔为交互终端 5 秒、重定向输出 30 秒。worker 的 `progress.json` 仅为观测快照，不作为续评依据。

## 模型下载约定

使用 `hf download` 下载模型；如果尚未安装 `hf` CLI，可执行：

```bash
uv tool install huggingface_hub
```

下载命令见 [LaCT](lact/README.md#模型下载)、[E²-TTT](e2_ttt/README.md#模型下载) 和 [In-Place TTT](in_place_ttt/README.md#模型与数据准备)。首批共五个 checkpoint：LaCT Momentum、两个 E²-TTT 1.3B 模型，以及第三方 In-Place TTT 32K 的 TTT／baseline；LaCT Muon 为可选下载。
