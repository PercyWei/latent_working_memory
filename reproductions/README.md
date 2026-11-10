# 论文复现

本目录保存论文方法的独立复现。训练和评估入口、依赖环境、源码来源及结果分别在各方法目录维护。

| 方法 | 入口 | 范围 |
|---|---|---|
| ICAE v1 | [icae](icae/README.md) | Llama-2-7B-Chat 压缩器及公开权重评估 |
| C-DIC | [cdic](cdic/README.md) | 多轮记忆、MSC 训练与评估 |
| LaCT | [lact](lact/README.md) | 官方语言模型实现与原生训练流程 |
| E²-TTT | [e2_ttt](e2_ttt/README.md) | 官方语言模型实现、状态转移检查及原生训练评估 |
| In-Place TTT | [in_place_ttt](in_place_ttt/README.md) | 官方训练、checkpoint 转换与 RULER 评估流程 |

## TTT baselines 第一阶段

LaCT、E²-TTT 和 In-Place TTT 的源码迁入、独立环境与 checkpoint 运行检查已完成，当前只剩正式评估。保留各自官方实现与训练框架，不加入新的写入目标或统一模型接口。验证范围、优化结论与耗时参考集中见 [复现状态](20261110_ttt_reproduction_status.md)，固定源码版本与本地差异见各目录的 `UPSTREAM.md`。

各方法的 `upstream/` 保存官方源码，外层保存本地运行入口与记录。它们使用的 CUDA、PyTorch 和训练框架不同，按各自说明配置环境，不将这些依赖合并到主项目环境。

本次实验使用用户授权的 `zcy-memory` 物理 GPU 0。入口允许单卡 0 或 1，默认 0，启动时显式设置 `CUDA_VISIBLE_DEVICES=0`；进程内使用逻辑设备 `cuda:0`。原始上游脚本可能包含其他 GPU 编号或多卡配置，执行前须使用各方法目录中注明的本地入口。

源码与启动入口可用、CPU 检查通过、GPU 最小运行通过、公开 checkpoint 评估复现和训练结果复现分别记录。具体结果以各方法记录为准，代码迁移完成不表示论文指标已复现。

TTT baselines 的模型保存在 `~/rivermind-data/models/reproductions/<method>/`，数据与实验输出分别保存在项目的 `data/`、`artifacts/reproductions/<method>/` 下。上游脚本额外生成的原生日志位置在各方法说明中注明，并排除出 Git。GPU 任务从同步后的服务器项目仓库运行。

三种方法所需评估数据的本地下载、RULER 构造命令及模型间共用范围统一见 [评估数据准备](20261108_ttt_data_preparation.md)。

## TTT checkpoint 评估

三个独立评估环境已在服务器安装，实际版本、安装命令和检查结果见 [评估环境](20261109_environment_setup.md)。在各方法环境中使用对应的 `eval.sh`，显式传入 `--checkpoint` 和 `--output-dir`。本地数据默认采用上述准备文档中的目录，可通过数据路径参数覆盖；输出目录须为新目录。

| 方法 | 入口 | 默认评估范围 | 完整命令与环境 |
|---|---|---|---|
| LaCT | [lact/eval.sh](lact/eval.sh) | 六项 RULER，4K／8K／16K／32K | [LaCT 说明](lact/README.md) |
| E²-TTT | [e2_ttt/eval.sh](e2_ttt/eval.sh) | 两项 RULER，512–16K；SQuAD completion、SWDE | [E²-TTT 说明](e2_ttt/README.md) |
| In-Place TTT | [in_place_ttt/eval.sh](in_place_ttt/eval.sh) | 十三项 RULER，4K–256K | [In-Place TTT 说明](in_place_ttt/README.md) |

三个方法顺序使用空闲 GPU 0，首次运行使用新输出目录，中断后以相同设置追加 `--resume`。完整命令、生成方式及结果文件分别在各方法 README 中维护；验证与优化结果见 [复现状态](20261110_ttt_reproduction_status.md)。

## 模型下载约定

在 `zcy-memory` 上使用 `hf download` 下载模型；如果尚未安装 `hf` CLI，可执行：

```bash
uv tool install huggingface_hub
```

下载命令及实际模型目录分别见 [LaCT](lact/README.md#模型下载)、[E²-TTT](e2_ttt/README.md#模型下载) 和 [In-Place TTT](in_place_ttt/README.md#模型下载)。命令使用仓库当前默认分支，不指定固定版本；正式评估时记录实际使用的 checkpoint 版本。首批共五个完整 checkpoint：LaCT Momentum、两个 E²-TTT 1.3B 模型，以及第三方 In-Place TTT 32K 的 TTT／baseline；LaCT 的两个 Muon 变体为可选下载。
