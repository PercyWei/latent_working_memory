# 20261108_TTT baselines 第一阶段复现

创建时间：20261108 16:59:08 Asia/Shanghai

最后修订时间：20261109 03:48:37 Asia/Shanghai

## 目标与范围

在 `reproductions/` 中分别接入 LaCT、E²-TTT 和 In-Place TTT 的官方实现，保留各自模型、更新规则、训练框架和评估协议，验证论文中的语言模型结果。本阶段不实现统一训练框架，不修改记忆接入方式，不加入新的 Ω 目标。

每个方法保存固定源码 revision、许可证、原始配置与运行说明。公开 checkpoint 的评估复现和从训练开始的复现分别记录；语法检查或最小运行不能视为论文指标复现。

## 执行安排

1. 迁入官方源码，阅读原论文及仓库文档，核对代码、数据、checkpoint 和目标结果。
2. 为每个方法记录具体目标、官方入口、环境要求、训练预算和评估产物；保留必要兼容改动的说明。
3. 运行现有的可执行检查；具备 GPU、模型和数据后，先核对公开 checkpoint 的原协议评估，再推进训练复现。
4. 将实际命令、环境、结果和未解决事项记录到对应方法目录。实验输出统一放在 `artifacts/reproductions/<method>/`，模型及数据不写入源码目录。

本次 GPU 实验仅使用 `zcy-memory` 的物理 GPU 1，须等现有任务释放该卡后再启动。当前先完成代码迁移与本地检查。采用原方法的本地日志与评估输出；不创建新的 SwanLab project。

## 运行条件

- 本地：macOS arm64，PyTorch 2.6.0，无 CUDA。
- 当前服务器：`zcy-memory`，仓库为 `/root/rivermind-data/latent_working_memory`，连接已确认。GPU 0、1 为 A800 80GB，正在运行已有双卡训练；本次不修改或停止该任务。
- 后续 GPU 实验从已通过 GitHub（必要时 Gitee）同步的服务器项目仓库执行；本次尚未启动 GPU 实验。

## 方法记录

| 方法 | 目录 | 状态 |
|---|---|---|
| LaCT | [lact](lact/README.md) | 官方 LM 与 flame 已接入；原写入函数的 CPU 前后向、分段时序检查通过 |
| E²-TTT | [e2_ttt](e2_ttt/README.md) | 官方模型、训练评估框架已接入；原标量核及验证函数的 CPU 状态转移检查通过 |
| In-Place TTT | [in_place_ttt](in_place_ttt/README.md) | 官方训练、导出、推理与评估配置已接入；原分块 MLP 的 CPU 计算对照通过 |

三个方法的核心算法源码均保留官方实现；外围入口限定物理 GPU 1，或在执行命令中显式指定单卡。源码版本、兼容接入及检查范围见各方法的 `UPSTREAM.md` 和执行记录。

服务器方法环境尚待落实；用户正在下载模型，已有 RULER 源码、原始素材和 NLTK 资源可复用，尚未启动 GPU 训练或评估。代码迁移与本地核心检查已完成，论文指标尚未复现。后续先在空闲 GPU 1 上验证方法环境与完整模型，再评估 LaCT 作者公开重训权重、E²-TTT 官方 1.3B MLP／SwiGLU 权重和第三方 In-Place TTT 32K 权重。下载命令见各方法 README，模型统一放在 `~/rivermind-data/models/reproductions/<method>/`。In-Place TTT 的官方论文权重受内部数据政策限制未发布，第三方模型不能代替原论文实验产物。

## 接入验收

已完成三个方法的固定源码接入、许可证保留、GPU 1 本地入口和原文设置记录。LaCT 的 GD / Momentum / Muon CPU 前后向与分段时序检查、E²-TTT 的原状态转移数值检查、In-Place TTT 的原分块 MLP 训练／推理 CPU 对照均通过，具体参数和适用范围见方法记录。

收尾检查通过：本地 shell 入口语法、In-Place 评估入口 Ruff 与 `--help`、LaCT 配置与模型路径、各方法文档链接、必要源码的 Git 可跟踪性。没有创建统一模型框架、新测试套件或在线实验项目。GPU 1 等待期间不占用现有训练资源。

各方法评估数据的任务、规模及完整下载与构造命令统一见 [评估数据准备](20261108_ttt_data_preparation.md)。

RULER 生成脚本的本地 CPU 验证已覆盖 LaCT 六项各 1 条（4K）、E²-TTT 两项各 2 条（512）、In-Place TTT 十三项各 1 条（4K）及额外 1 条 S-NIAH-1（256K），共 24 条；完整输入编码加输出预算均未超过目标长度。尚未全量生成或运行模型评估。

checkpoint 评估入口已按方法分别提供 `eval.sh`，直接读取本地 RULER JSONL 与 E²-TTT 的 Parquet；实现记录及运行说明见 [checkpoint 评估](20261109_checkpoint_evaluation.md)。本次没有新增或运行测试，也没有启动 GPU 评估。
