# 论文复现

本目录保存论文方法的独立复现。训练和评估入口、依赖环境、源码来源及结果分别在各方法目录维护。

| 方法 | 入口 | 范围 |
|---|---|---|
| ICAE v1 | [icae](icae/README.md) | Llama-2-7B-Chat 压缩器及公开权重评估 |
| C-DIC | [cdic](cdic/README.md) | 多轮记忆、MSC 训练与评估 |
| LaCT | [lact](lact/README.md) | 760M 模型的 RULER 评估 |
| E²-TTT | [e2_ttt](e2_ttt/README.md) | 1.3B 模型的 RULER、SQuAD completion、SWDE 评估 |
| In-Place TTT | [in_place_ttt](in_place_ttt/README.md) | Qwen3-4B 32K TTT／baseline 的 RULER 评估 |

## TTT 模型评估

三个方法的代码迁入和代表性运行检查已完成，尚未完成本轮计划的 benchmark 评估。使用各自的独立环境，从服务器项目根目录执行：

| 方法 | 入口 | 默认评估范围 | 完整命令与环境 |
|---|---|---|---|
| LaCT | [lact/eval.sh](lact/eval.sh) | 六项 RULER，4K／8K／16K／32K | [LaCT 说明](lact/README.md) |
| E²-TTT | [e2_ttt/eval.sh](e2_ttt/eval.sh) | 两项 RULER，512–16K；SQuAD completion、SWDE | [E²-TTT 说明](e2_ttt/README.md) |
| In-Place TTT | [in_place_ttt/eval.sh](in_place_ttt/eval.sh) | 十三项 RULER，4K–256K | [In-Place TTT 说明](in_place_ttt/README.md) |

首次使用新输出目录；续评在原命令末尾追加 `--resume`，保持 GPU 顺序与参数一致。模型目录为 `~/rivermind-data/models/reproductions/<method>/`，数据位于 `data/`，结果位于 `artifacts/reproductions/<method>/`。

环境安装见[评估环境](20261109_environment_setup.md)，数据下载与构造见[评估数据准备](20261108_ttt_data_preparation.md)。验证及优化结论见[复现状态](20261110_ttt_reproduction_status.md)，参数与单卡耗时依据见[运行参数测试](20261110_ttt_evaluation_tuning.md)及 [E²-TTT 批量推理记录](20261110_e2_ttt_batch_inference.md)，源码来源见各方法的 `UPSTREAM.md`。

## 模型下载约定

在 `zcy-memory` 上使用 `hf download` 下载模型；如果尚未安装 `hf` CLI，可执行：

```bash
uv tool install huggingface_hub
```

下载命令见 [LaCT](lact/README.md#模型下载)、[E²-TTT](e2_ttt/README.md#模型下载) 和 [In-Place TTT](in_place_ttt/README.md#模型与数据准备)。首批共五个 checkpoint：LaCT Momentum、两个 E²-TTT 1.3B 模型，以及第三方 In-Place TTT 32K 的 TTT／baseline；LaCT Muon 为可选下载。
