# 20261108_In-Place TTT 官方代码来源

创建时间：20261108 17:05:19 Asia/Shanghai

最后修订时间：20261116 21:06:35 Asia/Shanghai

## 来源

- 论文：[In-Place Test-Time Training, arXiv:2604.06169v1](https://arxiv.org/abs/2604.06169v1)。方法及实验对照以已下载的完整原文为准。
- 官方仓库：[ByteDance-Seed/In-Place-TTT](https://github.com/ByteDance-Seed/In-Place-TTT)，固定 revision 为 `be2324829b0e91c8fd10a74d4b43714fde6676e1`。
- `upstream/` 是该 revision 的来源副本，不含 Git 元数据；官方训练、推理源码和配置未修改。
- 许可证：[Apache-2.0](upstream/LICENSE)。第三方归属见 [CREDITS](upstream/CREDITS) 和 [third_party_licenses/](upstream/third_party_licenses/)。
- 官方训练依赖 VeOmni revision `9b91e164bea9e17f17ed490aab5e076c2335ca25`，未复制到本目录；独立评估不安装 VeOmni。

## 本地接入与优化边界

| 位置 | 职责 |
|---|---|
| `upstream/` | 未修改的官方来源，保留训练、推理及导出入口 |
| `scripts/` | 本地数据契约、严格加载、单样本生成、原生评分、续评、合并及代码来源记录 |
| `eval.sh`、`scripts/evaluate_parallel.py` | 单卡／多卡独立进程调度、全局样本分配与执行记录，不修改模型计算 |
| `../evaluation_runtime.py` | LaCT 与 In-Place TTT 共用的进度显示和子进程退出清理 |
| `optimized/` | 独立的模型子类，压紧 TTT 尾部存储、可选 MLP 分块；不改 checkpoint keys、上游文件、全局 AutoModel 注册或 OpenCompass 安装 |
| `scripts/validate_optimization.py` | 原始／优化生成、chunk 边界和样本隔离的配对验证 |

`original` 调用官方模型及原 OpenCompass `HuggingFaceBaseModel.generate`。`optimized` 复用一次完整编码的输入，保持特殊 token、生成参数、解码和评分；缓存压紧不裁剪 attention KV 或 fast weights，MLP 分块保留原 TTT chunk 的先读后更新及尾段处理。

`eval.sh` 统一经启动器运行，默认 `original`，每卡一个进程且 batch size 固定为 1；各卡评估互斥样本，完成后从原始预测重新计分。`--mlp-chunk-size` 默认 `0`，`original` 仅接受 `0`；`optimized` 设为 `0` 时仅压紧 TTT 尾部缓存、保留原 MLP 计算形状，正值同时启用 MLP 分块，TTT 模型要求分块长度为 TTT chunk 的整数倍。完整运行指令、数据规模和产物见 [README](README.md)，验证与性能结论统一见[复现状态记录](../20261110_ttt_reproduction_status.md)。

本地 wrapper 使用本次最大评估长度作为 `max_seq_len`，通过 OpenCompass 实际支持的 `generation_kwargs` 和 `max_out_len=64` 传入生成设置；不使用原配置中不受支持的 `gen_config`／`engine_config`，也不把聊天模板参数 `enable_thinking` 传入 base 模型。每条完整编码输入加 64-token 预算必须在长度档位内，不截断、不自动改 RoPE 或添加 YaRN。

数据固定从本地 RULER 目录读取。产物保存在 `<output-root>/eval/<output-id>/ruler/`，默认 ID 为 checkpoint 目录名与上海时间的分钟时间戳。`evaluation_plan.json` 保存固定评估协议，`launcher_execution.json` 记录各次执行；各轮样本分配、日志、逐样本预测和合并结果分别存放在 `attempts/attempt-<编号>/`。

运行计划记录 Git commit／dirty/status、上游 revision 和实现路径；执行与逐题结果关联实际运行来源。续评显式使用原 ID，保留 checkpoint、长度、推理模式、分块及生成／评分协议，按全局样本 ID 补齐未完成记录，允许调整 GPU 数量和顺序。运行目录独占锁及历史活跃 worker 检查防止重复启动。

## 资源与原文差异

当前使用第三方 `zhongweixie` 的完整 Qwen3-4B 32K TTT／baseline checkpoint，两者模型卡记录使用 ProLong 训练 6000 steps。它们不是论文官方训练产物，32K 以外属于外推评估。

| 项目 | 原文 | 当前官方代码或资源 |
|---|---|---|
| 主要继续预训练模型 | Qwen3-4B-Base | 推荐配置为 Qwen3-8B／LLaMA-3.1-8B，缺少对应 4B 的完整实验配置 |
| 训练长度与预算 | 32K、约 20B tokens，再 128K、约 15B tokens | 推荐配置为 65,536 长度、20B 数据预算、5,000 steps，不等同原文两阶段设置 |
| 训练语料与权重 | 作者内部数据及论文训练产物 | 作者未公开；第三方权重可以评估，但不能直接视为原文模型 |
| Qwen3-4B 推理更新裁剪 | 附录规定更新范数阈值 `1e-5` | 官方 Qwen3 推理代码直接执行 `current_w + dw`，未实施该裁剪 |
| OpenCompass | 用于 RULER 评估 | 官方没有固定版本；本轮独立环境固定为 0.5.5 |

两个 checkpoint 的代表性真实生成一致，不代表 MLP 分块在任意输入上数值等价：部分合成输入出现 logits 和内部状态差异。仅尾部压紧的适用范围与全量分块应区分，正式结果应记录实际模式。
