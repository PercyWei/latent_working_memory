# 20261108_In-Place TTT 官方代码来源

创建时间：20261108 17:05:19 Asia/Shanghai

最后修订时间：20261116 22:12:29 Asia/Shanghai

## 来源与迁入范围

- 论文：[In-Place Test-Time Training, arXiv:2604.06169v1](https://arxiv.org/abs/2604.06169v1)。
- 官方仓库：[ByteDance-Seed/In-Place-TTT](https://github.com/ByteDance-Seed/In-Place-TTT)，固定 revision 为 `be2324829b0e91c8fd10a74d4b43714fde6676e1`。
- `source/in_place_ttt/` 保留该 revision 的训练、推理、配置和导出源码，内容保持原样，省略 Git 元数据。
- 许可证：[Apache-2.0](source/in_place_ttt/LICENSE)。第三方归属见 [CREDITS](source/in_place_ttt/CREDITS) 和 [third_party_licenses/](source/in_place_ttt/third_party_licenses/)。
- 官方训练依赖 VeOmni revision `9b91e164bea9e17f17ed490aab5e076c2335ca25`；本地独立评估使用 PyTorch、Transformers 和 OpenCompass，版本见[评估环境](../20261109_environment_setup.md)。

## 代码职责

| 位置 | 职责 |
|---|---|
| `source/in_place_ttt/` | 固定 revision 的官方源码 |
| `optimized/` | 独立模型子类，压紧尾段存储及可选 MLP 分块 |
| `scripts/runtime.py` | checkpoint 严格加载、原始与优化路径的生成 |
| `eval.sh`、`scripts/evaluate_parallel.py`、`scripts/evaluate.py` | 统一启动、单卡／多卡调度与 B1 worker |
| `scripts/records.py`、`scripts/merge_evaluation.py` | 数据与协议、全局样本 ID、记录收集和原生评分汇总 |
| `scripts/provenance.py`、`../evaluation_runtime.py` | 代码来源记录、终端进度和子进程退出清理 |
| `scripts/validate_optimization.py` | 生成、TTT chunk 边界及样本隔离的配对验证 |

## 推理实现改进

`original` 使用官方模型和原 OpenCompass `HuggingFaceBaseModel.generate`；`optimized` 在独立子类和本地生成入口中实现以下改进：

| 改进 | 实现 |
|---|---|
| TTT 尾段存储压紧 | 上游已有未完成 chunk 的尾段缓存；本地将仍引用完整前缀底层存储的尾段张量复制为独立存储，减少缓存持有的显存 |
| 单次编码 | 将完整编码结果复用于长度检查与生成；原始路径的预算检查和 OpenCompass 生成分别编码。特殊 token、生成参数、停止条件及解码设置保持原样 |
| 可选 MLP 分块 | 在 token 维分段计算 MLP，减少中间激活显存。`--mlp-chunk-size` 默认 `0`，正值启用；TTT 层要求其为 TTT chunk 长度的整数倍，记忆更新仍按原 TTT chunk 执行 |

每条样本生成时新建 attention／TTT cache，保持样本状态独立、先读后更新和尾段累积规则。推理采用完整 attention，保留全部 attention KV，fast weights 沿原规则更新；模型参数及 RoPE 配置沿用 checkpoint。

`optimized` 默认保持原 MLP 计算形状并压紧尾段，`original` 的 MLP 分块参数固定为 `0`。

本地 wrapper 使用所选最大评估长度作为 `max_seq_len`，将上游评估配置中的生成设置接入 OpenCompass 支持的 `generation_kwargs` 和 `max_out_len=64`；输入采用 base completion。模型验证、数值差异与性能结果统一见[复现状态记录](../20261110_ttt_reproduction_status.md#in-place-ttt)。

## 评估流程改进

- **统一入口。** 单卡和多卡均由 `eval.sh` 启动，每卡一个 B1 进程。RULER 数据路径固定在 `scripts/records.py`。
- **按样本续评。** 以长度、任务、原始行构成全局 ID，重新分配未完成样本。模型、数据及生成／评分协议固定，GPU 数量和顺序可调整。
- **产物与溯源。** 结果保存到 `<output-root>/eval/<output-id>/ruler/`，各轮记录放在 `attempts/`。保存完整生成 IDs、耗时和峰值显存，以及 Git 状态、上游 revision、实现路径和依赖版本。
- **执行控制。** 先核对协议和历史记录，再创建本轮记录；目录锁与存活 worker 检查协调同目录执行。各 worker 在生成前核对共享 runtime。
- **原生生成与评分。** `input + answer_prefix` 使用 greedy 生成最多 64 tokens，保留 EOS，并检查总长度预算。合并时对每个任务／长度的完整预测调用 OpenCompass evaluator，再计算 13 项任务的等权平均。

具体命令、参数与输出文件见 [README](README.md)，环境版本见[评估环境](../20261109_environment_setup.md)。

## 公开权重与论文差异

当前使用第三方 `zhongweixie` 的完整 Qwen3-4B 32K TTT／baseline checkpoint，模型卡记录采用 ProLong 训练 6,000 steps。它们与论文训练产物不同，超过 32K 的结果按长度外推报告，评估沿用 checkpoint 的 RoPE 配置。

| 项目 | 原文 | 当前官方代码或资源 |
|---|---|---|
| 主要继续预训练模型 | Qwen3-4B-Base | 推荐配置为 Qwen3-8B／LLaMA-3.1-8B，缺少对应 4B 的完整实验配置 |
| 训练长度与预算 | 32K、约 20B tokens，再 128K、约 15B tokens | 推荐配置为 65,536 长度、20B 数据预算、5,000 steps |
| 训练语料与权重 | 作者内部数据及论文训练产物 | 作者未公开；当前使用上述第三方权重 |
| Qwen3-4B 推理更新裁剪 | 附录规定更新范数阈值 `1e-5` | 官方 Qwen3 推理代码直接执行 `current_w + dw` |
| OpenCompass | 用于 RULER 评估 | 官方没有固定版本；本轮独立环境固定为 0.5.5 |
