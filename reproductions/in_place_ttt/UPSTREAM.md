# 20261108_In-Place TTT 官方代码来源

创建时间：20261108 17:05:19 Asia/Shanghai

最后修订时间：20261117 01:10:58 Asia/Shanghai

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
| `optimized/model.py` | 独立模型子类、生成入口和可选 MLP 分块 |
| `optimized/padding.py` | 真实位置、独立有效 KV、原生 SDPA 分组及结束行的状态冻结 |
| `optimized/ttt.py` | 原 TTT 公式的分组计算、独立尾段和适配权重 |
| `scripts/runtime.py` | checkpoint 严格加载、原始与优化路径的生成 |
| `eval.sh`、`scripts/evaluate_parallel.py`、`scripts/evaluate.py` | 统一启动、单卡／多卡调度与各卡批量评估 |
| `scripts/batching.py` | 各长度 batch 上限、长度排序与相邻组批 |
| `scripts/records.py`、`scripts/merge_evaluation.py` | 数据与协议、全局样本 ID、记录收集和原生评分汇总 |
| `scripts/provenance.py`、`../evaluation_runtime.py` | 代码来源记录、终端进度和子进程退出清理 |
| `scripts/validate_optimization.py` | 生成、TTT chunk 边界及样本隔离的配对验证 |

## 推理实现改进

`original` 使用官方模型和原 OpenCompass `HuggingFaceBaseModel.generate`；`optimized` 在独立子类和本地生成入口中实现以下改进：

| 改进 | 实现 |
|---|---|
| 混长 batch | 单样本、等长和左 padding 输入统一调度；每条样本独立维护真实位置、attention KV、TTT 尾段及适配权重 |
| 分组计算 | QKV 投影和 FFN 批量执行；query／key 长度相同的样本组成 attention 组，调用原生 SDPA；TTT 按读出长度、是否更新、是否已有适配权重及其 dtype 分组，不要求所有混长行同时更新 |
| TTT 尾段存储压紧 | 上游已有未完成 chunk 的尾段缓存；本地将仍引用完整前缀底层存储的尾段张量复制为独立存储，减少缓存持有的显存 |
| 单次编码 | 将完整编码结果复用于长度检查与生成；原始路径的预算检查和 OpenCompass 生成分别编码。特殊 token、生成参数、停止条件及解码设置保持原样 |
| 可选 MLP 分块 | 在 token 维分段计算 MLP，减少中间激活显存。`--mlp-chunk-size` 默认 `0`，正值启用；TTT 层要求其为 TTT chunk 长度的整数倍，记忆更新仍按原 TTT chunk 执行 |

每批生成时新建 attention／TTT cache，各行状态独立。TTT 在完整 chunk 的读出后更新，更新后的权重用于后续 chunk；未完成尾段继续累积。卷积保留每个 TTT chunk 内的零填充边界，不跨 chunk 卷积。padding 不进入有效 KV 或 TTT 数据，不推进真实位置、尾段和更新计数。推理采用完整 attention，保留全部有效 attention KV，模型参数及 RoPE 配置沿用 checkpoint。

保留原 SDPA 后端，不切换为 varlen FlashAttention。批量路径支持与序列长度无关的 `default`、`linear`、`yarn` 和 `llama3` RoPE；`dynamic`、`longrope` 不在支持范围内。HF 原生停止条件继续生效，各行在首次 EOS 或其他停止条件触发后冻结真实状态；生成 IDs 按各行首次结束长度截取，保留 EOS。

`--inference-mode` 默认 `original`，保留上游 B1 生成路径；`optimized` 支持 TTT 和 baseline 的批量生成。`--mlp-chunk-size` 默认 `0`，不对 MLP 额外分块；`original` 仅接受 `0`。

本地 wrapper 使用所选最大评估长度作为 `max_seq_len`，将上游评估配置中的生成设置接入 OpenCompass 支持的 `generation_kwargs` 和 `max_out_len=64`；输入采用 base completion。模型验证、数值差异与性能结果统一见[复现状态记录](../20261110_ttt_reproduction_status.md#in-place-ttt)。

## 评估流程改进

- **统一入口与组批。** 单卡和多卡均由 `eval.sh` 启动，每卡一个进程。各 RULER 长度内按真实 token 数稳定降序，相邻组批并左 padding；`--batch-size` 默认 `1`，`--batch-size-by-length` 覆盖指定长度的上限。RULER 数据路径固定在 `scripts/records.py`。
- **按样本续评。** 以长度、任务、原始行构成全局 ID，重新分配未完成样本。模型、数据及生成／评分协议固定，GPU 数量、顺序和 batch 设置可调整。
- **产物与溯源。** 结果保存到 `<output-root>/eval/<output-id>/ruler/`，各轮记录放在 `attempts/`。保存完整生成 IDs、耗时和峰值显存，以及 Git 状态、上游 revision、实现路径和依赖版本。
- **执行控制。** 先核对协议和历史记录，再创建本轮记录；目录锁与存活 worker 检查协调同目录执行。各 worker 在生成前核对共享 runtime。
- **原生生成与评分。** `input + answer_prefix` 使用 greedy 生成最多 64 tokens，保留各行 EOS 和原停止条件，并检查总长度预算。合并时对每个任务／长度的完整预测调用 OpenCompass evaluator，再计算 13 项任务的等权平均。

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
