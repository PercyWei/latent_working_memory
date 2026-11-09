# 20261108_In-Place TTT 官方代码来源

创建时间：20261108 17:05:19 Asia/Shanghai

最后修订时间：20261109 03:48:27 Asia/Shanghai

## 来源

- 论文：[In-Place Test-Time Training, arXiv:2604.06169v1](https://arxiv.org/abs/2604.06169v1)。本文使用已下载的完整原文核对方法及实验表格。
- 官方仓库：[ByteDance-Seed/In-Place-TTT](https://github.com/ByteDance-Seed/In-Place-TTT)。
- 固定 revision：`be2324829b0e91c8fd10a74d4b43714fde6676e1`。
- `upstream/` 为该 revision 的工作树副本，排除 Git 元数据；源码与官方配置未修改。
- 许可证：Apache-2.0，见 [upstream/LICENSE](upstream/LICENSE)。第三方归属见 [upstream/CREDITS](upstream/CREDITS) 和 [upstream/third_party_licenses/](upstream/third_party_licenses/)。
- 官方要求的 VeOmni revision：`9b91e164bea9e17f17ed490aab5e076c2335ca25`。VeOmni 是外部运行依赖，未复制到本目录。

## 本地接入

`eval.sh` 使用已激活方法环境中的 `python`，固定 `CUDA_VISIBLE_DEVICES=1`。`scripts/evaluate.py` 保存本次计划，并在同一进程直接调用原 OpenCompass `HuggingFaceBaseModel.generate`、`extract_non_reasoning_content` 和 RULER 的五类 evaluator。只替换数据来源与外围调度：从本地 NVIDIA 格式 JSONL 读取全部 13 项任务，不现场调用 OpenCompass 生成器，不改变模型或评分实现。

官方 `upstream/eval.sh` 从物理 GPU 5 开始分配到多张卡，保留作来源参考。本地入口按单进程、batch size 1 顺序评估；每条样本由上游 `generate` 创建独立 TTT cache。每项每长度 100 条，统一 64-token 输出、base completion 和原预测后处理。

本地 wrapper 的 `max_seq_len` 使用本次最大长度，不沿用原模型配置文件里固定的 131,072 运行上限。生成前校验完整编码输入加 64-token 预算，超限报错；不自动修改 checkpoint 的 RoPE 或添加 YaRN。因此 256K 可以作为现有 32K checkpoint 的外推条件，但不能视为论文 128K 继续训练模型的复现。

当前官方模型配置的 `gen_config`／`engine_config` 不是 OpenCompass `HuggingFaceBaseModel` 的生成接口；本地入口通过其实际支持的 `generation_kwargs` 传入 greedy、top-k／top-p 及 cache 设置，通过 `max_out_len=64` 控制输出长度。`enable_thinking` 是聊天模板参数，不传入 base 模型的 `generate`。

## 已发现的复现差异

| 项目 | 原文 | 当前官方代码或资源 |
|---|---|---|
| 主要继续预训练模型 | Qwen3-4B-Base | README 推荐 Qwen3-8B 与 LLaMA-3.1-8B；未提供对应 4B 的完整实验配置 |
| 训练长度与预算 | 32K、约 20B tokens；再 128K、约 15B tokens | 推荐配置使用 65,536 长度、20B 数据预算和 5,000 steps；不能直接视为原文两阶段设置 |
| 训练语料 | 作者自建的预训练与长上下文混合数据 | 未随仓库发布，README 要求自行提供 `content_split` 数据；未随仓库发布数据处理脚本 |
| 训练后权重 | 论文报告继续预训练及从头训练模型 | 本次取得的仓库未提供 TTT 训练后 checkpoint 下载链接；基础 Qwen 权重不能代替 TTT 权重 |
| Qwen3-4B 推理更新裁剪 | Appendix 规定更新范数阈值为 `1e-5` | 当前 `inference_model/hf_qwen3/modeling_qwen3.py` 的更新直接执行 `current_w + dw`，未实施该裁剪 |
| OpenCompass | 原文使用该框架评估 RULER | 仓库包含配置，但 README 和依赖声明没有固定 OpenCompass revision，需在 GPU 环境中核对并记录实际版本 |

这些差异尚未通过算法修改补齐。尤其不能把换用公开语料或当前 8B 推荐配置后的结果称为原文 4B 结果复现。
