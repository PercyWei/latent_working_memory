# 20260904 ICAE v1 PwC 结果检查计划（20260905 10:35:18 CST）

创建时间：20260904 16:09:49 CST（UTC+08:00）

最后修订时间：20260905 10:35:18 CST（UTC+08:00）

状态：工程检查已通过；GPU 0 和 1 正在分片运行 ICAE-128 PwC 全量生成

## 术语

- **ICAE v1**：以 Llama-2-7B-Chat 为基础模型、使用 128 个连续 memory slots 的公开版本。
- **Memory slot**：ICAE encoder 输出、可直接作为 LLM 输入 embedding 的连续向量；下文简称 slot。
- **PwC**：Prompt-with-Context 数据集，每条记录包含 `input`、`prompt` 和 `answer`。
- **Strict load**：使用 `load_state_dict(..., strict=True)` 加载 checkpoint，不忽略缺失键或多余键。
- **Evaluator drift（评估器漂移）**：评估模型版本、接口或 prompt 差异引起的评分变化，不等同于 ICAE 输出能力变化。

## 目标与边界

本轮直接对 PwC test 运行公开的 instruction-fine-tuned ICAE v1 checkpoint，获得完整、可续跑且可审计的 ICAE-128 predictions，并检查其生成质量与运行成本。

本轮不运行完整上下文基线，因此不能复现论文 Table 4 的成对 win、lose、tie 指标。Table 4 只作为论文背景，不作为本轮验收目标。若后续需要严格复现该表，必须另行确定作者未公开的完整上下文 baseline 实现。

不再执行 source-only facts、反事实样本、Prompt only、Zero memory、20 样本人工门槛或 200 样本 pilot。少量输出抽查仅用于确认批处理程序正常写入，不构成功能性验收。

本轮也不包含：

- 从头预训练或 instruction fine-tuning；
- ICAE v2；
- C-DIC 训练；
- 使用当前 fine-tuned checkpoint 验收 pretrained ICAE 的重建 BLEU、Exact Match、cross-entropy loss 或 text-continuation PPL。

## 固定输入

### 代码与资源

- 项目：`/data/bywei/projects/latent_working_memory`
- 当前运行 commit：`e838848459dc8281193c88702f455a78b17e389f`
- ICAE 上游 commit：`469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd`
- 基础模型：`/data/bywei/models/meta-llama/Llama-2-7b-chat-hf`
- ICAE checkpoint：`/data/bywei/projects/latent_working_memory/checkpoints/icae/v1/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt`
- PwC test：`/data/bywei/projects/latent_working_memory/data/raw/pwc/PwC_test.jsonl`
- uv cache：`/data/bywei/cache/uv`

运行 manifest 必须记录精确 Git commit、dirty status、模型与数据 revision、环境和生成参数。

## 已完成的工程检查

### 1. 环境

服务器环境已确认：

- Python `3.10.x`；
- PyTorch `2.0.1+cu118`，`torch.version.cuda == "11.8"`；
- Transformers `4.31.0`；
- 项目内定制 PEFT `0.4.0.dev0`；
- NVIDIA A800-SXM4-80GB，bfloat16 可用。

服务器驱动支持 CUDA 13.0，而 PyTorch wheel 内含 CUDA 11.8 runtime；两者不同是预期情况。迁移代码不编译自定义 CUDA extension。

### 2. Checkpoint

公开 checkpoint 包含 452 个键，其中 129 个 tensor 和 323 个标量 `0.0` 占位符；LoRA A/B 各 64 组，实际 rank 为 128；memory embedding 形状为 `[131, 4096]`，无 memory head。

加载时先从本地 Llama-2 基础模型补回占位键，再使用 `strict=True` 加载完整 state dict。missing keys 与 unexpected keys 均为空。

固定模型参数：

- `mem_size=128`；
- `model_max_length=512`；
- encoder 压缩路径启用 LoRA；
- decoder 生成路径不启用 encoder LoRA；
- bfloat16 推理。

### 3. 单样本推理

20260904 19:42 CST，物理 GPU 0 将 28-token 上下文压缩为 `[1, 128, 4096]` 的 bfloat16 memory；数值全部有限，并对随机代码 `ZETA-4827` 生成正确答案。两次 greedy generation 的 token IDs 完全一致。

ICAE v1 使用 `model.eos_id=1` 作为停止 token；使用 Llama tokenizer EOS `2` 会在正确答案后继续生成。

## PwC ICAE-128 全量生成

### 运行配置

- 条件：仅 ICAE-128；
- 上下文最大长度：512 tokens；
- memory：128 slots；
- 解码：greedy；
- 最大生成长度：512 tokens；
- 停止 token：`model.eos_id=1`；
- seed：42；
- GPU：仅物理 GPU 0 和 1；
- 分片：按 `sample_index % 2` 划分，GPU 0 处理 shard 0，GPU 1 处理 shard 1；
- batch size：1；
- `answer` 只用于评估，不进入 tokenizer 或生成模型。

每条 prediction 保存：

- sample ID、sample index、`input`、`prompt` 和 reference answer；
- 输出文本和 token IDs；
- 上下文、prompt 和输出 token 数；
- memory shape 与 dtype；
- 压缩时间、生成时间和峰值显存。

每个 shard 使用独立输出文件，并依据 sample ID 断点续跑。两个 shard 完成后按 `sample_index` 合并；若出现重复 ID、缺失 ID 或错误记录，先生成异常清单，不静默丢弃。

### 当前运行

运行目录：

```text
/data/bywei/projects/latent_working_memory/artifacts/icae/20260904_pwc_reproduction/20260904_2020_e838848/
```

主要文件：

```text
predictions_icae_shard0.jsonl
predictions_icae_shard1.jsonl
predictions_icae_shard0.errors.jsonl
predictions_icae_shard1.errors.jsonl
predictions_icae_shard0.manifest.json
predictions_icae_shard1.manifest.json
logs/icae_shard0.log
logs/icae_shard1.log
```

此前单卡生成的 379 条 ICAE 结果已经按奇偶索引写入两个 shard，后续任务从这些记录继续运行。完整上下文任务在生成 104 条后停止，其部分产物仅保留作运行记录，不进入本轮统计。

## 结果评估

### 基础统计

全量生成完成后首先报告：

- 成功数、失败数和最终 denominator；
- 空输出率、输出长度分布和触及 512-token 上限的比例；
- 压缩时间、生成时间、吞吐量和峰值显存；
- 按输入长度、prompt 长度和答案长度分层的结果。

### 参考答案诊断

可计算 normalized exact match、token F1 和 ROUGE-1/2/L，但这些指标只用于定位问题。PwC 回答具有开放性，文本重叠分数不能单独代表回答质量。

### 可选外部评估

若使用外部模型评估单个 ICAE 回答，应给评估器提供原文、prompt、reference answer 和 ICAE 输出，并要求判断指令遵循、正确性与事实一致性。该协议不同于论文的双回答成对比较，结果不得命名为论文 Table 4 复现。

必须记录评估模型提供方、完整模型 ID、调用日期、完整 prompt、采样参数、解析规则、失败重试和原始判断。不同评估模型导致的数值差异属于协议差异，不要求最终结果完全一致。

## 完成标准与排查条件

本轮完成要求：

- 两个 shard 覆盖全部 PwC test 样本，最终 denominator 明确；
- predictions、错误记录、运行配置和资源统计可追溯；
- 无 `answer` 泄漏；
- 汇总报告明确区分生成结果、自动文本指标和外部评估结果；
- 不把本轮单系统评估表述为论文 Table 4 的成对结果复现。

以下情况应停止当前运行并排查：

- 只能通过 `strict=False` 加载 checkpoint；
- LoRA rank 或 special tokens 无法与 checkpoint 对齐；
- 输出连续为空、重复或包含非法扩展 token；
- 出现 NaN、持续 OOM 或同一输入的 greedy token IDs 不一致；
- 同一 shard 连续出现 3 条样本错误。

## 最终产物

```text
/data/bywei/projects/latent_working_memory/artifacts/icae/20260904_pwc_reproduction/20260904_2020_e838848/
├── predictions_icae_shard0.jsonl
├── predictions_icae_shard1.jsonl
├── predictions_icae.jsonl
├── generation_metrics.json
├── evaluation_config.json
├── evaluation_decisions.jsonl
├── evaluation_metrics.json
├── resource_usage.json
└── logs/
```

执行顺序：完成两个 ICAE shard → 合并并检查 ID 覆盖 → 汇总运行与生成指标 → 计算参考答案诊断指标 → 视需要运行外部评估。

## 参考

- [ICAE 论文](https://arxiv.org/abs/2307.06945)
- [ICAE 官方仓库](https://github.com/getao/icae)
- [C-DIC 复现与仓库规划](20260903_c_dic_reproduction_repository_plan.md)
