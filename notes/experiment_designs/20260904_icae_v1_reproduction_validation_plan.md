# 20260904 ICAE v1 论文结果复现计划（20260904 20:07:10 CST）

创建时间：20260904 16:09:49 CST（UTC+08:00）

最后修订时间：20260904 20:07:10 CST（UTC+08:00）

状态：环境、checkpoint strict load 和单样本推理已通过；下一步直接复现 PwC 论文结果

## 术语

- **ICAE v1**：以 Llama-2-7B-Chat 为基础模型、使用 128 个连续 memory slots 的公开版本。
- **Memory slot**：ICAE encoder 输出、可直接作为 LLM 输入 embedding 的连续向量；下文简称 slot。
- **PwC**：Prompt-with-Context 数据集，每条记录包含 `input`、`prompt` 和 `answer`。
- **Strict load**：使用 `load_state_dict(..., strict=True)` 加载 checkpoint，不忽略缺失键或多余键。
- **Full-context baseline（完整上下文基线）**：不压缩 `input`，直接将原始上下文提供给同一基础模型。
- **Pairwise judge（成对评估器）**：比较 ICAE 与完整上下文基线回答，并输出 win、lose 或 tie 的外部模型。
- **Evaluator drift（评估器漂移）**：评估模型版本、服务实现或 prompt 不同导致的评分变化，不等同于被评估模型能力变化。

## 目标与边界

本轮直接复现论文在 PwC test 上对 instruction-fine-tuned ICAE v1 的结果。核心任务是使用公开 checkpoint 生成 ICAE 与完整上下文基线的成对回答，再复现论文的评估流程并解释差异来源。

复现分为两层：

1. **工程复现**：环境、代码和 checkpoint 能无歧义加载，压缩与生成链路正确运行；该层已通过。
2. **结果复现**：完成 PwC test 全量生成，使用可审计的成对评估流程，与论文 Table 4 的结果比较。

不再执行 source-only facts、反事实样本、Prompt only、Zero memory、20 样本人工门槛或 200 样本 pilot。批量入口完成后直接运行 PwC test；少量样本仅可用于检查输出格式和断点续跑，不设置功能性验收阈值。

本轮不包含：

- 从头预训练或 instruction fine-tuning；
- ICAE v2；
- C-DIC 训练；
- 使用当前 fine-tuned checkpoint 验收 pretrained ICAE 的重建 BLEU、Exact Match、cross-entropy loss 或 text-continuation PPL。

公开文件 `llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt` 是 instruction-fine-tuned checkpoint。论文中的 pretrained ICAE `512→128` 重建结果属于另一 checkpoint 和任务设置。

## 固定输入

### 代码

- 项目：`/data/bywei/projects/latent_working_memory`
- 本次计划修订前基线：`6e2acda67ecf9da65e3c9a26159a4f45b1454b8f`
- ICAE 上游 commit：`469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd`
- 迁移说明：[UPSTREAM.md](../../reproductions/icae/UPSTREAM.md)
- 环境说明：[README.md](../../reproductions/icae/README.md)

实际运行必须在 manifest 中记录精确 Git commit 和 dirty status。服务器仓库的 `origin` 应为：

```text
https://github.com/PercyWei/latent_working_memory.git
```

### 运行路径

- 基础模型：`/data/bywei/models/meta-llama/Llama-2-7b-chat-hf`
- ICAE checkpoint：`/data/bywei/checkpoints/icae/v1/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt`
- PwC test：`/data/bywei/datasets/sggetao/PwC/PwC_test.jsonl`
- uv cache：`/data/bywei/cache/uv`

## 已完成的工程检查

### 1. 环境

服务器环境已确认：

- Python `3.10.x`；
- PyTorch `2.0.1+cu118`，`torch.version.cuda == "11.8"`；
- Transformers `4.31.0`；
- 项目内定制 PEFT `0.4.0.dev0`；
- NVIDIA A800-SXM4-80GB，bfloat16 可用。

服务器驱动支持 CUDA 13.0，而 PyTorch wheel 内含 CUDA 11.8 runtime；两者不同是预期情况。迁移代码不编译自定义 CUDA extension。

本项目实验默认只使用物理 GPU 0 和 1；单进程任务优先使用 GPU 0。环境或依赖未变化时，无需重复执行环境检查。

### 2. Checkpoint

公开 checkpoint 已确认包含 452 个键，其中 129 个 tensor 和 323 个标量 `0.0` 占位符；LoRA A/B 各 64 组，实际 rank 为 128；memory embedding 形状为 `[131, 4096]`，无 memory head。

加载流程为：先从本地 Llama-2 基础模型补回 `0.0` 占位键，再使用 `strict=True` 加载完整 state dict。检查结果中 missing keys 与 unexpected keys 均为空。

固定模型参数：

- `mem_size=128`；
- `model_max_length=512`；
- encoder 压缩路径启用 LoRA；
- decoder 生成路径不启用 encoder LoRA；
- bfloat16 推理。

### 3. 单样本推理

受测试的推理链路已经通过：

1. 将 `input` 截断到最多 512 tokens；
2. 附加 128 个 memory token IDs；
3. encoder 输出 `[batch, 128, 4096]` memory tensor；
4. decoder 接收 memory tensor 与 `[FT] prompt [FT]` embeddings；
5. 使用 KV cache 和 greedy decoding；
6. 遇到 `model.eos_id=1`、扩展 token 或生成长度上限时停止。

20260904 19:42 CST，物理 GPU 0 上的样本将 28-token 上下文压缩为 `[1, 128, 4096]` 的 bfloat16 memory；数值全部有限，对随机代码 `ZETA-4827` 生成了正确答案，两次生成的 token IDs 完全一致。结果保存在服务器：

```text
artifacts/icae/20260904_validation/smoke_single/result.json
```

## PwC 论文结果复现

### 1. 成对生成条件

对 PwC test 的每条有效记录生成两份回答：

| 条件 | 模型输入 |
|---|---|
| ICAE-128 | 将 `input` 压缩为 128 个 slots，再与 `[FT] prompt [FT]` 拼接 |
| Full context | 将未压缩的 `input` 与 `prompt` 直接提供给 Llama-2-7B-Chat |

两个条件必须使用同一基础模型、同一数据顺序和相同的解码策略。ICAE 路径按上游示例使用 greedy decoding，最大生成 512 tokens，并在 token `1` 停止。完整上下文基线的 prompt 模板、截断规则和停止条件应优先按论文或公开材料实现；无法确认的细节必须写入运行配置，不得默认为与 ICAE 的 `[FT]` 模板相同。

`answer` 只作为评估参考，不得进入生成输入。每条 prediction 至少保存：

- sample ID、`input`、`prompt` 和 reference answer；
- 条件名称、生成文本和 token IDs；
- 输入、压缩状态和输出 token 数；
- 压缩、prefill、generation 时间与峰值显存；
- 异常类型和是否成功完成。

全量生成应支持断点续跑，以 sample ID 去重。若使用 GPU 0 和 1 并行，每个进程处理互斥的样本集合，并分别记录设备和运行配置。

### 2. 评估协议

论文 Table 4 使用 GPT-4 对 ICAE 与完整上下文回答进行成对判断。评估过程与回答生成解耦：先冻结两套 predictions，再运行 judge。这样可以在不重新执行 7B 推理的情况下更换或复核评估模型。

若能够获得论文使用的 judge prompt 和等价 GPT-4 版本，则按原协议复现；否则将本次评估标记为 **protocol approximation（协议近似）**，并记录：

- 服务提供方、完整模型 ID、调用日期和接口版本；
- 完整 judge prompt、回答排列顺序和输出解析规则；
- temperature、最大输出长度、重试策略和失败样本；
- win、lose、tie 的原始计数、有效 denominator、比例和置信区间。

为识别位置偏差，可在协议近似结果之外增加 A/B 与 B/A 顺序互换检查，但不得将其与论文原始协议结果混为同一指标。

Normalized exact match、token F1 和 ROUGE-1/2/L 可作为补充诊断，不作为论文主结果的替代。PwC 回答具有开放性，文本重叠分数不能单独决定复现是否成功。

### 3. 论文参照

论文 Table 4 中，Llama-2-7B-Chat ICAE、`k=128` 相对完整上下文 Llama-2-7B-Chat 的 GPT-4 成对结果为：

- win：19.6%；
- lose：45.4%；
- tie：35.0%；
- win + tie：54.6%。

上述数值用于比较，不作为必须逐项达到的硬阈值。外部 judge 的模型版本、prompt、服务实现和随机性均可能造成系统偏移；当这些条件无法与论文完全一致时，只比较总体趋势、差异幅度和主要失败类型，并明确标注评估协议差异。

## 通过标准与排查条件

### 工程复现通过

- checkpoint 参数由实际结构确定并完成 strict load；
- memory tensor 形状与 dtype 正确，无 NaN/Inf；
- 单样本端到端推理和确定性检查通过；
- 批量运行不把 `answer` 泄漏给生成模型。

### 结果复现完成

- PwC test 的 ICAE 与完整上下文回答均已生成，失败样本和最终 denominator 明确；
- predictions、运行配置、资源统计和错误记录可追溯；
- judge 决策能够追溯到具体模型、prompt 和原始输出；
- 报告论文参照值、本次结果及其协议差异，不要求最终比例完全一致。

若结果与论文显著不同，不直接判定复现失败。先区分生成差异与评估器差异，再检查 prompt 序列化、截断长度、解码上限、EOS、基础模型版本和 judge 协议。

以下情况应立即停止当前运行并排查：

- 只能通过 `strict=False` 加载 checkpoint；
- LoRA rank 或 special tokens 无法与 checkpoint 对齐；
- 输出大面积为空、重复或包含非法扩展 token；
- 出现 NaN、持续 OOM 或同一输入的 greedy token IDs 不一致；
- judge 输出无法稳定解析，或失败样本未单独报告。

## 运行产物

统一保存到：

```text
/data/bywei/projects/latent_working_memory/artifacts/icae/20260904_pwc_reproduction/<run_id>/
├── manifest.json
├── environment.json
├── checkpoint_schema.json
├── config.resolved.json
├── predictions_icae.jsonl
├── predictions_full_context.jsonl
├── generation_metrics.json
├── judge_config.json
├── judge_decisions.jsonl
├── evaluation_metrics.json
├── resource_usage.json
└── logs/
```

`manifest.json` 至少记录 Git commit、dirty status、Python、PyTorch、CUDA runtime、driver、GPU、模型与数据 revision、LoRA rank、memory size、dtype、生成参数和运行时间。

执行顺序：确认已通过的工程检查仍适用于当前 commit → 实现并测试 PwC 批量生成入口 → 直接完成 ICAE 与完整上下文的全量生成 → 冻结 predictions → 运行外部 judge → 对照论文结果并记录协议差异。

## 参考

- [ICAE 论文](https://arxiv.org/abs/2307.06945)
- [ICAE 官方仓库](https://github.com/getao/icae)
- [C-DIC 复现与仓库规划](20260903_c_dic_reproduction_repository_plan.md)
