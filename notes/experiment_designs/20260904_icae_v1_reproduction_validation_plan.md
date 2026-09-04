# 20260904 ICAE v1 复现检查计划（20260904 16:09:49 CST）

创建时间：20260904 16:09:49 CST（UTC+08:00）

最后修订时间：20260904 19:21:37 CST（UTC+08:00）

状态：服务器环境检查已通过；checkpoint 加载与推理入口待验证

## 术语

- **ICAE v1**：以 Llama-2-7B-Chat 为基础模型、使用 128 个连续 memory slots 的公开版本。
- **Memory slot**：ICAE encoder 输出、可直接作为 LLM 输入 embedding 的连续向量；本文也简称 slot。
- **PwC**：Prompt-with-Context 数据集，每条记录包含 `input`、`prompt` 和 `answer`。
- **Strict load**：使用 `load_state_dict(..., strict=True)` 加载 checkpoint，不忽略缺失键或多余键。
- **Smoke test（冒烟测试）**：只验证最短端到端路径和明显错误，不等同于论文结果复现。
- **Functional validation（功能验证）**：通过对照和反事实输入确认输出确实依赖压缩后的上下文，而不只是语言模型先验。

## 目标与边界

本轮目标是确认迁移后的 ICAE v1 能在服务器上正确加载公开 checkpoint，将不超过 512 tokens 的上下文压缩为 128 个 slots，并根据 PwC prompt 生成受上下文约束的回答。

复现分为三级：

1. **工程复现**：环境、代码和 checkpoint 能无歧义加载并完成前向计算；
2. **功能复现**：ICAE 输出稳定优于无上下文或无有效 memory 的对照；
3. **结果复现**：在 PwC test 上使用可审计的评估流程，与论文报告的趋势和指标比较。

当前不包含：

- 从头预训练或 instruction fine-tuning；
- ICAE v2；
- C-DIC 训练；
- 仅凭若干流畅输出宣称复现成功。

公开文件 `llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt` 是 instruction-fine-tuned checkpoint。论文中 pretrained ICAE 的重建 BLEU、Exact Match、cross-entropy loss 和 text-continuation PPL 不能直接用该 checkpoint 验收；若要复现这些结果，需要另行取得对应的 pretrained checkpoint 和数据处理流程。

## 固定输入

### 代码

- 项目：`/data/bywei/projects/latent_working_memory`
- 当前计划基线：`bfdf06a9ab3f271297e09c968122c43ac576833e`
- ICAE 上游 commit：`469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd`
- 迁移说明：[UPSTREAM.md](../../reproductions/icae/UPSTREAM.md)
- 环境说明：[README.md](../../reproductions/icae/README.md)

实际运行可以使用基线的后继 commit，但必须在 run manifest 中记录精确 commit 和 dirty status。服务器仓库的 `origin` 应为：

```text
https://github.com/PercyWei/latent_working_memory.git
```

### 运行路径

- 基础模型：`/data/bywei/models/meta-llama/Llama-2-7b-chat-hf`
- ICAE checkpoint：`/data/bywei/checkpoints/icae/v1/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt`
- PwC train：`/data/bywei/datasets/sggetao/PwC/PwC_train.jsonl`
- PwC test：`/data/bywei/datasets/sggetao/PwC/PwC_test.jsonl`
- uv cache：`/data/bywei/cache/uv`

## 检查流程

### 1. 代码与环境

```bash
cd /data/bywei/projects/latent_working_memory

git remote -v
git status --short
git rev-parse HEAD

export HF_HOME="/data/bywei/cache/huggingface"
export UV_CACHE_DIR="/data/bywei/cache/uv"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv sync --project reproductions/icae --frozen
uv run --project reproductions/icae --no-sync icae-check-environment
uv run --project reproductions/icae --no-sync pytest -q reproductions/icae/tests
```

环境验收值：

- Python `3.10.x`；
- PyTorch `2.0.1+cu118`；
- `torch.version.cuda == "11.8"`；
- Transformers `4.31.0`；
- PEFT `0.4.0.dev0`，来源为项目中的 `vendor/peft`；
- `torch.cuda.is_available()` 为真；
- GPU 为 NVIDIA A800-SXM4-80GB；
- bfloat16 可用。

服务器 CUDA 13.0-capable driver 与 wheel 内 CUDA 11.8 runtime 不相等是预期情况。这里不编译自定义 CUDA extension。

### 2. Checkpoint 结构与参数确认

加载 checkpoint 到 CPU，记录：

- 顶层对象类型；
- state-dict key 数量；
- 每组 LoRA A/B、memory-token embedding 和可选 memory head 的形状与 dtype；
- checkpoint 中是否包含基础模型权重、零尺寸占位参数或仅 trainable 参数；
- 由 LoRA A/B 张量形状推断的实际 rank。

论文默认 LoRA rank 为 128，但公开 v1 代码默认值为 64。必须以公开 checkpoint 的实际张量形状确定实例化参数，并在 manifest 中记录；不能为了消除报错而使用 `strict=False`。

模型实例化后要求：

- 基础模型从本地目录加载，不访问网络；
- `mem_size=128`；
- `model_max_length=512`；
- checkpoint strict load 无 missing keys 和 unexpected keys；
- 记录总参数量、trainable 参数量和 checkpoint 加载后的 dtype；
- decoder 路径不启用 encoder LoRA，encoder 压缩路径显式启用 LoRA。

若 strict load 失败，先判断是 rank、special-token 数量、key 命名还是 checkpoint 保存方式不一致；不得直接跳过错误。

### 3. 受测试的推理入口

上游 `ft_inference.py` 不能直接作为复现入口：它引用未发布的 `instruct_ft_tokenize_function`，并含有 `memopry_mask` 拼写错误。应在 `src/icae_repro/` 中实现独立且受测试的推理入口，保留上游示例不变。

推理链路应固定为：

1. tokenizer 将 `input` 截断到最多 512 tokens；
2. 在上下文后附加 128 个 memory token IDs；
3. encoder 使用 LoRA 前向，读取 memory positions 对应的最后层 hidden states；
4. 得到形状 `[batch, 128, 4096]` 的 memory tensor；
5. decoder 输入为 memory tensor 与 `[FT] prompt [FT]` embeddings；
6. 使用 KV cache 做 greedy decoding；
7. 遇到 EOS、非法扩展 token 或 `max_new_tokens` 时停止。

每次运行检查：

- memory、logits 和 KV cache 中无 NaN/Inf；
- 输入的 `answer` 字段不进入 tokenizer 或模型；
- 推理处于 `eval()` 和 inference/no-grad mode；
- 固定输入重复运行得到相同 token IDs；
- 保存输入 token 数、memory slot 数、输出 token 数、压缩耗时、生成耗时和峰值显存。

### 4. 最小功能测试

先构造 20 组 source-only facts，每组包含模型无法依赖常识猜出的随机实体和值。例如上下文给出唯一编号，prompt 只询问该编号。每组另构造只修改目标值的反事实版本。

比较四个条件：

| 条件 | 目的 |
|---|---|
| Prompt only | 测量无上下文先验 |
| Zero memory | 排除 prompt 模板或 special token 自身的作用 |
| ICAE 128 slots | 待验证对象 |
| Full context | 确认基础模型和 prompt 模板能够完成任务 |

冒烟验收目标：

- 20 组全部完成，无 OOM、异常退出或空输出；
- ICAE 和 Full context 的目标事实准确率至少为 90%；
- Prompt only 与 Zero memory 不应系统恢复随机目标值；
- 修改上下文中的目标值后，ICAE 回答随之正确改变；
- 同一输入重复运行 3 次，生成 token IDs 完全一致。

若 Full context 也失败，优先检查 prompt 序列化和 tokenizer；若 Full context 成功而 ICAE 与 Zero memory 接近，优先检查 checkpoint、memory positions、LoRA 开关和 decoder 输入。

### 5. PwC 分阶段检查

#### 20 样本人工审计

按 tokenizer 后的 `input` 长度固定抽取短、中、长样本，覆盖不同 prompt 类型。逐条保存：

- 原始 input、prompt、reference answer；
- 四个条件的输出；
- 是否引用上下文中的具体事实；
- 是否存在与输入冲突的内容；
- ICAE 相对 Prompt only、Zero memory 和 Full context 的人工成对判断。

#### 200 样本 pilot

冻结样本 ID、模板、生成参数和随机种子后，报告：

- 非空输出率和失败率；
- normalized exact match、token F1、ROUGE-1/2/L，仅作为诊断指标；
- ICAE 对 Prompt only、Zero memory、Full context 的成对 win/tie/lose；
- 按输入长度和答案长度分层的结果；
- 压缩、prefill、generation 延迟与峰值显存。

PwC 答案具有开放性，自动文本重叠不能单独作为复现结论。任何 LLM judge 必须记录模型版本、完整 prompt、采样参数、失败重试和原始判定。

#### 全量 test

仅在 20 样本和 200 样本检查通过后运行 18,146 条 PwC test。先估算总 GPU 时间和输出空间，支持断点续跑，并保证每个 sample ID 只计一次。

## 论文结果参照

论文 Table 4 中，Llama-2-7B-Chat ICAE、`k=128` 相对使用原始上下文的 Llama-2-7B-Chat，GPT-4 成对判断为：

- win：19.6%；
- lose：45.4%；
- tie：35.0%；
- win + tie：54.6%。

这些数字是结果复现的参照，不是工程冒烟阈值。只有测试集、输入模板、生成配置、judge prompt 和 judge 模型版本均可比时，才能讨论数值复现；否则只报告方向和差异来源。

论文中的 Llama-2-7B pretrained ICAE `512→128` 重建 BLEU 99.5、loss 0.009 属于另一 checkpoint/任务设置，不用于验收当前 fine-tuned checkpoint。

## 通过标准与停止条件

### 工程复现通过

- 环境版本符合记录；
- checkpoint 参数由实际张量确定并 strict load；
- memory tensor 形状和 dtype 正确，无非有限值；
- 单样本端到端推理确定性通过；
- 运行产物包含完整 manifest 和资源统计。

### 功能复现通过

- source-only fact 与反事实测试表明输出受 memory 内容控制；
- ICAE 明显优于 Prompt only 和 Zero memory；
- PwC 小样本中能够稳定利用上下文，而非只生成通用回答；
- 结论在固定样本和配置下可重复。

### 立即停止并排查

- 只能通过 `strict=False` 加载；
- LoRA rank 或 special tokens 无法从 checkpoint 对齐；
- Full context 与 Prompt only 同样失败；
- ICAE 与 Zero memory 行为接近；
- 输出不随反事实上下文改变；
- 出现 NaN、非法 token、持续 OOM 或相同输入非确定性。

在工程复现和功能复现通过前，不开始 C-DIC GPU 训练，也不将流畅示例输出写成论文复现结论。

## 运行产物

统一保存到：

```text
/data/bywei/projects/latent_working_memory/artifacts/icae/20260904_validation/<run_id>/
├── manifest.json
├── environment.json
├── checkpoint_schema.json
├── config.resolved.json
├── sample_ids.json
├── predictions.jsonl
├── metrics.json
├── resource_usage.json
└── logs/
```

`manifest.json` 至少记录 Git commit、dirty status、`uv.lock` hash、Python、PyTorch、CUDA runtime、driver、GPU、模型与数据路径、LoRA rank、memory size、dtype、生成参数和运行时间。

建议执行顺序：环境 → checkpoint schema → strict load → 单样本推理 → source-only fact 对照 → 20 条 PwC → 200 条 pilot → 决定是否全量运行。

## 参考

- [ICAE 论文](https://arxiv.org/abs/2307.06945)
- [ICAE 官方仓库](https://github.com/getao/icae)
- [C-DIC 复现与仓库规划](20260903_c_dic_reproduction_repository_plan.md)
