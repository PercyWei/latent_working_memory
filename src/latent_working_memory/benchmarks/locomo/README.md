# 20261124_公共 LoCoMo 评估

创建时间：20261124 15:43:38 Asia/Shanghai

最后修订时间：20261124 15:43:38 Asia/Shanghai

公共入口支持 Full-context、ICAE v2 和 v3.1.1。数据使用 LoCoMo 完整历史及类别 1–4，共 10 段对话、1,540 道题。各方法在原有独立环境中加载模型，数据准备见[公共 benchmark 说明](../README.md)。

## 运行

从项目根目录执行，先激活所选方法的环境。Full-context 和 v3.1.1 使用主项目环境，ICAE 使用独立的 ICAE v2 环境。

```bash
# Full-context
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/benchmarks/scripts/eval_locomo.sh \
  --method full-context \
  --model-path ~/rivermind-data/models/Qwen3-4B-Instruct-2507 \
  --dtype bf16 --attention-implementation flash_attention_2 \
  --batch-size-per-gpu 2 --output-id qwen3-4b-full

# ICAE v2
source ~/rivermind-data/envs/icae-v2/bin/activate
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/benchmarks/scripts/eval_locomo.sh \
  --method icae --stage finetuned \
  --model-path ~/rivermind-data/models/Mistral-7B-Instruct-v0.2 \
  --checkpoint ~/rivermind-data/models/reproductions/icae/mistral_7b_ft_icae.safetensors \
  --batch-size-per-gpu 2 --output-id icae-ft-4x

# v3.1.1
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/benchmarks/scripts/eval_locomo.sh \
  --method v3.1.1 \
  --checkpoint "artifacts/v3-1/<run>/train/checkpoints/global_step_<N>" \
  --batch-size-per-gpu 2 --output-id v3-1-1-step-N
```

| 参数 | 默认值与约定 |
|---|---|
| `--method` | 必填，`full-context`、`icae`、`v3.1.1` |
| `--benchmark-data` | 含 `locomo10.json` 的目录，默认 `data/benchmarks/locomo` |
| `--output-root` | 默认 `artifacts/locomo` |
| `--output-id` | 默认 `<method>_<YYYYMMDD-HHMM>`，上海时间 |
| `--batch-size-per-gpu` | 每卡同时回答的问题数上限，默认 1 |
| `--max-conversations` | 只评估数据中前 N 段完整对话，默认全部 |
| `--resume` | 提供原 ID、数据和方法参数，仅补齐缺失问题，允许调整 GPU 数量和批量 |

方法专属参数通过 `--method <方法> --help` 查看。v3.1.1 支持 `--model-path` 指定迁移后的基座目录。Full-context 的 `--dtype` 支持 `fp32`、`bf16`；FP32 权重配合 FlashAttention 2 时使用 `--mixed-precision`。

## 执行与状态

- 同一对话的未完成问题分配给同一 GPU，正文只准备一次。卡内先完成当前对话，再释放状态并处理下一段。
- Full-context 复用正文前缀 KV；ICAE 和 v3.1.1 复用压缩结果及 Reader KV。同一批次的各题均从正文末状态独立开始。
- ICAE 和 v3.1.1 按问题 token 长度组批。Full-context 的滑动窗口模型也按等长后缀组批，以保持原始位置与注意力范围。
- 回答采用 greedy，最多 50 tokens。F1 按题平均，正文状态和记忆用量按对话统计。
- 恢复时先收集所有已完成题目，再分配剩余会话。部分完成的对话重新准备一次正文状态。中断写入的最后一行在恢复时截去。

方法原有 Full-context、v3.1.1 运行目录和看板入口也调用同一会话计算层。公共入口使用下述统一产物。

## 结果

文件位于 `<output-root>/eval/<output-id>/locomo/`。

| 文件 | 内容 |
|---|---|
| `evaluation_config.json` | 方法、模型身份、数据身份、选样及评分协议 |
| `progress/worker-*.jsonl` | 每次启动各 worker 的逐题记录，作为续评依据 |
| `progress/worker-*.log` | 各 worker 的模型加载和进度日志 |
| `work.json` | 本次启动的待评题目分配和 batch 设置 |
| `execution.json` | 最近启动的 Git commit、GPU、耗时和执行状态 |
| `details.jsonl` | 全部完成后按数据原顺序汇总的逐题结果 |
| `summary.json` | `metrics` 中的总体和分类 F1，`conversations` 中的会话统计 |

```bash
jq '{method, metrics}' artifacts/locomo/eval/qwen3-4b-full/locomo/summary.json
```

## 接入方法

在 [registry.py](registry.py) 注册方法模块。模块提供 `add_arguments(parser)`、`configuration(args)` 和 `load_adapter(settings)`，分别负责专属参数、可持久化的模型配置和实际加载。

适配器实现两个操作。

| 操作 | 约定 |
|---|---|
| `prepare_context(history, questions)` | 处理正文，返回 `PreparedConversation`，其中包含正文状态、编码后的问题、组批键和会话统计 |
| `answer_batch(state, questions, complete)` | 为各题创建独立状态分支，完成一题即回调 `complete(question, answer)`，正文状态保持只读 |

公共层拥有会话调度、分片、逐题持久化、恢复和评分。适配器负责 chat template、位置编码、模型窗口、缓存分支和生成计算。TTT 方法接入时需完整保存快速权重、动量和未完成更新块，并保持原来的问题阶段更新规则。

## 关键验证

`tests/locomo/test_shared_context.py` 在主项目环境检查 Full-context 和 v3.1.1 的独立计算与缓存结果一致性、问题顺序、批量、缓存只读及断点记录。ICAE 在其环境运行 `reproductions/icae/evaluation/tests/test_shared_context.py`，覆盖预训练与指令微调两阶段。

设置 `LOCOMO_TEST_DEVICE=cuda` 可在所选 GPU 上执行这些关键检查。A100 的验证还使用了两段完整 LoCoMo 对话的四道题，检查真实 Qwen3-4B 多卡调度、单卡续评和 v3.1.1 checkpoint 的 Reader 一致性。该小样本用于验证实现。
