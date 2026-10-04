# 20261003_v4 逐层 Attention Matching 记忆

创建时间：20261003 21:51:43 Asia/Shanghai（UTC+08:00）

最后修订时间：20261004 15:51:24 Asia/Shanghai（UTC+08:00）

本目录提供独立的流式 memory meta-training 入口：冻结 HF Llama／Qwen2／Qwen3，每层使用固定数量的 latent slots，达到 pending 阈值时以 attention matching 执行局部 SGD，外层根据写入后的 next-token CE 学习新增模块。完整结构见[框架笔记](../../../../notes/v4/20261003_attention_matching_meta_memory.md)。

## 数据与配置

训练和开发集使用同一种 UTF-8 JSONL 格式，每行恰好包含两个字符串字段，文件内 `id` 唯一：

```json
{"id": "document-0001", "text": "一条完整训练文本……"}
```

分词采用 `add_special_tokens=False`，不插入 BOS/EOS，不截断、不打包多个文档，也不自动过滤长度。每条文本 token 数必须在 `[pending_size + recent_size + 2, max_seq_length]` 内，不满足时明确报错。样例配置对应 386～4096 tokens；上面的短 JSON 仅说明格式。分词结果留在内存，各 epoch 复用，不保存派生 token 数据副本。

[配置样例](../../../../configs/v4/attention_matching/experiment.json) 是唯一配置入口，顶层只有 `model` 和 `training`。`train_file`、`dev_file`、`output_dir` 都相对于配置文件所在目录解析；模型使用 HF repo ID，或显式绝对本地路径。样例中的数据路径尚未创建，运行前应替换成实际 canonical JSONL 路径。`revision` 可指定固定模型 revision。

| 设置 | 样例值与语义 |
|---|---|
| `pending_size`／`recent_size` | 256／128；首次边界在第 384 个 token 处理完后 |
| `num_slots`／`memory_dim` | 每层 32 个 slots，每个 128 维 |
| `query_dim`／`value_dim`／`num_probes` | 64／128／16 |
| `query_mode` | `conditioned`：仅根据 pending 生成压缩查询；`fixed`：输入无关的可学习查询对照 |
| `query_normalization` | 默认 `none`：不归一化；`fixed_norm`：固定查询范数，与 `query_mode` 独立 |
| `read_query_norm`／`compression_query_norm` | 固定范数模式的读取／压缩查询目标范数；正数，省略或 `null` 时均取 `sqrt(query_dim)`，样例维度对应 8 |
| `inner_steps`／`inner_lr` | 1／0.1；测试时也使用同一局部更新 |
| `correction_scale` | 0.1；共享的固定 Q/O 修正尺度 |
| `backbone_dtype` | 默认 `float32`，可选 `bfloat16`；新增模块与 slots 为 FP32 |
| `global_batch_size` | 8；各 rank 一次处理一条轨迹，再累积全局 batch |

固定范数模式对读取查询，以及压缩生成器的初始查询和最终输出施加归一化；两个范数为固定超参数，不是可学习参数。默认模式直接使用这些查询，不施加范数约束。配置解析后将实际范数数值写入 run/checkpoint；`none` 模式下这两个数值不参与计算。

配置是开发起点，尚未由正式实验确认有效性或显存预算。改变查询生成或归一化设置时应使用新的输出目录和独立初始化；同一 run 的恢复要求配置、分词后数据、设备类型和 world size 一致。

## 运行

从项目根目录使用根 `.venv/` 对应的锁定环境。以下命令仅是操作说明，本次没有启动正式训练或连接 SwanLab。

双卡训练仅使用物理 GPU 0、1：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --frozen python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m latent_working_memory.v4.train \
  --config configs/v4/attention_matching/experiment.json --device cuda
```

单卡执行使用 `CUDA_VISIBLE_DEVICES=0`，直接调用相同模块；CPU 工程验证使用 `--device cpu`：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m latent_working_memory.v4.train \
  --config configs/v4/attention_matching/experiment.json --device cuda
```

`--stop-after-steps N` 停止于全局 optimizer step `N`，用于显式缩短本次执行，不修改 epoch 预算。新 run 要求输出目录不存在或为空。恢复时沿用同一配置、输出目录和启动方式，增加：

```text
--resume artifacts/v4/attention-matching_20261003/train/qwen3-0.6b_conditioned_20261003/checkpoints/step-000050.pt
```

恢复读取新增权重、optimizer、epoch／样本游标和各 rank RNG；冻结基座仍从配置加载。Run 身份还记录实际解析的 `resolved_model_revision`，tokenizer 保存在输出目录的 `tokenizer/` 中。Checkpoint 不包含正在进行的文本流状态，保存点位于完成一个全局 batch 后。局部 slots 在每条新文本开始时重新初始化，不在文档之间串联。

开发集在 `eval_every` 间隔及 epoch 结束时评估，仍执行相同边界写入，但不保留外层训练图。

单条流式 greedy 生成读取纯文本 prompt，从 checkpoint 恢复新增权重，并使用 run 中记录的模型 revision 与 tokenizer。输出 JSON 包含 prompt、生成文本及 token IDs：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m latent_working_memory.v4.inference \
  --checkpoint artifacts/v4/attention-matching_20261003/train/qwen3-0.6b_conditioned_20261003/checkpoints/step-000050.pt \
  --prompt-file data/attention-matching/prompt.txt \
  --output artifacts/v4/attention-matching_20261003/eval/greedy-sample.json \
  --max-new-tokens 128 --device cuda
```

这里的 prompt 路径与 checkpoint 需替换为实际文件。生成中继续执行相同的局部 TTT；prompt 与新增 tokens 的总长度不能超过基座绝对位置上限。当前没有正式 benchmark 评估管线。

| 产物 | 内容 |
|---|---|
| `run.json` | 解析后的配置、实际模型 revision、数据身份、设备类型与 world size |
| `tokenizer/` | 本次运行保存的 tokenizer，供后续推理复用 |
| `metrics.jsonl` | 真实 step 对应的 `train/*` 与 `dev/*` 增量记录 |
| `checkpoints/step-XXXXXX.pt` | 新增权重、optimizer、进度与 RNG |
| `training-result.json` | 完成状态、累计 steps／样本访问数／目标 token 数与最后 checkpoint |

## 实现契约与验证

`memory.py` 定义 source projection、共享 reader、查询生成与范数设置和可微局部写入；`model.py` 管理 Q/O hooks、原始 KV、源表示缓存及流式边界；`engine.py` 使用 verl `BaseEngine` 的 replicated/DDP 执行外层优化；`data.py`、`checkpoint.py` 与 `train.py` 分别处理输入、恢复和训练生命周期；`inference.py` 提供 checkpoint 的单条 greedy 生成入口。

- 仅 pending 的逐层源表示进入局部目标。源缓存每层最多包含 `B+W` 个表示；基座的原始 KV 不能替代这份缓存。
- 首次写入前禁用 memory 读取修正，但可学习初始 slots 参与第一次写入。边界更新之后，新 memory 影响后续前向；保留的 recent KV 不重算，绝对 position 不重置。
- 局部损失对 value 维求和、对压缩查询平均，再乘 0.5。内循环只求 slots 的偏导；训练时保留 teacher、generator 与跨边界状态的完整 meta-gradient。
- 外层仅统计首次写入后产生的 logits，并按全局有效目标 token 数归一化。边界前最后一个 logit 不计入写后监督，即使其 target 位于边界之后。
- 使用 eager attention 支持二阶反传；没有 FlashAttention、FSDP、跨边界截断或一阶 meta-gradient 近似。推理缓存有界不代表精确 meta-training 的显存有界。

工程测试入口：

```bash
uv run --frozen pytest tests/v4 -q
```

重点检查边界与 CE 偏移、两次以上写入的状态传递、删缓存后的绝对位置、局部内梯度与外层超梯度、以及 DDP 的尾批归一化。测试通过仅确认实现契约，不代表真实长文本实验已取得效果。

首个实现不包含独立历史约束、自适应扩容或 generator 自身的 TTT；generator 仅在离线外循环学习，部署参数冻结。当前目标只匹配新 pending 的读取行为，不能据此保证旧历史不会退化；该性质需要后续任务实验验证。
