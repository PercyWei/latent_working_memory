# 20261003_v4 逐层 Attention Matching 记忆

创建时间：20261003 21:51:43 Asia/Shanghai（UTC+08:00）

最后修订时间：20261013 17:23:57 Asia/Shanghai（UTC+08:00）

本目录提供独立的流式 memory meta-training 入口：冻结 HF Llama／Qwen2／Qwen3，每层使用固定数量的 latent slots，达到 pending 阈值时以 attention matching 执行局部 SGD，外层根据写入后的监督 token CE 学习新增模块。完整结构见[框架笔记](../../../notes/v4/20261003_attention_matching_meta_memory.md)。

## 数据

### 格式约定

- UTF-8 JSONL，每行一篇完整正文及其 QA；同一文件内正文唯一。
- 顶层仅包含 `text`、`qas`，每个 QA 仅包含 `question`、`answer`；文本字段均为非空字符串，`qas` 为非空列表。
- QA 关联完整正文，不要求与段落或压缩窗口对齐。来源专属转换在数据准备阶段完成。

```json
{"text": "完整正文", "qas": [{"question": "问题一", "answer": "答案一"}, {"question": "问题二", "answer": "答案二"}]}
```

### 读取与监督

- [data.py](data.py) 统一读取并将每个 QA 展开为独立 episode，重复使用正文，各自初始化记忆；外层只计算当前答案的 CE，局部 Attention Matching 目标由模型内部生成。
- 提示使用统一短答 system 指令；user 消息包含 `Document:` 正文与 `Question:` 问题，assistant 消息为参考答案。按基座 chat template 分词，设置 `enable_thinking=False`。
- [TokenizedEpisode](data_types.py) 分别保存 `prompt_ids`、`target_ids`，后者包含模板结束 token；`id` 标识 QA，`group_id` 根据正文内容生成。engine 在前向时拼接 token，并由 prompt 长度确定监督起点。
- 先检查原始 train/dev 正文无重叠，再依次过滤展开后的 episode：
  1. **人为上限**：仅作用于 train；`max_seq_length` 省略、设为 `null` 或 CLI `--max-seq-length 0` 时关闭。
  2. **模型窗口**：train/dev 均强制受基座 `max_position_embeddings` 限制。
  3. **有效监督**：完整输入至少为 `pending_size + recent_size + 2` tokens，答案须从首次 memory 写入后的有效目标位置开始。
- 计长包含正文、问题、答案及模板 token；不截断、不打包多个文档，不符合条件时整条过滤。分词结果在内存中供各 epoch 复用，文档数、QA 数与过滤统计写入 `data-statistics.json`。

### 当前数据：δ-Mem QASPER

- **来源与转换**：[δ-Mem 发布集](https://huggingface.co/datasets/huaXiaKyrie/delta-mem-qasper-data) 的原始文件保存在 `data/raw/delta-mem-qasper/`。去除对话编号及固定提问包装，按原顺序拼接正文，并将同一论文的全部 QA 聚合为一行。
- **固定划分**：`data/delta-mem-qasper/` 保存 `train.jsonl`、`dev.jsonl`、`test.jsonl`，按原始论文划分，保存时不做长度过滤；来源 ID 与转换规则记录在 `preparation.json`。
- **实验预设**：train 人为上限 12288 tokens；dev 计算答案 CE，test 留作独立评估。dev/test 均为本实验留出集。
- **构造与使用**：见[数据说明](../../../data/delta-mem-qasper/README.md)与[构造入口](../data_preparation/delta_mem_qasper/README.md)。

## 配置

- [配置样例](../../../configs/v4/attention_matching/experiment.json) 是唯一配置入口，顶层只有 `model` 和 `training`。
- `training.train_file`、`dev_file`、`output_dir` 相对于配置文件所在目录解析；预设读取上述 QASPER train/dev 划分，数据来源通过 `tags` 标记。
- 模型使用 HF repo ID 或显式绝对本地路径，`revision` 可指定固定模型 revision。

| 设置 | 样例值与语义 |
|---|---|
| `max_seq_length` | train 的人为上限，预设 12288，省略或 `null` 关闭；完整输入包含正文、问题、答案和 chat template |
| `pending_size`／`recent_size` | 256／128；首次边界在第 384 个 token 处理完后 |
| `num_slots`／`memory_dim` | 每层 32 个 slots，每个 128 维 |
| `query_dim`／`value_dim`／`num_probes` | 64／128／16 |
| `query_mode` | `conditioned`：仅根据 pending 生成压缩查询；`fixed`：输入无关的可学习查询对照 |
| `query_normalization` | 默认 `none`：不归一化；`fixed_norm`：固定查询范数，与 `query_mode` 独立 |
| `read_query_norm`／`compression_query_norm` | 固定范数模式的读取／压缩查询目标范数；正数，省略或 `null` 时均取 `sqrt(query_dim)`，样例维度对应 8 |
| `inner_steps`／`inner_lr` | 1／0.1；测试时也使用同一局部更新 |
| `correction_scale` | 0.1；共享的固定 Q/O 修正尺度 |
| `backbone_dtype` | 默认 `float32`；GPU 样例为 `bfloat16`，新增模块与 slots 为 FP32 |
| `attention_implementation` | 默认 `sdpa`，可选 `eager`；局部 memory reader 始终使用支持二阶反传的显式计算 |
| `loss_chunk_size` | 256；仅为有效监督位置计算 LM head／CE，按块使用非重入重计算 |
| `global_batch_size`／`micro_batch_size_per_gpu` | 8／2；每卡并行处理变长轨迹，在全局 batch 内累积梯度；双卡完整 batch 对应每卡两次 microbatch |
| `save_total_limit` | 2；保留最近成功保存的两个 checkpoint 目录 |
| `swanlab_project`／`group`／`run_name`／`tags` | 默认仅本地记录；在线跟踪显式指定已有 project 和 group，run 名称可指定，tags 使用命名空间 |

固定范数模式对读取查询，以及压缩生成器的初始查询和最终输出施加归一化；两个范数为固定超参数，不是可学习参数。默认模式直接使用这些查询，不施加范数约束。配置解析后将实际范数数值写入 run/checkpoint；`none` 模式下这两个数值不参与计算。

配置是开发起点，尚未由正式实验确认有效性或显存预算。改变查询生成或归一化设置时应使用新的输出目录和独立初始化；同一 run 的恢复要求配置、分词后数据、设备类型和 world size 一致。

## 运行

统一入口与新训练、恢复训练、单条生成指令见[运行指南](scripts/README.md)。

依赖 `cuda124`，PyTorch 2.6.0、Transformers 4.57.6、verl 0.8.0。

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
--resume artifacts/v4/attention-matching_20261011/train/qwen3-0.6b_conditioned_20261011/checkpoints/global_step_50
```

数据调度使用 verl 的 collator 和 StatefulDataLoader。恢复读取新增权重、optimizer、加载器状态、epoch／样本游标及各 rank RNG；先固定原 run 的 `resolved_model_revision`，再加载冻结基座，并复用输出目录的 `tokenizer/`。Checkpoint 不包含正在进行的文本流状态，保存点位于完成一个全局 batch 后。局部 slots 在每条 QA episode 开始时重新初始化。

开发集在 `eval_every` 间隔、epoch 结束及主动停止点评估，仍执行相同边界写入，但不保留外层训练图。QASPER 的 dev 指标为答案 token CE，不等于生成 EM/F1。

在线跟踪只接受已有 SwanLab project，要求显式设置 `group`。凭据使用项目 `.env` 或终端中的 `SWANLAB_API_KEY`，每次恢复沿用本地 `swanlab.json` 的 run 身份。训练与验证只追加当前真实 optimizer step 的原生指标；不上传累计曲线快照，也不自动创建 project。

单条流式 greedy 生成读取纯文本 prompt，从 checkpoint 恢复新增权重，并使用 run 中记录的模型 revision 与 tokenizer。输出 JSON 包含 prompt、生成文本及 token IDs：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m latent_working_memory.v4.inference \
  --checkpoint artifacts/v4/attention-matching_20261011/train/qwen3-0.6b_conditioned_20261011/checkpoints/global_step_50 \
  --prompt-file data/attention-matching/prompt.txt \
  --output artifacts/v4/attention-matching_20261011/eval/greedy-sample.json \
  --max-new-tokens 128 --device cuda
```

这里的 prompt 路径与 checkpoint 需替换为实际文件。生成中继续执行相同的局部 TTT；prompt 与新增 tokens 的总长度不能超过基座绝对位置上限。当前没有正式 benchmark 评估管线。

| 产物 | 内容 |
|---|---|
| `run.json` | 解析后的配置、实际模型 revision、数据身份、设备类型与 world size |
| `data-statistics.json` | 实际长度过滤、分组划分、样本数和有效监督 token 数 |
| `tokenizer/` | 本次运行保存的 tokenizer，供后续推理复用 |
| `metrics.jsonl` | 真实 step 对应的 `train/*`、`dev/*`、step 耗时与峰值显存；资源量取各 rank 最大值 |
| `checkpoints/global_step_N/state.pt` | 新增权重、optimizer、进度与各 rank RNG；作为完整保存成功的标记 |
| `checkpoints/global_step_N/data_<rank>.pt` | StatefulDataLoader 状态；checkpoint tracker 和保留数量由 verl 管理 |
| `swanlab.json` | 启用在线跟踪时保存 project／group／run 身份 |
| `training-result.json` | 完成状态、累计 steps／样本访问数／目标 token 数与最后 checkpoint |

## 实现契约与验证

`data_preparation/delta_mem_qasper/` 将源数据构造成文档 QA 格式并按论文划分，`data.py` 统一读取、展开 QA 并筛选，`data_types.py` 定义 prompt／target 分离的 episode；`memory.py` 定义 source projection、共享 reader、查询生成与可微局部写入；`model.py` 管理 Q/O hooks、原始 KV、源缓存、变长 microbatch 和分块 CE；`engine.py` 使用 verl `BaseEngine` 的 replicated/DDP 执行外层优化；`training_data.py`、`checkpoint.py` 和 `tracking.py` 分别接入加载器、checkpoint 生命周期和增量指标。设备、RNG 与凭据 helper 位于项目公共模块，v3／v4 共用。

- 仅 pending 的逐层源表示进入局部目标。源缓存每层最多包含 `B+W` 个表示；基座的原始 KV 不能替代这份缓存。
- 首次写入前禁用 memory 读取修正，但可学习初始 slots 参与第一次写入。边界更新之后，新 memory 影响后续前向；保留的 recent KV 不重算，绝对 position 不重置。
- 局部损失对 value 维求和、对压缩查询平均，再乘 0.5。内循环只求 slots 的偏导；训练时保留 teacher、generator 与跨边界状态的完整 meta-gradient。
- 外层仅统计首次写入后产生且 target 位于 episode 监督后缀的 logits，按全局有效目标 token 数归一化。边界前最后一个 logit 不计入写后监督，即使其 target 位于边界之后。
- 主干可以使用 SDPA；局部 reader 保留显式 matmul／softmax 及精确二阶梯度。主干使用一份冻结权重，不复制 encoder／decoder。
- microbatch 在共同边界推进，padding 不参与监督，只有完整到达写入边界的样本更新各自 memory。生成 prefill 仅计算最后位置 logits。
- LM head／CE 使用无状态的局部重计算；尚未启用带可变 KV／hook 状态的主干重计算，也没有跨边界截断或一阶 meta-gradient 近似。推理缓存有界不代表精确 meta-training 的显存有界。

工程测试入口：

```bash
uv run --frozen python -m compileall -q src/latent_working_memory/v4
uv run --frozen ruff check src/latent_working_memory/v4
```

已有检查覆盖边界、meta-gradient、续训和 DDP；真实 GPU 长文本的质量、显存与吞吐需通过实验测量。

首个实现不包含独立历史约束、自适应扩容或 generator 自身的 TTT；generator 仅在离线外循环学习，部署参数冻结。当前目标只匹配新 pending 的读取行为，不能据此保证旧历史不会退化；该性质需要后续任务实验验证。
