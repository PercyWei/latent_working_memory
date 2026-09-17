# 20260915_GMSA 底座与 v2 动态记忆开发

创建时间：20260915 19:10:20 UTC+08:00

最后修订时间：20260917 19:23:25 UTC+08:00

## 当前入口：固定容量重构预训练

`memory_codec.py` 的 `MemoryCodec` 是当前共用读写框架：旧记忆经写入对齐后，与完整新文本联合编码，再生成 K 个 slots。首次写入和后续更新共享参数。Encoder 与 Decoder 复用一个完整冻结基座，保持原生 causal attention；写入启用 `encoder` LoRA，重构读取禁用全部 LoRA。独立 `decoder` LoRA 已注册但不激活。

`compression.py` 提供三种压缩模块：`mean` 对应 v2.1 的连续分组均值，`weighted` 对应 v2.2 的零初始化组内打分，`spectral` 对应 v2.3 的可训练特征变换、Fourier 长度变换与输出残差。`feature_layer` 选择 `last` 或 `mean`。

| 文件 | 职责 |
|---|---|
| `pretrain/prepare_data.py` | 无 tokenizer 的独立构造，仅保存 Parquet 位置、候选字符范围与 token 切分计划 |
| `pretrain/data.py` | 读取已保存数据，启动时分词和按真实长度筛选，epoch 复用 |
| `pretrain/objective.py` | 每次写入后的累计历史 AE 与紧邻续文 LM，完整跨压缩步骤反向传播 |
| `pretrain/engine.py` | verl BaseEngine＋PyTorch DDP，沿用 v1 replicated engine 的执行方式 |
| `pretrain/train.py` | 固定 epoch、阶段切换、保存与恢复 |
| `pretrain/tracking.py` | SwanLab 运行关联、原生曲线、资源与进度、最终汇总 |
| `pretrain/evaluation.py` | 各次压缩后的损失、一次压缩对照、最终自由重构 |
| `pretrain/checkpoint.py` | 可变权重、optimizer、游标与各 rank RNG，不保存数据集 |

数据先独立构造并保存：`single/` 与 `multi/` 各自保存 train/dev/test 索引，直接引用原始 Parquet 的文件路径、row group、组内行号、候选字符区间及目标 `write_token_ends`，不另存正文。文件路径相对于数据集目录。构造沿用 v1 质量过滤、去重和来源划分，构造无需 tokenizer，候选字符预算为 `ceil(4 × L × content_reserve_ratio) + 4 × continuation_reserve_tokens`；默认主内容系数 1.5、续文预算 768，实际 LM 目标仍为 512 tokens。训练启动后按文件和 row group 合并读取请求，每个 row group 只读一次、每篇文章在内存中复用，再按当前 tokenizer 分词、按目标 token 计划截取并筛选一次，各 epoch 完整复用保留样本。缓冲只扩大候选读取范围，实际写入和 LM 目标分别限定为 L 与 Q 个连续 tokens。六组共用 multi dev/test；静态 baseline 不加载 multi 训练数据。真实单次输入为 `[2K,8K]`；多次为 3～5 段、每段 `[K,3K]`、总长不超过 8K；续文不足 Q 或超出模型窗口的样本也排除，AE／AE＋LM 使用相同筛选。候选数与排除原因写入 `data-filtering.json`，训练步数按保留数量计算。

```bash
uv run --frozen python -m latent_working_memory.v2.pretrain.prepare_data \
  --config configs/data_preparation/fineweb-reconstruction-k512-doc100k.json \
  --output-dir data/fineweb-reconstruction-k512-doc100k_20260917
```


训练后端依赖 `verl==0.8.0`，复用 `codex/verl-v1@e0bfa37` 中经过收敛的 replicated 方案。它是项目对 verl 的 DDP engine 扩展，不使用旧 FSDP2 实验实现。尾批保留全部真实样本；空 rank 执行零权重占位计算参与同步，不增加样本计数。CUDA 采用 BF16 autocast、FP32 可训练权重，CPU 验证采用 FP32。

变长输入只在末尾补齐，补齐状态不进入 pooling 或损失；causal attention 保证有效 token 不会读取末尾补齐位置。模型接收全有效二维 mask，避免 packed-position 检测生成阻止 FlashAttention 的四维 mask。对齐输出转回基座 dtype；基础 pooling 在 FP32 中使用连续分段归约，保持原分组边界并避免原子累加的顺序波动。共享基座的 adapter 上下文缓存模块列表，使用 PEFT 的层级开关，并恢复 requires_grad 与各模块的 train/eval 状态；checkpoint 重算仍在对应上下文内执行。

`micro_batch_size` 是单卡一次并行的轨迹上限，六组配置设为 2；`global_batch_size=8` 控制一次参数更新的总样本数。同一 rank 的候选按预计读取长度排序，在相同容量下合批；AE／LM 任务和压缩次数可以不同。encoder／decoder 的补齐后位置预算分别为 `micro_batch_encoder_tokens=4096`、`micro_batch_decoder_tokens=8192`，按每次压缩的实际活跃样本检查，超过预算时拆批。

每条样本分别拼接 prompt 与目标前缀，再统一右侧补齐，按各自目标位置计算损失。同一压缩步骤只调用一次批量读取；已结束轨迹从后续写入与读取中移除，记忆通过可微索引保留跨压缩步骤的梯度。损失先按各自目标 tokens、各自压缩次数平均，再按全局轨迹数平均。`resources/mean_microbatch_size` 记录轨迹组初始平均大小，`resources/mean_active_microbatch_size` 记录各次压缩时实际活跃的平均批大小。

`model.padding_free=true` 使用 PyTorch 2.14 的公开 `torch.nn.attention.varlen.varlen_attn`。Encoder 和 Decoder 的有效 embeddings 展平成一条张量，传入各样本独立的 position IDs 与累计长度；Transformer 的 attention、MLP 和投影都只处理有效位置。输出按长度拆回原样本，连续记忆保留梯度。通过 Transformers `AttentionInterface` 注册薄适配函数，内部计算、反向与 GQA 均由 PyTorch 提供，无独立 `flash-attn` 依赖。对齐模块和带 KV cache 的生成继续使用原生 SDPA。

该模式需要 CUDA、BF16/FP16、完整 causal attention 和零 attention dropout，`attention_implementation` 保持 `sdpa`。`padding_free=false` 保留矩形补齐路径用于对照。位置预算在 padding-free 模式下按有效位置之和计算，在矩形模式下按补齐后位置数计算。每条样本仍独立检查模型窗口，展平总长不视为单条上下文长度。

六份可执行起点配置位于 `configs/v2/pretrain/qwen3-4b_pooling_*`，包含 AE／AE＋LM × warm-up／直接多次压缩四组主实验，以及 `ae-static`、`ae-lm-static` 两组全程单次压缩 baseline。本次六组均使用基础 pooling。默认完整 Encoder（`encoder_layers: null`）、K=512、Q=512、全局 batch=8、学习率 1e-4；两阶段各 32000 条候选训练轨迹，warm-up 组为 1＋2 epochs，直接多次压缩组为 3 epochs，静态 baseline 为单次压缩 3 epochs，实际 optimizer steps 按筛选后的样本数计算。模型名沿用已有 Qwen3 配置；真实数据路径、基座 revision 与超参数需按实际实验确定。

```bash
uv sync --frozen
CUDA_VISIBLE_DEVICES=4,5 uv run --frozen python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m latent_working_memory.v2.pretrain.train \
  --experiment configs/v2/pretrain/qwen3-4b_pooling_ae-warmup/experiment.json \
  --output-dir artifacts/v2/reconstruction/qwen3-4b_pooling_ae-warmup_20260916 \
  --swanlab-group qwen3-4b_pooling_reconstruction_20260916 \
  --swanlab-tag study:reconstruction
```

`experiment.json` 引用同目录的 `model.json`、`selection.json`；`selection.json` 只引用已构造的 `dataset_dir`，相对于主仓库根目录。来源与候选配额单独配置在 `configs/data_preparation/`。修改 `compression` 可运行其他压缩版本。SwanLab 默认 online，要求显式传入 `--swanlab-group <实验组>`，使用用户指定的 `latent-working-memory-v2` project；工程测试可选择 offline／disabled。Run 名称取输出目录名，同一训练的 dev／test 追加到同一 run，恢复沿用 run ID。模型、超参数与解析后的真实层数进入 config，固定 tags 标记 scope、method、data，study 标签由启动参数指定。

AE＋LM 使用 `training.lm_ratio` 指定每条轨迹本次选择 LM 的概率，否则选择 AE；六组中 AE-only 为 0，AE＋LM 为 0.5。任务贯穿轨迹内全部压缩步骤，每次只执行一次读取；训练损失是选中任务的 token 平均、压缩次数平均与全局样本平均，不再使用 `lm_weight`。任务分配由 seed 和全局 step 确定，在 rank 划分前完成，独立于 dropout RNG；各 epoch 复用文本和切点，但可以选择不同任务。dev/test 始终同时计算 AE 和 LM，保持完整配对评估。

训练日志分项 NLL 只统计选中该任务的样本，没有选中时为 null，SwanLab 不上传该项；`ae_samples`、`lm_samples` 和各自的 `*_tokens` 在本地日志记录实际样本数及监督量。旧的 AE＋LM 双损失短测使用不同训练协议，其 loss 与吞吐不代表本设置；新配置不能直接续训旧配置的 checkpoint。

warm-up 结束后复制已训练的读取对齐初始化写入对齐，重建 AdamW；直接多次压缩时两端从同一预训练 LSA 独立初始化。单次压缩 checkpoint 进行多次压缩评估时，临时以已训练 Ar 参数作为 Aw，评估后恢复；多次压缩 checkpoint 使用训练后的 Aw。各阶段使用固定学习率，训练预算以完整 epochs 配置。Ar、Aw 各自是预训练 block 0＋最终 norm 的独立副本，与共享基座不共享权重。只加载一次完整基座，然后直接复制对齐所需层；对齐骨架在 meta device 上构造，不分配真实词表层或 LM head。只训练对齐 blocks，最终 norm 冻结。Encoder 与 decoder 隐状态计算各自使用 checkpoint，适配器选择在重计算时重新执行。词表投影与 CE 按 `lm_head_chunk_size=256` 分块并分别 checkpoint，只计算有效目标位置，保留 EOS 和每条样本的 token 平均。单设备可直接运行该模块；CPU 验证添加 `--device cpu`。

中断续训使用相同配置、输出目录和 world size，添加 `--resume <output-dir>/checkpoints/step-XXXXXX.pt`。`--stop-after-steps N` 仅截短本次执行，不改变总预算。启动时读取同一构造身份并重新进行确定性的分词、筛选，随后恢复权重、optimizer、epoch／batch 游标及 RNG；阶段数据随机数与顺序打乱相互独立。

产物包含 `run.json`、`provenance.json`、`data-summary.json`、`epoch-plan.json`、训练日志、checkpoint、dev／test 结果和 `training-result.json`。`checkpoint_limit=2` 保留最近两份 checkpoint，并额外保留单次压缩训练结束和完整训练结束的 checkpoint。只有完成全部 epochs 才生成最终 test。AE 与 LM 分别记录各次压缩后的 NLL，包含最后一次压缩，不另列最终 NLL；同时保留 AE／LM 各自的轨迹平均 NLL、一次压缩 NLL 和配对差值。AE 最后一次压缩后的自由重构使用完整 token 序列 EM（`generation/final_round_exact_match`），另记录生成触顶比例与样本数。

本地小模型验证与当前实验设置见[实验记录](../../../notes/v2/20260916_fixed_capacity_reconstruction_experiment.md)。已完成真实 FineWeb＋Qwen3 GPU 短测，完整训练和论文指标复现尚未完成；执行与性能结果见[性能优化记录](../../../notes/v2/20260917_reconstruction_performance_optimization.md)。容量增长与 QA 适配仍属后续工作。

训练与 dev 使用按 optimizer step 增量记录的原生曲线；dev 的 AE／LM、一次压缩对照和配对差值合并到三个面板，分母与细分统计进入表格。资源与累计进度单独分区。最终评估只上传最后 checkpoint 的汇总图和生成样例，不逐步上传累计曲线图片。

## 初始 GMSA 迁移与更新原型

以下 `gmsa.py`、`static/` 和 `working_memory.py` 保留初始迁移及原型接口，用于上游对照；当前重构入口使用上面的统一 codec。来源提交、许可证和迁移区别见 [GMSA 迁移边界](UPSTREAM.md)。

## 模型边界

静态路径：`encoder LoRA → group mean → LSA → decoder`。
`GMSA.encode(context_ids, context_mask, ratio)` 返回 pre-LSA 记忆列表，
`align(memories)` 对齐到 decoder 输入空间，`read(...)` 计算目标 loss，`generate(...)` 自由生成。
问题和答案不送入 encoder。记忆宽度沿用基座 hidden size，不增加 v1 的 512 维瓶颈。

初始更新原型位于 `working_memory.py`，包含 `MemoryState`、`MemoryUpdater` 和 `WorkingMemory`；
`gmsa.py` 保存静态压缩底座，`gmsa_config.py` 的 `GMSAConfig` 只描述该底座，
`gmsa_checkpoint.py` 负责 GMSA 静态阶段的保存与加载，`static/` 保存静态训练与评估流程。
后续动态训练与评估流程再放入 `dynamic/`，模型组件不按训练阶段归类。

`WorkingMemory` 固定 GMSA 参数，使用 `MemoryUpdater` 更新 pooled 状态：

$$
Z_t=\operatorname{Pool}_{r_t}(E(x_t)),\qquad
M_t=U_\theta(M_{t-1},Z_t;g_t),\qquad
p(y_t)=R(A(M_t),q_t).
$$

首次写入直接使用 $Z_1$。后续 updater 以旧状态和新 pooled 特征为输入，采用 Transformer decoder blocks
和门控残差更新，初始残差为零；新增行从当前 pooled 特征初始化。旧状态不与新 token 直接求平均。
首次容量由输入长度和压缩率决定；后续 `grow_by` 为非负整数，且不能超过当前 pooled 行数或状态上限。
这是原型的显式契约，不是自适应策略：增长动作由调用方提供，不会恢复此前已经丢失的事实。

`MemoryState` 保存 `values`、`seen_tokens`、`updates`。默认保留完整跨写入计算图；
调用 `detached()` 才截断状态梯度。LSA／reader 参数虽冻结，读取时仍保留输入梯度。
动态模块与静态模型组合后的权重可通过 PyTorch `state_dict` 使用；当前尚未定义动态实验 checkpoint／恢复协议。

## 静态训练与数据

根目录环境执行 `uv run --frozen` 或 `.venv/bin/python`。
配置位于 `configs/v2/pretrain/gmsa-qwen3-4b/` 和 `configs/v2/finetune/gmsa-qwen3-4b/`。
共享数据路径直接由参数指定，不预填不存在的训练集。使用规范 JSONL：

```json
{"input": "完整上下文", "prompt": "问题或指令", "answer": ["参考答案"]}
```

AE 只读取 input，使用上游重建指令；QA 使用首个参考答案训练，评估对全部参考答案取最佳分数。
不添加聊天模板；prompt 需在数据内明确。超过长度预算时失败，不隐式截断。
模型配置同时用于两阶段；QA 从 AE checkpoint 初始化权重，但新建 optimizer。

下面的命令模板从项目根目录执行，真实数据、基座和显存预算仍需在正式训练前确定。
示例双卡下 batch 为 `2 × 1 × 16 = 32`，单卡运行时全局 batch 不同。

```bash
CUDA_VISIBLE_DEVICES=4,5 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m latent_working_memory.v2.static.train \
  --model-config configs/v2/pretrain/gmsa-qwen3-4b/model.json \
  --training-config configs/v2/pretrain/gmsa-qwen3-4b/experiment.json \
  --train-file /path/to/shared/train.jsonl --eval-file /path/to/shared/dev.jsonl \
  --output-dir artifacts/v2/gmsa-static/pretrain

CUDA_VISIBLE_DEVICES=4,5 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m latent_working_memory.v2.static.train \
  --model-config configs/v2/pretrain/gmsa-qwen3-4b/model.json \
  --training-config configs/v2/finetune/gmsa-qwen3-4b/experiment.json \
  --train-file /path/to/shared/train.jsonl --eval-file /path/to/shared/dev.jsonl \
  --initialize-from artifacts/v2/gmsa-static/pretrain/final \
  --output-dir artifacts/v2/gmsa-static/finetune
```

同一训练恢复时保留原参数和输出目录，用 `--resume <output-dir>/checkpoint-N` 替换 `--initialize-from`。
模型／训练设置、数据路径及 world size 必须与 `run.json` 一致；恢复精度还要求输入文件内容不变。
`provenance.json` 保存源码提交与 dirty 标志、上游提交、实际依赖版本、设备和解析到的基座 revision；恢复时保留首次运行记录。
不保存输入的下载校验清单。`final/` 是最终模型快照，`checkpoint-N/` 才包含训练恢复状态。
训练 loss 与 dev loss 为含 EOS 的目标 token CE；dev 每个压缩率分别报告 batch loss 的样本加权平均，
不是全语料 token-weighted NLL。完整状态保存包含冻结模型，checkpoint 可能较大。

## 静态生成评估

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m latent_working_memory.v2.static.evaluate \
  --checkpoint artifacts/v2/gmsa-static/finetune/final \
  --data /path/to/shared/test.jsonl --stage finetune \
  --max-context-tokens 4096 --max-target-tokens 1024 --max-new-tokens 128 \
  --output-dir artifacts/v2/gmsa-static/evaluation
```

逐样本、逐压缩率保存预测、EM／token F1、生成触顶标志、参考答案和条件。
比较正确 memory、空 memory、完整原文，以及存在不同来源且压缩后行数相同的错误 memory。
错误记忆无合格 donor 时省略，因此 summary 分母按条件分别记录；不能将不同覆盖面直接相减。
完整原文对照使用相同训练后 decoder，当前没有冻结初始基座对照。
指标为英文 SQuAD 式归一化后的 EM／token F1，不作为中文 QA 的完整指标方案。
AE 模式的指标同样是重建文本匹配，不等于论文全部重建指标。
评估目录保存 `evaluation.json`、`predictions.jsonl`、`summary.json`，不覆盖已有输出。

## 初始迁移验证记录

- 随机小型 Llama／Qwen3：pooling 尾组、padding、mask、target 因果位置、AE／QA 冻结关系、生成 batch 一致性。
- 上游单样本数值比较：3 种长度 × 2 个压缩率的 logits／loss 一致。
- AE→QA→生成评估端到端入口，以及同 run 中断恢复；恢复后权重与不中断训练逐参数一致。
- 动态首次写入、固定容量内容修改、扩展、容量边界、跨更新梯度及显式 detach。
- 本地双进程 Gloo 静态训练、保存与全部压缩率 dev 评估；记录位于 `artifacts/v2/validation/gmsa-migration_20260915/`。

初始迁移时 v2 共 14 项测试通过（含显式指定上游源码的数值比较）。全仓测试 288 项通过、1 项失败：
`tests/test_cdic_reproduction.py::test_required_cdic_sources_are_present` 仍要求已不存在的
`src/cdic_repro/checkpoint.py` 和 `src/cdic_repro/icae_adapter.py` 旧路径；原分支 HEAD 已不存在这两项，本轮没有修改 C-DIC。
测试输出保存为上述验证目录中的 `v2-tests.log` 和 `pytest.log`。

这些检查验证实现契约，不证明随机小模型具备压缩效果，也不代表 CUDA／真实 Qwen3-4B 显存验证。
当前实验采用上面的统一 codec 与重构入口；后续进行真实基座上的训练验证。

## 实际运行准备

`pretrain.profile` 使用真实基座检查最长单次压缩、3 次／5 次压缩和 AE＋LM 的峰值显存、吞吐与梯度。`pretrain.experiment` 按显式实验列表调度 GPU 4–7，每卡一个独立进程，记录命令、PID、退出码和完成状态；任一实验失败后暂停新的排队任务。数据与模型配置保持各实验自身的契约。
