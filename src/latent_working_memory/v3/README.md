# v3：token memory 容量分配对照

实现依据：`notes/v3/20260926_step1_damage_guided_capacity_experiment.md`。首版实现五个比较对象及动态方法的**局部写入版本**，用于检验“按信息损失程度扩容”能否在相近容量下改善旧信息保留。

## 模型与方法

统一使用 **Qwen/Qwen3-4B-Instruct-2507，每块 64 slots**。

| 共用部分 | 实现 |
|---|---|
| 写入 | 新文本末尾添加可训练 gist embeddings，取其最终隐藏状态作为记忆 |
| 可训练参数 | 编码 LoRA（rank 128、alpha 32，attention/MLP projections）与 gist embeddings |
| 初始化 | gist embeddings 使用零均值高斯分布，标准差 0.02 |
| 读取 | 从相同基座复制独立冻结 Decoder，不安装 LoRA；训练时仍保留读取损失对记忆的梯度 |
| 数值与位置 | 两端基座 bfloat16、LoRA/embeddings float32；CUDA 训练、验证和最终评估使用 BF16 autocast；普通因果位置编码，dropout 为 0 |
| 激活重计算 | 编码器和解码器默认启用原生非重入逐层 checkpoint；`model.gradient_checkpointing` 控制开关 |
| 训练框架 | 复用 verl `BaseEngine`、优化器和 replicated DDP；一个外层 forward 并行展开一个 microbatch 的完整轨迹 |

| 方法标识 | 写入范围与容量 | 训练流程 |
|---|---|---|
| `icae_single` | 完整原文一次压缩为一个块 | AE＋LM → 全历史 QA |
| `icae_multi` | 每段独立压缩，拼接所有块 | 多段 AE＋LM → 全历史 QA |
| `autocompressors` | 新段与累计记忆共同压缩，始终追加 | 随机分段的 next-token LM |
| `memory_change` | 覆盖末块或追加新块；按归一化表示变化决定 | 单段 AE＋LM → 动作预热 QA → 策略 QA |
| `information_loss` | 同样的局部写入；按历史可读性损失决定 | 与上一组相同，后两阶段各自训练 |

动态追加输入为 `X_t`，覆盖输入为 `[A, X_t]`，其中 `A` 是当前最后一个块；更早的块不传入写入器。两种动作共享 LoRA 与 gist embeddings。完整历史动作标记、限制 attention、双 LoRA 三种写入版本留待后续架构对照。

冻结读取端、统一 LoRA 与统一初始化均属于本实验的共同设置。三个基线遵循相应的记忆组织与训练目标，在此共同设置下进行对照。

## 两种扩容规则

记忆变化分数对每个 slot 做 RMS 归一化，计算覆盖候选与旧末块的相对 Frobenius 距离；`I >= threshold_i` 时追加。实际保存的向量不归一化。

信息损失使用数据中固定的历史门控 QA：

```text
L0   = 更新前记忆的平均答案 NLL
Lrw  = 覆盖候选记忆的平均答案 NLL
Lapp = 追加候选记忆的平均答案 NLL
d = Lrw - L0; g = Lrw - Lapp
追加 ⇔ g > threshold_g 或 (d > threshold_d 且 g > eta)
```

- 每题先平均答案 token，再平均题目；门控无梯度，不筛题。
- 未选中候选不计算辅助训练损失；任务 QA 不包含门控题。
- 信息损失法在评估时也使用门控参考答案，输出明确标为 **offline oracle**。
- `0.1` 等默认阈值仅用于启动配置，尚未校准。每组策略训练的阈值写入 checkpoint；容量对照需要分别训练多个阈值配置。

## 数据与损失

| 数据 | 规范格式 | 处理 |
|---|---|---|
| 基础预训练 | `fineweb-reconstruction-k512-doc100k_20260917` 的原文索引，配套同级 `raw/` | 根据原文位置与已保存切点，用当前基座分词并读取正文及紧邻的 512-token 续文 |
| 已有文本成品 | 基础构造器的 `TextSample`，目录内 `train/dev/test.jsonl` | 由 `pretrain_data_view=text_samples` 显式选择；按输入长度筛选，不截断文本或目标 |
| QA | FactQA 发布目录的 `preparation.json` 与 `train/dev/test.jsonl` | 校验已有来源、事实与题池隔离；保留原文、分段、更新点及 `usage` |

- QA 加载支持现行构造流程的 **6–10 段**，不会强制重切为旧笔记中的八段，也不会使用预训练长度筛选参数裁剪轨迹。
- 训练入口读取并验证数据后记录实际 token 长度与题数。AE＋LM 的 train/dev 在长度筛选后必须仍包含两种任务。
- 阶段衔接保存并核对 AE＋LM 来源文档与去重簇，QA 不能与其重叠。
- 三个 baseline 的首组基础训练选择 `reconstruction_single`：使用 `single/` 的最终切点，正文 1024–4096 tokens。两个 ICAE 从同一正文各构造一条 AE 与 LM 样本，分别训练自己的参数；ICAE-multi 按 1024 tokens 独立写入后联合读取。后续完整 QA 轨迹提供更长的历史与更多记忆块训练。
- 动态共享预训练选择 `reconstruction_first_write`：使用 `multi/` 的第一个切点，筛选 768–1024 tokens 的正文，并读取该切点后紧邻的 512 tokens 作为 LM 目标；不执行原索引的后续递归写入。两种视图均继承原来源与划分，只在内存组织样本，不保存派生副本；目录名中的 `k512` 不决定当前模型的 64 slots。

| 目标 | 实现细节 |
|---|---|
| AE / LM | 分别重建输入、预测未写入的续写；目标末尾添加 EOS。比例由输入数据决定，统计随 run 保存 |
| ICAE QA | 完整记忆上读取全部任务题，按实际题数平均 |
| 动态 QA | 每个更新点读取 `task_new_qa_ids + task_old_qa_ids`，按实际题数平均，再平均更新点 |
| 动作预热 | 每次更新以 0.5 概率追加；由 seed、epoch、trajectory_id 确定，两组动作日程相同 |
| 策略训练 | 采用各自门控；选中路径保留完整跨步梯度，一条轨迹内不更新参数、不 detach 记忆 |
| AutoCompressors LM | 随机分段，累计记忆参与预测及写入；默认每两段为一个 BPTT 子块，之后 detach 累计记忆，参数在整条样本结束后更新 |

AutoCompressors 在索引数据上每条原始索引只生成一个 continuation 样本，拼接正文与 512-token 续文进行 LM 训练，避免同文被 AE/LM 两份重复使用；该长度也保证有足够的随机分段。对已有 `TextSample`，AE 样本使用输入全文，continuation 样本使用输入和续写的 token 流。子块内保留跨分段的下一 token 预测，子块首 token 不计损失；按目标 token 数平均。冻结读取端的损失通过当前子块内的记忆写入回传，不使用 QA 微调。当前随机分段范围为 768–1024 tokens，尾段可以更短；首组训练轨迹较短，评估时需留意更长累计记忆的泛化表现。

每个 global batch 按实际样本数平均，包括不足一批的尾批。配置项 `micro_batch_size_per_gpu` 控制每卡一次并行处理的完整样本数，`gradient_accumulation_steps` 控制累积次数；全局 batch 根据两者与 GPU 数的乘积计算并记录。五方法默认每卡 microbatch 为 2、累积 2 次，双卡全局 batch 为 8；`qa_batch_size` 默认为 8。

批量读写采用独立行的右侧 padding，仅使用有效前缀输出；因果 attention 保证有效位置不会读取右侧 padding，因此不传 padding mask，保留 SDPA 的纯 causal 路径。预训练样本在既定 global batch 的每卡分片内按目标长度、输入长度分组，减少补齐计算；采样、各卡样本归属和损失权重保持不变。动态分支按轨迹独立决策。`qa_batch_size` 控制每条轨迹一次读取的题数，批量调用合并各活跃轨迹的题目，但保留原有每题、更新点和轨迹的损失权重。多个样本共享调用的计时按参与样本分摊，调用/题目数仍按每条样本的逻辑工作量记录。

两端职责固定后，Transformer 逐层重算无需切换 adapter 状态；训练时保持 train 模式并关闭 dropout，生成时临时切换 Decoder 为 eval，使用 KV cache 后恢复原模式。独立 Decoder 每卡增加约 8 GB 的 BF16 权重，以换取逐层重算节省的激活；长目标的词表投影与 CE 继续按 256 个有效位置独立分块重算。当前保留 PyTorch 2.6 环境，不接入 v2 依赖其他接口的 padding-free 后端。实际 4B 长轨迹的显存与吞吐需在 GPU 上测量。

`model.dtype` 指基座权重存储精度；CUDA 的计算使用 BF16 autocast，CPU 保留原精度。实际 `autocast_dtype` 记录于运行元数据和 SwanLab 阶段配置，损失与门控统计使用 FP32。

checkpoint 继续只保存 gist embeddings 和编码 LoRA 等可训练状态，不保存冻结 Decoder。已有共享预训练产物可以通过 `--init-checkpoint` 初始化后续阶段；更改工程设置后的继续训练以新阶段记录为准，不保证与旧计算路径逐步数值一致。

## 启动

终端和公司 GPU 网站使用 [run_gpu.sh](scripts/run_gpu.sh)。脚本从自身位置定位仓库并切换到项目根目录，自动设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`；无需先配置路径环境变量。默认运行五方法的 `smoke` 流程，模型读取 `~/models/Qwen3-4B-Instruct-2507`，SwanLab project 为 `latent-working-memory-v3`，新运行标识按上海时间自动生成。

```bash
bash ~/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh --mode smoke --gpus 4,5
bash ~/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh --mode full --gpus 4,5
```

公司 GPU 网站填写脚本的实际绝对路径即可。`--gpus 4,5` 启动两个训练进程，最终评估使用 GPU 4；卡号默认仍为 `0,1`。只运行一个方法时增加 `--method`。命令行参数覆盖方法预设，`all` 下应用于所有方法的各个阶段；例如 `--micro-batch-size-per-gpu 4 --gradient-accumulation-steps 1` 保持双卡全局 batch 为 8。参数与数据准备见 [GPU 任务说明](scripts/README.md)。

公开入口按完整方法运行：ICAE 自动执行 pretrain → QA，动态方法执行共享预训练 → warmup → policy，AutoCompressors 执行 LM；`smoke`、`pilot` 仅缩短各阶段预算，仍走完整流程。需要先单独准备两种动态方法的共同起点时，使用 `--method shared_pretrain`。

```bash
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode full --method shared_pretrain --gpus 4,5 --run-id main-01
```

已有共享预训练 checkpoint 时，动态方法通过 `--init-checkpoint` 接着执行 warmup → policy → 最终评估，不读取预训练数据：

```bash
bash src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode full --method memory_change --gpus 4,5 \
  --init-checkpoint artifacts/v3/capacity_main-01/train/shared-pretrain-k64_main-01/pretrain/checkpoints/step-NNNNNN.pt
```

外部 checkpoint 的同阶段 `run.json` 提供 `run-id`，省略时自动继承，显式指定不同值会报错。将 `--method` 改为 `information_loss` 并保持同一 checkpoint，即可在同一系列中分别启动两个动态方法；它们与共享预训练使用相同名称后缀。`--method dynamic` 顺序运行这两种方法，`--method all` 运行全部五种方法。

SwanLab 以一个完整方法为一个 run，阶段间累计 optimizer step。ICAE 的 pretrain 与 QA 共用 run；动态方法的 warmup 与 policy 共用 run，其步数从自身 warmup 开始，不包含共享预训练；AutoCompressors 使用一个 LM run。共享预训练单独记为 `shared-pretrain-k64_<run-id>`，因此一次 `all` 完整流程共六个 run。正式方法名为 `<method>-k64_<run-id>`，试跑名称与路径见 [GPU 任务说明](scripts/README.md)。

阶段由不同训练进程执行，前一进程 finish 后，下一进程 resume 同一 SwanLab ID，页面仍显示一个 run。跨阶段只继承 LoRA 与 gist embeddings，optimizer 重新初始化；checkpoint 仍分阶段保存在 `train/<run-name>/<stage>/checkpoints/`。来源 checkpoint 的准确路径、step、SwanLab ID 和 URL 记录在 config 中；共享来源以这些记录为准。中断恢复使用内部训练模块的 `--resume` 与该阶段保存的配置，沿用原目录、卡数及批处理设置。

## 评估与产物

默认评估最后一个 checkpoint，不改写训练阈值。最终评估追加到对应方法的 SwanLab run，使用该方法的累计 optimizer step。所有方法在同一批轨迹的最终记忆上回答评估题；按末段新事实、更早旧事实以及段距离分别统计。

```bash
CUDA_VISIBLE_DEVICES=4 uv run --frozen python -m latent_working_memory.v3.evaluate \
  --checkpoint artifacts/v3/capacity_main-01/train/memory-change-k64_main-01/policy/checkpoints/step-NNNNNN.pt \
  --dataset-dir /absolute/path/to/fineweb-factqa \
  --output-dir artifacts/v3/capacity_main-01/eval/memory-change-k64_main-01/policy \
  --split test --device cuda --max-new-tokens 64 --log-to-swanlab

uv run --frozen python -m latent_working_memory.v3.compare \
  /absolute/path/to/eval-one/summary.json /absolute/path/to/eval-two/summary.json \
  --output-dir artifacts/v3/capacity_main-01/compare/dynamic
```

| 产物 | 内容 |
|---|---|
| `experiment.json` / `swanlab.json` | 方法的阶段配置与预训练来源，以及唯一的 SwanLab run 身份 |
| `<stage>/config.json` / `<stage>/run.json` | 该阶段配置、模型 revision、来源范围、数据统计与跨阶段运行身份 |
| `<stage>/metrics.jsonl` | 该阶段 optimizer step 对应的训练、开发集损失、容量、整步耗时及 CUDA 峰值显存 |
| `<stage>/checkpoints/step-*.pt` | LoRA、gist embeddings、optimizer、训练游标与 RNG；不重复保存冻结基座 |
| `trajectories.jsonl` | 动作轨迹、门控分数、容量、计时、逐题答案与 NLL/EM/F1 |
| `summary.json` | 问答质量、按段距离分层、平均/最终容量与读写成本 |
| `points.json` / `points.csv` | 同一题池的质量—容量比较点，保留阈值与 oracle 标记 |

写入计时包含实际生成的所有候选；门控题实例数与任务读取分开。CUDA 计时显式同步。ICAE-single 的平均 slots 指唯一一次完整压缩状态；其他方法平均所有更新点的保存容量。生成采用 greedy decoding；提示及答案格式由配置固定。

启动器默认在 `latent-working-memory-v3` 中记录 `train`、`dev` 和最终 `evaluation`，同一系列使用同一 group；`--tracking disabled` 仅保存本地。单独调用评估模块时，`--log-to-swanlab` 将结果追加到 checkpoint 对应的方法 run。未执行的评估指标不会补零。

## 验证与边界

```bash
uv run --frozen pytest -q tests/v3
uv run --frozen ruff check src/latent_working_memory/v3 tests/v3
```

测试使用本地随机初始化 tiny Llama/Qwen3，覆盖可微冻结读取、分支梯度、历史输入范围、门控隔离、数据契约、BPTT 截断及 CPU DDP。它们验证实现行为，不代表真实 QA 质量或 4B GPU 性能；正式运行前需在 GPU 上测量完整轨迹显存。

参考实现：[ICAE](https://github.com/getao/icae)、[AutoCompressors](https://github.com/princeton-nlp/AutoCompressors)、[verl Model Engine](https://verl.readthedocs.io/en/latest/workers/model_engine.html)。
