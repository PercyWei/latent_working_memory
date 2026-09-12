# 20260912_短文本预训练目标对比实验

创建时间：20260912 23:38:14 UTC+08:00
最后修订时间：20260913 00:02:23 UTC+08:00

本实验比较 AE-only、从开始联合 AE/LM、AE warm-up 后联合训练。目标是在短文本、低压缩率条件下判断读写结构能否建立忠实重建，以及 LM 目标对这一能力的影响。实验沿用[预训练数据类型对比](20260911_pretraining_data_comparison.md)的产物与报告布局。本轮用户明确指定仅使用物理 GPU 4、5。

## 1. 数据来源与对比设计

来源池为 `data/v1/fineweb-4096-doc100k_20260910/`。直接筛选旧样本无法满足短 LM 独立评估面板，因此从同一来源池的 eligible 原文重新构造，继承原文档 train/dev/test 划分及近重复簇身份。semantic 保留完整句界；random 从原文 token 边界随机截断。X 与 LM 的 Y 均为 96–128 tokens，Y 紧邻 X。AE 重建完整 X；不进行 QA 或下游适配。

| 配置／数据 | 位置 |
|---|---|
| 数据构造配置 | `configs/data_preparation/fineweb-4096-doc100k_pretrain-objective-comparison-128.json` |
| 训练配置 | `configs/v1/pretrain-objective-comparison-128/{ae-only,joint,ae-warmup}.json` |
| 派生数据 | `data/v1/fineweb-4096-doc100k_20260910/derived/pretrain-objective-comparison-128_20260912/` |

实际训练集共 32,000 条，来自 3,200 篇原文档，AE/LM × semantic/random 各 8,000 条；每个单元按 X 的 96–103、104–111、112–119、120–128 四档各取 2,000 条。三组共享同一个 mixed 训练目录，采样器决定启用任务。每套来源的 dev/test 各 240 条（AE/LM 各 120），同一 split 的两来源、两任务之间也使用独立文档，共 480 篇。目标长度受相同区间约束，精确长度分布随产物记录，不宣称逐 token 完全匹配。

## 2. 训练与评估设置

| 组别 | 前 5,000 步 | 后 15,000 步 | AE／LM 累计样本访问 |
|---|---|---|---|
| ae-only | AE | AE | 160,000／0 |
| joint | AE+LM | AE+LM | 80,000／80,000 |
| ae-warmup | AE | AE+LM | 100,000／60,000 |

每组 20,000 次 optimizer 更新。AE-only 每步 8 条 AE（两来源各 4）；联合每步 4 条 AE 与 4 条 LM（每个任务两来源各 2）。样本在各任务／来源池内独立打乱轮转。AE-only 的损失权重为 1／0，联合为 0.5／0.5；样本先按目标 tokens 平均，任务内部再等权平均。三组固定总访问预算，AE／LM 暴露量不同，因此不能把 warm-up 的收益单独解释为顺序效应。

| 配置项 | 设置 |
|---|---|
| 基座 | Llama-2-7B-Chat，冻结 |
| Writer | 512 维，3 层，8 头，FFN 2048 |
| 读取 LoRA | rank 16，alpha 32，dropout 0，q_proj/v_proj |
| 容量 | 固定名义压缩率 2，K=ceil(len(X)/2)，48–64 个位置 |
| 课程 | 关闭长度／压缩率课程；仅 ae-warmup 在完成 5,000 步后切换任务 |
| Batch | 双卡 × 每卡 microbatch 2 × 梯度累积 2，全局 8 |
| 精度 | BF16，关闭梯度检查点 |
| 优化 | AdamW，weight decay 0.01，梯度裁剪 1 |
| 学习率 | 峰值 3e-5，600 步 warmup，20,000 步余弦衰减至 3e-6 |
| Seeds | 模型 42，训练采样 20260907，数据构造 20260912 |
| GPU | 仅物理 4、5；双卡训练显式设置 CUDA_VISIBLE_DEVICES=4,5 |

A、C 共用完全相同的前 5,000 步 AE 轨迹。先并发运行 A、B；A 完成后，从 A 的 step 5,000 checkpoint 派生 C；继承所有可训练参数、optimizer、各任务／来源采样流及每个 rank 的 RNG 状态，C 从 step 5,001 继续原学习率进度。实际执行 55,000 次训练更新，三条逻辑轨迹均为 20,000 步。派生只允许目标权重及 warm-up 配置变化，并保存来源 checkpoint。C 的前段指标引用 A，不伪装为独立训练。

每 1,000 步保存 checkpoint 并评估 dev NLL；初始化、每 2,000 步及 step 5,000 做 AE 自由生成。每来源固定 60 条 AE 自由生成样本，共 120 条。两 GPU 按来源分担 dev 评估。

AE 对照为 memory、wrong_memory、full_context、base_full_context；LM 另加 no_memory。报告 token 加权 NLL/PPL、AE BLEU-4、正确前缀比例和整段 token 完全匹配率。最终 test 使用 step 20,000，不用 test 选择训练方案。AE-only 的 LM 结果仅作为无 LM 训练的迁移诊断。真实前缀后缀生成属于独立诊断，不改变正式自由生成口径。

## 3. 复现命令与产物

命令在服务器项目根目录 `/data/bywei/projects/latent_working_memory`，使用根目录 `.venv/` 执行。

```bash
.venv/bin/python -m latent_working_memory.data_preparation.objective_comparison \
  --spec configs/data_preparation/fineweb-4096-doc100k_pretrain-objective-comparison-128.json \
  --config configs/v1/pretrain-objective-comparison-128/joint.json \
  --output-dir data/v1/fineweb-4096-doc100k_20260910/derived/pretrain-objective-comparison-128_20260912
```

实验系列根目录为 `artifacts/v1/pretrain-objective-comparison-128_20260912/`。训练目录为 `train/pretrain-ae-only-r2_mixed-16k_20260912/` 和 `train/pretrain-{joint,ae-warmup}-r2_mixed-32k_20260912/`，规模表示该组可参与训练的样本数；AE-only 仅访问共享目录中的 16,000 条 AE，另外两组使用全部 32,000 条；评估和跨组比较分别放在 `eval/`、`compare/`，调度命令、状态、日志、报告清单及结果汇总放在 `plan/`。

SwanLab project 为 `latent-working-memory-v1`，group 为 `pretrain-objective-comparison-128_20260912`，显式标签 `study:pretrain-objective-comparison`。job_type 使用 train/evaluate/compare；超参数保存在 config。

## 4. 执行记录与结果

20260912 23:38:14 UTC+08:00：完成配置及训练支持；目标配额、warm-up 边界、任务损失、配置、训练 CLI、评估等 15 项相关测试通过。两端已提交代码一致，GPU 4、5 空闲。

20260912 23:50:50 UTC+08:00：数据构造完成。训练集 32,000 条、3,200 篇文档；semantic/random 的 dev/test 各 240 篇独立文档。逐样本验证 X/Y 长度、AE token 目标一致性、文档／来源／近重复簇跨 split 隔离。统计保存在 `plan/data-statistics.json`，不是下载完整性检查。四个训练单元的 X 平均长度在 111.62–112.00 tokens，LM Y 平均长度为 semantic 112.13、random 110.62。训练集约 3.58M 输入 tokens；各组累计访问 160,000 次，不等于 160,000 条独立文本。

为减少重复计算，所有组在 CPU 内存中缓存冻结基座的文本隐藏状态；不缓存可训练投影、Writer 或读取输出。每次运行独立建立缓存，不新增持久化缓存格式。每个 microbatch 的不同任务共享一次读取前向；双卡分别评估一个来源。冻结特征缓存、梯度路径与派生继承 optimizer 等测试通过。

调度入口如下；最多同时运行两个双卡训练任务；A 完成后允许启动 C。三组训练均完成后，各组的两来源 test 并行执行，然后自动发布评估与跨组比较，生成 `plan/results.md` 并追加到本文。最终评估额外提供 1／8／32 个真实前缀 tokens，只计分剩余后缀；诊断逐条记录单独保存为 `test-step-020000-prefix.jsonl`，不混入正式自由重建指标。

```bash
LWM_ALLOWED_PHYSICAL_GPUS=4,5 CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m latent_working_memory.data_preparation.pretrain_objective_series \
  --spec configs/experiments/pretrain-objective-comparison-128.json \
  --output-dir artifacts/v1/pretrain-objective-comparison-128_20260912
```

20260912 23:58:40 UTC+08:00：服务器相关测试 25 项通过；真实双卡 joint smoke 完成 16 步，每步全局 8 条、AE/LM 各 4 条，step 中位耗时 0.448 秒，每卡每任务峰值约 14.97 GiB。额外两个双卡任务并发各完成 64 步，中位耗时分别为 0.623、0.617 秒，短时总吞吐约提高 44%。据此采用最大并发数 2，仍仅使用物理 GPU 4、5，单任务 batch、梯度累积与优化设置不变。这是调度测速，不作为方法效率结论；正式任务的耗时包含共享 GPU 的竞争。测速产物独立标记为 smoke，不参与科学比较。

GitHub 首次同步成功；后续服务器连接 GitHub 出现 TLS 错误，按约定通过 Gitee 完成仓库同步。首个 smoke 因默认 GPU 白名单仅允许 0、1 而在模型加载前退出，修正为显式 `LWM_ALLOWED_PHYSICAL_GPUS=4,5` 后通过；未使用其他卡。

20260913 00:01:20 UTC+08:00：正式调度启动，初始代码提交 `ae6248f`，调度 PID 为 `960164`。A（AE-only，16k 可用 AE 样本）与 B（joint，32k 样本）并发双卡运行；C 等待 A 完成后从 A 的 step 5,000 派生。各命令同时显式设置 `CUDA_VISIBLE_DEVICES=4,5` 和 `LWM_ALLOWED_PHYSICAL_GPUS=4,5`。系列目录及 run 名称的 `20260912` 保留创建日期，实际执行跨入 20260913。

运行状态和失败原因保存在 `plan/status.json`，完整命令在 `plan/commands.json`，训练 stdout 分别为 `plan/{ae-only,joint,ae-warmup}-train.log`。调度器独立于 SSH 会话运行；训练后自动执行独立 test、前缀诊断和 SwanLab 比较发布，并追加最终结果。当前正式实验尚未完成。
