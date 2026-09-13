# 20260912_短文本预训练目标对比实验

创建时间：20260912 23:38:14 UTC+08:00
最后修订时间：20260913 14:02:02 UTC+08:00

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

实验系列根目录为 `artifacts/v1/pretrain-objective-comparison-128_20260912/`。训练目录为 `train/pretrain_ae-only_r2_mixed-16k_20260912/` 和 `train/pretrain_{joint,ae-warmup}_r2_mixed-32k_20260912/`，规模表示该组可参与训练的样本数；AE-only 仅访问共享目录中的 16,000 条 AE，另外两组使用全部 32,000 条；评估和跨组比较分别放在 `eval/`、`compare/`，调度命令、状态、日志、报告清单及结果汇总放在 `plan/`。

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

运行状态和失败原因保存在 `plan/status.json`，完整命令在 `plan/commands.json`，训练 stdout 分别为 `plan/{ae-only,joint,ae-warmup}-train.log`。调度器独立于 SSH 会话运行；训练后自动执行独立 test、前缀诊断和 SwanLab 比较发布，并追加最终结果。当时正式实验尚未完成，最终结果见下文。

20260913 00:07:17 UTC+08:00：A、B 的 step 0 dev 评估完成，均已完成超过 60 次正式 optimizer 更新。A 每步 AE=8，B 每步 AE=4／LM=4；正式并发步耗时约 0.62 秒，损失与梯度有限。通过进程 PID 与 GPU UUID 的对应关系确认四个训练进程仅占用物理 GPU 4、5。

| 训练组 | 当前运行 |
|---|---|
| AE-only | [SwanLab](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/ul50zs7s) |
| 直接联合 | [SwanLab](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/qsfx08ca) |
| AE warm-up | 等待 A 完成后自动创建；前 5,000 步引用 A |

完成时间：20260913 09:16:53 UTC+08:00

| 训练组 | 测试来源 | AE NLL | LM NLL | AE BLEU-4 | 正确前缀 | 整段匹配 |
|---|---|---:|---:|---:|---:|---:|
| ae-only | semantic | 2.5189 | 3.1072 | 1.875 | 0.316% | 0.000% |
| ae-only | random | 2.4843 | 3.1531 | 1.609 | 0.891% | 0.000% |
| joint | semantic | 2.4397 | 2.3957 | 1.363 | 0.187% | 0.000% |
| joint | random | 2.3843 | 2.5497 | 1.805 | 0.714% | 0.000% |
| ae-warmup | semantic | 2.4311 | 2.3982 | 1.780 | 0.263% | 0.000% |
| ae-warmup | random | 2.3790 | 2.5387 | 1.470 | 0.827% | 0.000% |

以上为固定 step 20,000 的独立 test，NLL 不含 EOS。结果来自单模型 seed；warm-up 与直接联合的目标暴露量不同。C 前 5,000 步继承 A，完整轨迹关联见 plan/series.json。

| 训练组 | 训练 | 评估 |
|---|---|---|
| ae-only | [训练](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/ul50zs7s) | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/y4qrfh4w) |
| joint | [训练](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/qsfx08ca) | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/28xxvtjk) |
| ae-warmup | [训练](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/6ufuyuyr) | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/xyb659i7) |

跨组比较：[SwanLab](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/6ih300w5)。

20260913 12:43:08 UTC+08:00：规整本系列 7 个 SwanLab 运行名，保留原 run ID、状态、配置和已有指标。训练、评估和比较产物目录同步改名，更新配置及当前报告清单；原执行命令、日志、provenance、checkpoint 内嵌路径及云端既有 config 保留执行时原值，旧路径按 `plan/run-renames.json` 映射到新路径。未重新训练、评估或合并云端运行。

| 职责 | 当前运行名 | Run ID |
|---|---|---|
| train | `pretrain_ae-only_r2_mixed-16k_20260912` | `p6wovabv` |
| train | `pretrain_joint_r2_mixed-32k_20260912` | `2pkz4uim` |
| train | `pretrain_ae-warmup_r2_mixed-32k_20260912` | `zgt0wyfd` |
| eval | `pretrain_ae-only_r2_mixed-16k_eval_20260912` | `y4qrfh4w` |
| eval | `pretrain_joint_r2_mixed-32k_eval_20260912` | `28xxvtjk` |
| eval | `pretrain_ae-warmup_r2_mixed-32k_eval_20260912` | `xyb659i7` |
| compare | `pretrain_r2_mixed-32k_compare_20260912` | `6ih300w5` |

## 5. 评估追加到训练运行

评估程序增加 `--training-run <训练目录>`；要求 online 模式，checkpoint 必须来自该训练目录。读取训练目录的 `swanlab.json`，使用原 run ID 和 `resume="must"` 追加到已完成的训练 run；不接管仍在运行的训练。云端训练配置和名称沿用原值，不以评估配置覆盖，也不设置新的 job_type、group 或 tags。

指标、图表、生成样例和评估元数据统一放在 `evaluation/<split>/*`，横轴使用实际 checkpoint step。本轮为 `evaluation/test/*`、step 20,000。评估报告仍保存于 `eval/`；每次成功追加在训练目录的 `evaluation-publications/<split>-step-<step>.json` 记录来源，拦截重复发布。同次发布必须包含同一 checkpoint、同一 split 的全部待比较来源；不能分两次追加同一步的 semantic/random。

已保存的报告可直接补写，无需模型推理。以下命令在服务器仓库根目录执行，仅追加三组测试结果，不重新发布跨组比较：

```bash
.venv/bin/python -m latent_working_memory.v1.publish_reports \
  --reports artifacts/v1/pretrain-objective-comparison-128_20260912/plan/reports.json \
  --training-run ae-only artifacts/v1/pretrain-objective-comparison-128_20260912/train/pretrain_ae-only_r2_mixed-16k_20260912 \
  --training-run joint artifacts/v1/pretrain-objective-comparison-128_20260912/train/pretrain_joint_r2_mixed-32k_20260912 \
  --training-run ae-warmup artifacts/v1/pretrain-objective-comparison-128_20260912/train/pretrain_ae-warmup_r2_mixed-32k_20260912 \
  --swanlab-mode online
```

调度器后续默认把最终测试写回训练 run，跨组 compare 继续独立保存；仍支持显式创建独立 eval run。本次历史 eval/compare 记录保留。SwanLab 恢复并结束 run 可能更新云端结束时间，原训练时间以执行日志及资源记录为准。

20260913 12:55:47 UTC+08:00：追加功能已在本地和服务器分别通过 16 项相关测试，通过 GitHub 同步后从服务器仓库补写 A/B/C 三组的 semantic/random 测试报告。未运行新的训练或模型推理，未新建或删除云端 run。云端核对 30 个测试标量与本地报告一致、step 均为 20,000；每组存在 25 个评估媒体指标（图表、表格、样例及元数据）。按 key、step、值和时间戳核对原训练 loss/AE NLL 共 9,000 个曲线采样点，全部一致；云端配置、名称、job_type、group、tags、创建及结束时间保持不变。查询接口的近似汇总 median 会变化，不将其用于数据完整性判断。执行命令、日志和核验结果分别保存在 `plan/evaluation-append-command.json`、`plan/evaluation-append.log`、`plan/evaluation-append-verification.json`。

## 6. 精简 dev 与 evaluation 看板

三组训练 run 的主看板均由 dev 204 张、evaluation 361 张，精简为 dev 9 张、evaluation 10 张。每类保留 AE NLL、LM NLL、AE BLEU-4、正确前缀比例、整段匹配率共 5 张核心图；每张合并 semantic/random 与各对照条件。dev 使用原始 optimizer step 的累计曲线，自由生成指标仅连接实际执行过生成的评估步，不补造未执行的点。C 仅展示自身已有的 dev 历史，前 5,000 步的继承关系仍按原记录解释。

PPL、样本计数、分层统计、配对差值和真实前缀诊断合并到 summary/details 两张表；数值量纲不同的核心指标不共用坐标轴。同一输入的各条件预测合成一张文本卡片，保留两个来源全部 120 组样例，分两页展示；evaluation 另保留一个元数据面板。

新版训练记录与评估追加入口默认生成该布局，不再逐项创建这些冗余标量图；完整统计与逐样本记录继续保存在原 JSON/JSONL。历史 565 张旧图通过看板布局接口移入 Hidden，未删除底层指标或实验；每个训练目录的 `evaluation-publications/chart-layout-before-compaction.json` 保存原布局，`chart-compaction.json` 保存迁移统计。训练、资源等其他分区以及独立 eval/compare run 未变。

旧实验的迁移入口如下，在服务器仓库根目录执行，本轮 A/B/C 均已完成：

```bash
.venv/bin/python -m latent_working_memory.v1.compact_reports \
  --training-run artifacts/v1/pretrain-objective-comparison-128_20260912/train/pretrain_ae-only_r2_mixed-16k_20260912 \
  --step 20000
```

20260913 13:13:38 UTC+08:00：本地和服务器分别通过 14 项相关测试；服务器 GitHub 拉取超时后经 Gitee 同步。使用已有报告重新组织图表，未训练或重新推理。逐张读取三组共 15 张云端 dev 核心图，验证全部曲线名称、真实步数和数值与本地生成结果一致；原训练配置及 loss/AE NLL 共 9,000 个训练曲线采样点保持不变。核验结果见 `plan/chart-compaction-verification.json`，执行日志见 `plan/{ae-only,joint,ae-warmup}-chart-compaction.log`。


## 7. 删除后的训练 run 重建

用户删除三组原训练 run 后，已从原始 JSONL 训练日志、逐检查点 dev 报告和最终 test 报告重新上传。上文训练链接已更新；第 4 节运行改名表中的旧 ID 保留作为历史记录。独立 eval 与 compare 未重新发布。

| 组别 | 已删除 ID | 新 ID |
|---|---|---|
| ae-only | `p6wovabv` | `ul50zs7s` |
| joint | `2pkz4uim` | `qsfx08ca` |
| ae-warmup | `zgt0wyfd` | `6ufuyuyr` |

重传入口为 `latent_working_memory.v1.reupload_training --training-run <训练目录>`。配置构造、训练标量、dev 图表直接复用正式训练的函数，最终 test 复用评估追加入口。A/B 分别回放 20,000 步和 21 次 dev，C 回放第 5,001–20,000 步及 15 次 dev，接续继承的累计 token 计数。每个 dev 帧仅包含该步及之前的数据，避免把已有的后续报告提前显示。三组均保留 dev 9 个、evaluation 10 个面板，没有旧版冗余隐藏图。

本地和服务器均通过 11 项相关测试，其中实际执行小模型训练后回放保存日志，逐项比较配置、标量、图表、样例和步数。云端进一步核对 420,002 个完整标量记录点、30 张核心图、dev 历史步数、配置和标签，与原始产物一致。group/job_type 按原身份传入 SDK 并保存在本地；当前云端查询接口不返回这两个字段，原 run 的历史查询也如此，因此不将空返回值作为其验证结果。SwanLab 将顶层空配置值表示为 `{}`，配置核验按其序列化结果比较。

训练目录的 `swanlab.json` 已更新为新身份，`reupload.json` 保存新旧 ID、回放范围和完成状态；旧身份及旧评估发布凭据归档于 `swanlab-archive/<旧 ID>/`。新发布凭据引用新训练 run。执行命令、上传日志和核验结果分别保存于 `plan/reupload-commands.json`、`plan/{ae-only,joint,ae-warmup}-reupload.log`、`plan/reupload-verification.json`。

此次仅重传，不重新训练或推理。新 run 的创建时间、上传时间和 SDK 环境记录对应本次上传；原训练耗时、吞吐和显存指标仍来自原日志。后续重新训练沿用同一记录逻辑与布局；具体测量值由新的训练过程产生。


## 8. dev 改为原生增量曲线

dev 的训练轨迹改用 SwanLab 原生 `LINE` 面板。每次评估仅记录该 optimizer step 的标量，如 `dev/overview/ae/correct_prefix_ratio/semantic/memory`；同指标的来源／对照序列在一个面板中合并。原生面板在 run 初始化时配置一次，后续评估和事后回放只追加数据点，不再上传累计历史图。未执行自由生成的步数不记录生成指标，也不补零。

SwanLab 原生折线图的云端接口限制每图最多 8 个 Y 指标。四个 AE 指标各合并 semantic/random × 4 个条件，共 4 张；LM NLL 的 10 条曲线按来源拆为 2 张，每张保留全部 5 个条件。dev 当前共 10 个可见面板：6 张原生曲线、2 张统计表和2 页生成样例。曲线横轴直接使用 optimizer step，无媒体快照的步数切换；统计表和生成样例继续按检查点浏览。最终 test 的 evaluation 面板保持原样。

A/B/C 已在原 run 上补写全部 dev 标量，ID 和链接未变；原 5 张 ECharts 累计图移入 Hidden。单序列自动面板默认隐藏，仅展示合并后的原生曲线。完整历史快照与 JSON/JSONL 保留。迁移入口为 `latent_working_memory.v1.migrate_dev_scalars --training-run <训练目录>`；正常训练与删除后重传均自动采用新记录方式，不需要再执行旧的累计图迁移命令。

本地和服务器各通过 13 项相关测试，覆盖稀疏生成指标、每图曲线限制、面板仅初始化一次，以及实际训练／回放逐项一致。云端核验三组共 1,794 个 dev 标量点、18 个原生面板的指标关联和横轴配置；9,000 个原训练曲线采样点仍与日志一致，其他可见分区布局未变。

执行命令和核验结果见 `plan/dev-scalars-commands.json`、`plan/dev-scalars-verification.json`；日志为 `plan/{ae-only,joint,ae-warmup}-dev-scalars.log`。各训练目录的 `evaluation-publications/layout-before-dev-scalars.json` 和 `dev-scalar-migration.json` 保存迁移前布局及迁移结果。
