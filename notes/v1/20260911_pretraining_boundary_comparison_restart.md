# 20260912_双卡预训练边界对比实验（12:39:04 UTC+08:00）

创建时间：20260911 16:27:19 UTC+08:00
最后修订时间：20260912 12:39:04 UTC+08:00

## 实验设置

semantic、random、mixed 三组从相同模型种子重新初始化，并行进行同步数据并行训练：semantic 使用物理 GPU 0、1，random 使用 4、5，mixed 使用 6、7。每卡 microbatch 2，梯度累积 2，全局有效 batch 8；关闭梯度检查点，BF16。每组 20,000 步，模型种子 42，数据种子 20260907。学习率峰值 3e-5，600 步 warmup，余弦衰减至 3e-6；长度课程为前 6,000 步，压缩率 2/4/8 等概率。AE/LM 权重均为 1。

沿用 `data/v1/boundary-comparison-2048-20260910`，每组训练集 157,320 条，AE/LM 数量相等，mixed 两来源各占一半。输入和 LM 目标各不超过 2048 tokens。三组共享 semantic/random 两套 dev/test。每 1,000 步保存 checkpoint 并评估两套 dev，各取固定 120 条；每 2,000 步及初始基线进行小规模 AE 生成评估。每组结束后，在该组分配的两卡分别评估两套 test，各组独立推进。

数据并行按全局 AE/LM 样本数归一化损失，按长度交错分配样本，每次参数更新前合并梯度。主进程写入指标和 checkpoint，保存全局采样状态以及每个 rank 的随机状态，恢复时校验 world_size、配置及数据身份。周期 dev 由主进程执行，另一进程等待。

## SwanLab 与产物

项目为 `latent-working-memory-v1`，group 为 `lwm-boundary-comparison-2048-20260911`。训练 run 名称为 `pretrain-semantic-157k-20260911`、`pretrain-random-157k-20260911`、`pretrain-mixed-157k-20260911`。测试 run 名称采用 `evaluate-训练来源-test-测试来源-157k-20260911`，其中 157k 仍表示对应模型的训练集规模，评估样本数记录在 config 中。job_type 为 train/evaluate，tags 保留 scope:main、method:latent-working-memory、study:boundary-comparison、data:fineweb 及实际来源。

配置：`configs/v1/pretrain_boundary_comparison_dual_a800.json`。训练产物：`artifacts/v1/experiments/boundary-comparison-2048-20260911/`。测试产物：`artifacts/v1/evaluations/boundary-comparison-2048-20260911/`。调度状态：`artifacts/v1/experiment-plans/boundary-comparison-2048-20260911/`。

本次重跑沿用现有数据集。原 20260910 系列本地与服务器训练产物和调度日志按用户要求清理；单卡、双卡性能测试报告、结果和独立测试入口按用户要求清理；正式训练的梯度同步模块保留。云端旧记录的删除状态单独核实，不将新旧 run 混用。

## 启动记录

20260911 16:29:25 UTC+08:00 启动双卡系列，训练代码提交为 `5b32bb4`。首组 `pretrain-semantic-157k-20260911` 已建立 [SwanLab 运行](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/vb5bos79)，首先执行初始双 dev 基线评估，之后进行参数更新。后续改为三组并行调度。

20260911 16:54:34 UTC+08:00，按用户授权将 random 分配到 GPU 4、5，mixed 分配到 GPU 6、7；semantic 继续使用 GPU 0、1。各组显式设置 `CUDA_VISIBLE_DEVICES` 和 `LWM_ALLOWED_PHYSICAL_GPUS` 为该组设备编号，训练配置保持一致。

## 最终结果

三组均完成 20,000 步训练、最终双 dev 评估和两套 test。全部任务于 20260912 00:39:22 UTC+08:00 结束。最终 checkpoint 为各训练目录下的 `checkpoints/pretrain-step-020000.pt`，六次 test 均正常退出；全部训练记录中的 loss 和梯度范数均为有限值。

下表为最终 checkpoint 的 teacher-forcing NLL（按目标 token 加权，不含 EOS，越低越好）。每套 test 请求固定 120 个样本，按合法压缩容量展开评估；AE 自回归生成选取 12 个 AE 样本，每个样本评估三种容量，共 36 次生成。各训练来源共享同一测试面板。

| 训练来源 | semantic test AE | semantic test LM | random test AE | random test LM |
|---|---:|---:|---:|---:|
| semantic | 2.1312 | 1.9941 | 2.0756 | 2.2444 |
| random | 2.1584 | 2.0163 | 2.0627 | 2.2408 |
| mixed | 2.1233 | 1.9852 | 2.0440 | 2.2173 |

mixed 在四项 NLL 上均最低，但只有单一 seed 和小规模固定测试面板，差异尚无统计显著性结论。三组 LM 正确 memory 均优于各自的空 memory 和错误 memory 对照，表明存在输入相关的信息利用。

mixed 的 semantic test LM NLL 为 1.9852，保留等量近期原文的对照为 1.9842，二者接近；random test 上分别为 2.2173 和 2.1777，压缩记忆仍落后。random 训练组在 semantic test 上略优于自己的近期原文对照（2.0163 对 2.0240），其余组合尚未超过该对照。对照使用各组训练后的 reader LoRA，不能将不同组的对照变化直接归因于记忆存储能力。

六组 AE 生成评估的完整重建率均为 0%，归一化 token 编辑距离为 0.948–0.971，BLEU-4 为 0.328–0.958（0–100 标度）。mixed 的两套 test 正确前缀比例分别约 0.596% 和 0.679%。当前模型在 teacher forcing 下的预测改善尚未转化为忠实的自回归重建能力；后续分析应优先核对生成读出路径与训练路径的一致性，再分析记忆依赖和生成误差累积。

| 训练来源 | 累计压缩输入 tokens | 累计目标 tokens |
|---|---:|---:|
| semantic | 64,634,170 | 56,669,206 |
| random | 65,695,227 | 55,479,358 |
| mixed | 65,114,675 | 56,028,257 |

上述计数仅统计训练，不含评估；三组均采样 160,000 次，长度课程和循环采样使其不等同于完整遍历训练集一次。

测试报告位于 `artifacts/v1/evaluations/boundary-comparison-2048-20260911/evaluate-{训练来源}-test-{测试来源}-157k-20260911/test-step-020000.json`，逐条结果位于同名 `.jsonl`。

SwanLab 测试记录：semantic 训练组的 [semantic test](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/q830jqgt)、[random test](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/aaho5zki)；random 训练组的 [semantic test](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/zkkxq86z)、[random test](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/zo41umze)；mixed 训练组的 [semantic test](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/9vqipeih)、[random test](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/ew086khr)。

## 当前评估与展示

AE 评估正确记忆与错误记忆，记录 NLL、PPL、BLEU-4 和正确前缀比例。LM 评估正确记忆、错误记忆、空记忆、完整原文及关闭 reader LoRA 的完整原文，记录 NLL/PPL。NLL 按目标内容 token 加权，BLEU 按语料计算，正确前缀比例按生成记录宏平均；表格同时记录评估次数、目标 token 数及生成次数。重建样例保留原文和预测文本。

统计分为总体与长度×压缩率联合分组。联合分组从逐条记录重新聚合，长度与有效压缩率均按 2 的幂次上界分桶。原始记录保留实际 memory 容量。每个模型评估 run 有 12 张图：总体 AE 四项、LM 两项，以及正确记忆的联合分组 AE 四项、LM 两项。总体图按对照条件分组，两套测试来源相邻；联合图横轴依次为长度、ratio，每个位置的 semantic/random 相邻。对照条件使用不同色系，测试来源使用同色系的深浅色。浮点展示最多四位小数，柱顶数值隐藏，精确值保留在报告中，缺失分组留空。

图、表、样例分别位于 `charts/*`、`tables/*`、`examples/*`。单模型评估同时比较两套测试来源；跨模型汇总比较三种训练来源。

`python -m latent_working_memory.v1.publish_reports` 从已有 JSONL 重新聚合结果并发布，不运行模型推理。`--reports` 指向 JSON 列表，每项包含 `training_source`、`evaluation_source`、`report`；report 指向原始 JSON，逐条记录使用同名 JSONL。重复传入 `--evaluation-output 训练来源 输出目录` 为每个模型发布评估，`--output-dir` 发布跨模型汇总。项目、group、tags 和 online 模式通过启动参数指定。

本轮精简结果发布到 `evaluate-{semantic,random,mixed}-joint-157k-20260912`，跨模型 run 为 `compare-boundary-joint-157k-20260912`，均保存在现有 `latent-working-memory-v1` 项目及原 group 中。原始六份测试报告保留；正文最终结果章节记录首次完成时的评估结果。
