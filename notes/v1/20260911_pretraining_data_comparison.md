# 20260912_双卡预训练数据类型对比实验（16:11:15 UTC+08:00）

创建时间：20260911 16:27:19 UTC+08:00
最后修订时间：20260912 16:11:15 UTC+08:00

本文记录预训练数据类型对比的具体配置、运行与结果。阶段目标、记忆写入与读取、损失和评估方法见 [预训练与 AE/LM 评估](20260910_pretraining_and_evaluation.md)。

## 1. 数据与实验设置

semantic、random、mixed 三组从相同模型种子重新初始化，并行进行同步数据并行训练：semantic 使用物理 GPU 0、1，random 使用 4、5，mixed 使用 6、7。每卡 microbatch 2，梯度累积 2，全局有效 batch 8；关闭梯度检查点，BF16。每组 20,000 步，模型种子 42，数据种子 20260907。学习率峰值 3e-5，600 步 warmup，余弦衰减至 3e-6；长度课程为前 6,000 步，压缩率 2/4/8 等概率。AE/LM 权重均为 1。

沿用 `data/v1/boundary-comparison-2048-20260910`，每组训练集 157,320 条，AE/LM 数量相等，mixed 两来源各占一半。输入和 LM 目标各不超过 2048 tokens。三组共享 semantic/random 两套 dev/test。每 1,000 步保存 checkpoint 并评估两套 dev，各取固定 120 条；每 2,000 步及初始基线进行小规模 AE 生成评估。每组结束后，在该组分配的两卡分别评估两套 test，各组独立推进。

基座为 Llama-2-7B-Chat，写入与读取窗口均为 4096，memory 上限为 4096。记忆维度 512，Writer 为 3 层、8 个注意力头、前馈维度 2048；读取 LoRA rank=16、alpha=32，作用于 q_proj/v_proj，dropout=0。AdamW 的 weight decay 为 0.01，梯度裁剪阈值为 1.0。

AE 输入 X、LM 输入 X 与目标 Y 均为 32–2048 tokens。按 X 长度分为 32–64、65–128、129–256、257–512、513–1024、1025–2048 六档。每组每档每任务有 13,110 条；mixed 的两种来源各 6,555 条。每套 dev 共 3,468 条，每档每任务 289 条；每套 test 共 3,108 条，每档每任务 259 条。

长度课程的六档起始概率为 25%、25%、20%、15%、10%、5%，前 6,000 步线性过渡到各 1/6。实验数据选择 seed 为 20260910。每组采样 160,000 次，约为训练集数量的 1.02 倍；各长度池的实际遍历次数受课程采样影响。

数据选择配置为 `configs/data_preparation/boundary_comparison.json`，入口为 `python -m latent_working_memory.data_preparation.experiment`，结果记录在 `data/v1/boundary-comparison-2048-20260910/selection.json`。筛选保持来源划分、完整样本和上下文预算，跨来源按任务及 X/Y 去重，train 去除 19 条、test 去除 1 条重复内容。

## 2. SwanLab 与产物

项目为 `latent-working-memory-v1`，group 为 `lwm-boundary-comparison-2048-20260911`。训练 run 名称为 `pretrain-semantic-157k-20260911`、`pretrain-random-157k-20260911`、`pretrain-mixed-157k-20260911`。首次测试 run 名称采用 `evaluate-训练来源-test-测试来源-157k-20260911`，其中 157k 仍表示对应模型的训练集规模，评估样本数记录在 config 中。job_type 为 train/evaluate，tags 保留 scope:main、method:latent-working-memory、study:boundary-comparison、data:fineweb 及实际来源。

配置：`configs/v1/pretrain_boundary_comparison_dual_a800.json`。服务器仓库根目录为 `/data/bywei/projects/latent_working_memory`，本系列产物统一位于 `artifacts/v1/pretrain-data-comparison-2048_20260911/`。名称用 `-` 连接同一语义组内的词，用 `_` 分隔语义组；这里 `pretrain-data-comparison-2048` 为实验属性组，`20260911` 为日期组：

| 子目录 | 内容 |
|---|---|
| `train/pretrain-{semantic,random,mixed}-157k-20260911/` | 三组训练配置、来源记录、逐步指标、dev 结果、checkpoint 与 SwanLab 记录 |
| `eval/pretrain-{semantic,random,mixed}-157k-eval-20260912/` | 各模型完整评估，两套测试来源分别保存 JSON/JSONL |
| `eval/evaluate-{训练来源}-test-{测试来源}-157k-20260911/` | 首次六组测试的原始 JSON/JSONL |
| `compare/pretrain-157k-compare-20260912/` | 跨模型比较清单与 SwanLab 记录 |
| `plan/` | 调度脚本、命令、任务状态、启动日志与评估清单 |

后续训练、评估和比较的输出目录分别使用本系列下的 `train/`、`eval/`、`compare/`，各次运行使用独立的 run 目录；恢复训练使用原训练目录及其 checkpoint。`plan/run_jobs.py` 与 `plan/commands.json` 记录该轮训练和首次测试的调度，完整评估清单为 `plan/complete-ae-reports.json`，首次测试清单为 `plan/reports.json`。历史日志与 SwanLab 缓存中的原始启动命令保留执行时的位置；可执行脚本和可读取的报告清单使用迁移后的路径。

本次重跑沿用现有数据集。原 20260910 系列本地与服务器训练产物和调度日志按用户要求清理；单卡、双卡性能测试报告、结果和独立测试入口按用户要求清理；正式训练的梯度同步模块保留。云端旧记录的删除状态单独核实，不将新旧 run 混用。

## 3. 训练运行记录

20260910 23:05:42 UTC+08:00 曾启动单卡配置，代码提交为 `2ed8f6d`，每卡 microbatch 1、梯度累积 8、开启梯度检查点。semantic 与 random 分别在 GPU 0、1 执行，mixed 排队；该轮随后停止，原产物已清理。以下正式结果来自重新初始化的双卡训练。

20260911 16:29:25 UTC+08:00 启动双卡系列，训练代码提交为 `5b32bb4`。首组 `pretrain-semantic-157k-20260911` 已建立 [SwanLab 运行](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/vb5bos79)，首先执行初始双 dev 基线评估，之后进行参数更新。后续改为三组并行调度。

20260911 16:54:34 UTC+08:00，按用户授权将 random 分配到 GPU 4、5，mixed 分配到 GPU 6、7；semantic 继续使用 GPU 0、1。各组显式设置 `CUDA_VISIBLE_DEVICES` 和 `LWM_ALLOWED_PHYSICAL_GPUS` 为该组设备编号，训练配置保持一致。

## 4. 训练结束与首次测试结果

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

测试报告位于 `artifacts/v1/pretrain-data-comparison-2048_20260911/eval/evaluate-{训练来源}-test-{测试来源}-157k-20260911/test-step-020000.json`，逐条结果位于同名 `.jsonl`。

SwanLab 测试记录：semantic 训练组的 [semantic test](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/q830jqgt)、[random test](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/aaho5zki)；random 训练组的 [semantic test](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/zkkxq86z)、[random test](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/zo41umze)；mixed 训练组的 [semantic test](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/9vqipeih)、[random test](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/ew086khr)。

## 5. 完整评估设置与发布

采用 [预训练评估方法](20260910_pretraining_and_evaluation.md#5-aelm-评估) 中的 AE 四条件与 LM 五条件。各模型使用 step 20,000 checkpoint，每套 test 请求 120 个独立文档样本，AE 生成面板取 12 条，容量按名义压缩率 2/4/8 展开。正文上一节保留首次测试时的指标和诊断；本轮补齐 AE 完整原文与基座原文对照。

单模型评估 run 各有 12 张图：总体 AE 四项、LM 两项，以及记忆条件下长度 × 压缩率联合分组的对应六项。跨模型比较使用 6 张图，总计 42 张图。semantic、random、mixed 训练来源分别使用蓝、橙、绿色系；两套测试来源使用同色系深浅色，图例显示完整来源标签。图、表和样例分别发布，媒体 step 固定为 0，checkpoint_step 为 20,000。

本轮完整评估的 run 名称为 `pretrain-semantic-157k-eval-20260912`、`pretrain-random-157k-eval-20260912`、`pretrain-mixed-157k-eval-20260912`；跨模型比较为 `pretrain-157k-compare-20260912`。run 名称直接取输出目录名，项目和 group 沿用本系列设置。完整 AE 对照通过重新执行模型评估获得。

本轮单模型启动命令（输出目录现已有发布记录，再次评估时使用新的 run 目录）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src artifacts/v1/pretrain-code-20260908/.venv/bin/python -m latent_working_memory.v1.evaluate \
  --checkpoint artifacts/v1/pretrain-data-comparison-2048_20260911/train/pretrain-random-157k-20260911/checkpoints/pretrain-step-020000.pt \
  --evaluation-dirs artifacts/v1/pretrain-data-comparison-2048_20260911/plan/evaluation-dirs.json \
  --output-dir artifacts/v1/pretrain-data-comparison-2048_20260911/eval/pretrain-random-157k-eval-20260912 \
  --split test --examples 120 --generation-examples 12 \
  --swanlab-mode online --swanlab-project latent-working-memory-v1 \
  --swanlab-group lwm-boundary-comparison-2048-20260911 --swanlab-tag study:boundary-comparison
```

三组评估完成后，汇总清单的 report 路径指向各新评估目录下的 `{semantic,random}/test-step-020000.json`，使用 `publish_reports --reports 清单路径 --output-dir artifacts/v1/pretrain-data-comparison-2048_20260911/compare/pretrain-157k-compare-20260912` 发布跨模型比较，并指定同一项目、group、tags 和 online 模式。单模型评估已在执行结束时发布，汇总命令只创建 compare run。

## 6. 完整评估运行记录

20260912 13:51:57 UTC+08:00 开始，14:13:26 完成，代码提交为 `5ded1be`。semantic、random 模型分别在 GPU 0、1 完成两套测试；mixed 的两套测试分别在 GPU 0、1 并行完成。三组使用原有 step 20,000 checkpoint。每组 semantic test 实际覆盖 120 个独立文档，random test 覆盖 117 个独立文档，与原评估面板一致；每套测试的 AE 四种条件各有 36 条配对生成记录，完整原文条件每篇仅推理一次并复用于各容量。

六份新报告均通过指标有限性、对照条件覆盖、目标长度配对、生成覆盖和样本身份检查。三组模型在同一测试集上的基座完整原文损失及生成结果一致。云端 42 张图已核对条件、图例、配色与单次 step 0 上传。

| 运行 | SwanLab |
|---|---|
| `pretrain-semantic-157k-eval-20260912` | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/19fgzh1r) |
| `pretrain-random-157k-eval-20260912` | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/81yl0lnd) |
| `pretrain-mixed-157k-eval-20260912` | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/d3z6hkfd) |
| `pretrain-157k-compare-20260912` | [比较](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/xltjprbj) |

原始结果位于 `artifacts/v1/pretrain-data-comparison-2048_20260911/eval/pretrain-{训练来源}-157k-eval-20260912/{测试来源}/test-step-020000.{json,jsonl}`；汇总清单位于本系列调度目录的 `complete-ae-reports.json`。

服务器已清理 12 个旧合并展示目录、原六组独立测试的 SwanLab 缓存及被替代的 mixed 串行启动日志。原六份 JSON/JSONL、正式训练记录、checkpoint 和新评估产物保留。云端旧 run 由用户手动清理。
