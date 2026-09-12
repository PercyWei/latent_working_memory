# 20260911_预训练数据类型对比实验

创建时间：20260911 16:27:19 UTC+08:00
最后修订时间：20260912 22:57:45 UTC+08:00

本文记录预训练数据类型对比的具体配置、运行与结果。阶段目标、记忆写入与读取、损失和评估方法见 [预训练与 AE/LM 评估](20260910_pretraining_and_evaluation.md)。

## 1. 数据与实验设置

### 1.1 对比组

三组从相同模型种子重新初始化，使用等规模数据及相同训练配置，并行执行同步数据并行训练。三组共享 semantic/random 两套 dev/test。

| 训练组 | semantic 占比 | random 占比 | 训练样本数 | 物理 GPU |
|---|---:|---:|---:|---|
| semantic | 100% | 0% | 157,320 | 0、1 |
| random | 0% | 100% | 157,320 | 4、5 |
| mixed | 50% | 50% | 157,320 | 6、7 |

### 1.2 数据配额与长度课程

数据位于 `data/v1/fineweb-2048_20260910/`。AE 输入 X、LM 输入 X 与目标 Y 均为 32–2048 tokens；按 X 长度分为六档，各档 AE/LM 数量相等。

| 数据划分 | 每组／每套总数 | 每长度档、每任务样本数 | 来源安排 |
|---|---:|---:|---|
| train | 157,320 | 13,110 | 按训练组配比；mixed 每档每任务的两来源各 6,555 条 |
| dev | 3,468 | 289 | semantic、random 各一套，三组共享 |
| test | 3,108 | 259 | semantic、random 各一套，三组共享 |

长度采样概率在前 6,000 步线性过渡，之后保持目标分布：

| X 长度（tokens） | 起始概率 | 目标概率 |
|---|---:|---:|
| 32–64 | 25% | 1/6 |
| 65–128 | 25% | 1/6 |
| 129–256 | 20% | 1/6 |
| 257–512 | 15% | 1/6 |
| 513–1024 | 10% | 1/6 |
| 1025–2048 | 5% | 1/6 |

每组采样 160,000 次，约为训练集数量的 1.02 倍；各长度池的实际遍历次数受课程采样影响。

数据选择配置为 `configs/data_preparation/boundary_comparison.json`，入口为 `python -m latent_working_memory.data_preparation.experiment`，结果记录在数据目录的 `selection.json`。筛选保持来源划分、完整样本和上下文预算；跨来源按任务及 X/Y 去重，train 去除 19 条、test 去除 1 条重复内容。

### 1.3 数据目录与复现

目录名 `fineweb-2048_20260910` 表示 FineWeb、X/Y 上限 2048 tokens 及创建日期。semantic/random 单类数据集共 163,896 条（train 157,320 + dev 3,468 + test 3,108）；mixed 仅保存 157,320 条训练样本，评估共用两套来源数据。精确数量写入各数据集的 `preparation.json`，选择结果汇总在 `selection.json`。

```text
fineweb-2048_20260910/
├── semantic/
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── preparation.json
├── random/
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── preparation.json
├── mixed/
│   ├── train.jsonl
│   └── preparation.json
└── selection.json
```

semantic/random 的训练与评估划分使用同一份准备记录；mixed 的训练通过 `--evaluation-dirs` 显式引用 semantic/random 两套评估数据。该映射由 `selection.json` 的 `evaluation_dirs` 提供，本实验将其保存为产物目录下的 `plan/evaluation-dirs.json`。`selection.json` 同时记录来源配置、配额、筛选统计和训练数据目录。

按原配置重新选择数据的命令为：

```bash
.venv/bin/python \
  -m latent_working_memory.data_preparation.experiment \
  --spec configs/data_preparation/boundary_comparison.json \
  --config configs/v1/pretrain_boundary_comparison_dual_a800.json \
  --output-dir data/v1/fineweb-2048_20260910
```

重新构造使用尚未存在的输出目录，并保留相同来源文件、选择 seed 和训练配置。样本筛选与顺序可重复生成；每次独立准备产生新的数据身份。

20260912 完成数据目录整理，七份样本文件保留原内容与行序；semantic/random 合并训练和评估的准备记录。60 个预训练 checkpoint 的评估数据身份同步到合并后的数据集身份，模型及 optimizer 张量、采样状态保持原值。调度命令和数据目录映射均使用上述路径。

### 1.4 模型与记忆配置

| 配置项 | 设置 |
|---|---|
| 语言模型基座 | Llama-2-7B-Chat |
| 写入／读取窗口 | 各 4096 tokens |
| 记忆维度 | 512 |
| Writer | 3 层，8 个注意力头，前馈维度 2048 |
| 读取 LoRA | rank=16，alpha=32，dropout=0；作用于 q_proj/v_proj |
| memory 容量上限 | 4096 个位置 |
| 名义压缩率 | 2、4、8，各以 1/3 概率采样 |

### 1.5 优化与执行配置

训练配置见 `configs/v1/pretrain_boundary_comparison_dual_a800.json`。

| 配置项 | 设置 |
|---|---|
| 每组训练步数 | 20,000 次 optimizer 更新 |
| 每卡 microbatch | 2 条样本 |
| 梯度累积 | 2 次 |
| 全局有效 batch | 8 条样本（2 卡 × 2 × 2） |
| 数值精度／梯度检查点 | BF16／关闭 |
| 优化器 | AdamW，weight decay=0.01 |
| 学习率 | 峰值 3e-5；600 步 warmup，余弦衰减至 3e-6 |
| 梯度裁剪阈值 | 1.0 |
| AE／LM 损失权重 | 各 1 |
| 模型 seed | 42 |
| 训练采样 seed | 20260907 |
| 实验数据选择 seed | 20260910 |

### 1.6 保存与周期评估

| 项目 | 时机 | 面板／产物 |
|---|---|---|
| dev 条件预测 | 初始化、每 1,000 步及最后一步 | 每套固定 120 条样本 |
| dev AE 自由生成 | 初始化、每 2,000 步 | 每套选取 12 条 AE 样本 |
| checkpoint | 每 1,000 步 | 保存可训练参数、optimizer 与恢复状态 |
| 独立 test | 每组训练结束后 | 在该组两卡分别评估 semantic/random test |

各训练组独立推进，完成训练后即执行自己的两套 test。

## 2. SwanLab 与产物

| 元数据 | 设置 |
|---|---|
| project | `latent-working-memory-v1` |
| group | `lwm-boundary-comparison-2048-20260911` |
| job_type | `train`、`evaluate`、`compare` |
| 公共 tags | `scope:main`、`method:latent-working-memory`、`study:boundary-comparison`、`data:fineweb`，另附实际来源标签 |
| 训练 run 名称 | `pretrain-{semantic,random,mixed}-157k-20260911` |
| 首次测试 run 名称 | `evaluate-{训练来源}-test-{测试来源}-157k-20260911` |

名称中的 157k 表示对应模型的训练集规模，评估样本数记录在 config 中。

服务器仓库根目录为 `/data/bywei/projects/latent_working_memory`，本系列产物统一位于 `artifacts/v1/pretrain-data-comparison-2048_20260911/`。名称用 `-` 连接同一语义组内的词，用 `_` 分隔语义组；这里 `pretrain-data-comparison-2048` 为实验属性组，`20260911` 为日期组：

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

首次测试的 SwanLab 记录：

| 训练来源 | semantic test | random test |
|---|---|---|
| semantic | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/q830jqgt) | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/aaho5zki) |
| random | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/zkkxq86z) | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/zo41umze) |
| mixed | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/9vqipeih) | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/ew086khr) |

## 5. 完整评估设置与发布

采用 [预训练评估方法](20260910_pretraining_and_evaluation.md#5-aelm-评估) 中的 AE 四条件与 LM 五条件，补齐 AE 完整原文与基座原文对照。上一节保留首次测试时的指标和诊断。

| 评估项 | 设置 |
|---|---|
| checkpoint | 三组均为 step 20,000 |
| 测试来源 | semantic、random |
| 条件预测面板 | 每套请求 120 个独立文档样本 |
| AE 生成面板 | 每套 12 条 AE 样本 |
| 容量设置 | 按名义压缩率 2、4、8 展开 |
| 媒体 step／checkpoint_step | 0／20,000 |

| 图表范围 | 每个 run 的图数 | 内容 |
|---|---:|---|
| 单模型总体 | 6 | AE 四项、LM 两项，比较两套测试来源及各对照条件 |
| 单模型联合分组 | 6 | memory 条件的上述六项，按长度 × 压缩率分组 |
| 跨模型比较 | 6 | 三种训练来源 × 两种测试来源的总体六项 |

三组单模型评估加一组跨模型比较，共 42 张图。semantic、random、mixed 训练来源分别使用蓝、橙、绿色系；两套测试来源使用同色系深浅色，图例显示完整来源标签。图、表和样例分别发布。

本轮完整评估的 run 名称为 `pretrain-semantic-157k-eval-20260912`、`pretrain-random-157k-eval-20260912`、`pretrain-mixed-157k-eval-20260912`；跨模型比较为 `pretrain-157k-compare-20260912`。run 名称直接取输出目录名，项目和 group 沿用本系列设置。完整 AE 对照通过重新执行模型评估获得。

本轮单模型启动命令（输出目录现已有发布记录，再次评估时使用新的 run 目录）：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m latent_working_memory.v1.evaluate \
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
