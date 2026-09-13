# 20260912_动态梯度传播对比实验

创建时间：20260912 23:54:16 UTC+08:00
最后修订时间：20260913 13:03:48 UTC+08:00

本系列比较完整 BPTT、源 token TBPTT 和更新次数 TBPTT。各组共享初始化参数、文本课程与评估记录。训练方法见[动态训练与 QA 评估](20260911_dynamic_training_and_evaluation.md)。本轮先运行完整 BPTT。

## 1. 数据与训练设置

| 项目 | 设置 |
|---|---|
| 数据 | SQuAD 1.1；398 篇训练文章、44 篇 dev、48 篇 test |
| 长度记录 | `data/v1/squad/llama-2-7b-chat_index.json` |
| 初始化 | mixed-157k 预训练 step 20,000 |
| K | 64、128、256、512、1024 |
| 目标压缩率 r | 2、4、8；文本长度在 [0.9rK, 1.5rK] 内 |
| 训练量 | 3 个 epoch；每轮 5 个 micro epoch；每个 100 条文本；共 1,500 篇次 |
| 参数更新 | 双卡，每卡 microbatch 1，全局 batch 2；共 750 步 |
| QA | 每次更新抽取 1 个当前问题、1 个历史问题；每题最多访问 2 次 |
| 数值精度 | BF16 |
| 激活重算 | QA 读取启用，基座逐层梯度检查点关闭 |
| 优化器 | AdamW；学习率 3e-5，weight decay 0.01，梯度裁剪 1.0 |
| GPU | 物理 GPU 6、7 |
| 环境 | 项目根目录 `.venv`，仓库源码 |

三个 epoch 的 r=2/4/8 占比分别为 60/30/10%、30/40/30%、10/30/60%。每个 K 在每个 epoch 出现一次。正式配置位于 `configs/v1/dynamic-bptt-comparison-squad/`：`full.json`、`tokens1024.json`、`updates4.json`。

初始化 checkpoint 为：

```text
artifacts/v1/pretrain-data-comparison-2048_20260911/train/pretrain-mixed-157k-20260911/checkpoints/pretrain-step-020000.pt
```

## 2. 保存与评估

每 100 步及训练结束保存 checkpoint，并进行 dev NLL 评估。初始化、每 250 步及训练结束进行 dev 自由生成。训练完成后独立执行 test 评估。

每个 split 在 5 个 K、3 个目标压缩率下各取 2 份文本，共 30 份。每份文本最多选取 2 次当前问题读取和 2 次历史问题读取；五种条件共用这些记录，生成上限为 64 tokens。结果保存 EM、F1、含 EOS 的 NLL、触顶率、保持距离与配对差异。

## 3. 命名与产物

实验系列和 SwanLab group 为 `dynamic-bptt-comparison-squad_20260912`，project 为 `latent-working-memory-v1`，研究标签为 `study:dynamic-bptt-comparison`。

| 产物 | 相对实验系列目录的位置 |
|---|---|
| 共享评估记录与数据检查 | `plan/evaluation-plan.json`、`plan/data-validation.json` |
| 完整 BPTT 训练 | `train/dynamic-full_squad_mixed-157k_20260912/` |
| 最终 test | `eval/dynamic-full-eval_squad_mixed-157k_20260912/` |
| 启动和性能记录 | `plan/` |
| 后续跨组比较 | `compare/` |

训练 run 保存配置、初始化来源、运行环境、逐步日志、micro epoch 文本记录、dev 报告及 `checkpoints/dynamic-step-000750.pt`。恢复日志名称包含恢复起始 step。

## 4. 执行命令

从服务器仓库根目录执行。先准备共享记录：

```bash
.venv/bin/python -m latent_working_memory.data_preparation.dynamic \
  --config configs/v1/dynamic-bptt-comparison-squad/full.json \
  --checkpoint artifacts/v1/pretrain-data-comparison-2048_20260911/train/pretrain-mixed-157k-20260911/checkpoints/pretrain-step-020000.pt \
  --index data/v1/squad/llama-2-7b-chat_index.json \
  --output-dir artifacts/v1/dynamic-bptt-comparison-squad_20260912/plan
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=6,7 LWM_ALLOWED_PHYSICAL_GPUS=6,7 \
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
PYTORCH_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m latent_working_memory.v1.dynamic train \
  --config configs/v1/dynamic-bptt-comparison-squad/full.json \
  --checkpoint artifacts/v1/pretrain-data-comparison-2048_20260911/train/pretrain-mixed-157k-20260911/checkpoints/pretrain-step-020000.pt \
  --index data/v1/squad/llama-2-7b-chat_index.json \
  --evaluation-plan artifacts/v1/dynamic-bptt-comparison-squad_20260912/plan/evaluation-plan.json \
  --output-dir artifacts/v1/dynamic-bptt-comparison-squad_20260912/train/dynamic-full_squad_mixed-157k_20260912 \
  --swanlab-mode online --swanlab-group dynamic-bptt-comparison-squad_20260912 \
  --swanlab-tag study:dynamic-bptt-comparison
```

独立 test 使用同一双卡启动前缀，将模式改为 `evaluate --split test`，checkpoint 指向本轮 `checkpoints/dynamic-step-000750.pt`，输出目录使用上表的 eval 路径，其余配置、共享评估记录及 SwanLab 分组相同。

## 5. 启动前验证

实现提交为 `cb9d585`。本地和服务器根目录 `.venv` 各通过 37 项测试，覆盖动态梯度、双卡一致性、恢复、共享评估、NLL／生成分离及报告聚合。服务器通过 Gitee 完成 GitHub 连接超时后的仓库同步。

完整课程实际使用 1,500 篇次、3,911,749 源 tokens、15,458 次后续记忆更新及 30,916 次 QA 监督。各 K 使用 300 篇次。dev/test 各含 30 份文本，分别选中 108／104 次问题读取；五种条件分别产生 540／520 条评估记录。

GPU 6、7 上的 K=1024 完整 BPTT 测试结果如下，每步使用 2 份文本：

| 测试文本 | 每步耗时 | 单卡峰值分配显存 |
|---|---:|---:|
| 更新次数最多 | 65.69 秒 | 27.73 GiB |
| 初始化文本最长 | 13.21 秒 | 20.10 GiB |

两种测试均完成一次完整反向传播，截断次数为 0。性能测试仅写入 `plan/profile-full-k1024.jsonl` 和本地日志。正式训练使用相同 BF16、QA 激活重算和显存分配设置。

## 6. 正式运行

完整 BPTT 于 20260913 00:00:29 UTC+08:00 启动，启动时仓库提交为 `ae6248f`，调度进程 PID 为 959569。命令与启动信息分别保存在 `plan/commands-full.sh`、`plan/launch-full.json`，训练及后续 test 的终端日志为 `plan/full-train-and-eval.log`。

训练 run：[dynamic-full_squad_mixed-157k_20260912](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/3sibkt6l)。进程依次执行初始化 dev、训练与周期 dev、最终 dev，以及独立 test。

### 训练曲线展示

训练 run 使用 8 张评估图，总体图位于 `evaluation/dev/*`，容量图位于 `evaluation/dev/by-capacity/*`。总体图中，NLL、EM、F1 和触顶率各使用一张历史折线图，以真实 optimizer step 为横轴，五种对照条件为五条曲线。NLL 包含全部 dev 评估点；EM、F1 和触顶率包含 step 0、250、500、750 的生成评估结果。容量图按同样的四个指标比较各 K 的 memory 表现。每张图可切换 `all`、`arrival`、`delayed`；总体 NLL、EM、F1 图还可切换 `paired` 查看配对差值。默认显示 `all`。完整评估数值保存在本地 JSON/JSONL。

已完成的 run 可从 `dev/dev-step-*.json` 重绘历史图。发布命令为 `.venv/bin/python -m latent_working_memory.v1.dynamic_reporting --training-run <train_run_dir> --media-step 753 --swanlab-mode online`。媒体发布位置 753 用于追加新图，图内横轴和实际训练终点均为 750；发布记录保存为 `evaluation-history-000753.json`。

重建展示 run 使用 `.venv/bin/python -m latent_working_memory.v1.dynamic_reporting --training-run <train_run_dir> --output-dir <series_dir>/plan/swanlab/dynamic_BPTT_squad_mixed-157k_20260912 --swanlab-mode online`。新 run 按原始 step 0–750 发布训练指标、8 张评估图、分组表格和问答样例，沿用原配置、group 和 tags。发布目录保存新 `swanlab.json` 与 `republication.json`，后者记录源训练目录、源 run ID 和评估步数；训练日志与 checkpoint 位于原训练目录。

当前展示 run：[dynamic_BPTT_squad_mixed-157k_20260912](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/ah4uvdar)。已恢复 750 步训练指标、10 次 dev 评估、8 张评估图、分组表格和问答样例，云端 750 个 loss 值已与原始日志逐项核对，绝对误差小于 `1e-12`。原 run `3sibkt6l` 保留。新 run 的创建时间和上传时间对应重建时刻，曲线以原始 optimizer step 为横轴。

当前展示 run 的全部 80 份图表记录均采用兼容格式，覆盖 10 个评估点，最新媒体记录和训练终点均为 step 750。云端回读核对记录保存在展示目录的 `cloud-verification.json`。更新已有展示 run 时，同时指定源训练目录 `--training-run`、展示目录 `--output-dir` 和新的 `--media-step`。组合图在顶层提供默认视图的 `series`，满足 SwanLab 对首次渲染曲线数量的校验。
