# 20260911_预训练数据类型对比实验

创建时间：20260911 16:27:19 UTC+08:00
最后修订时间：20260912 23:14:23 UTC+08:00

本实验比较 semantic、random 和等比例混合数据对 AE 重建与 LM 记忆读取的影响。三组使用相同初始化、训练配置和测试面板。方法与指标定义见 [预训练与 AE/LM 评估](20260910_pretraining_and_evaluation.md)。

## 1. 数据来源与对比设计

### 数据来源与配置

来源为 `data/v1/fineweb-4096-doc100k_20260910/`。从中筛选 X/Y 各不超过 2048 tokens、满足压缩及完整原文读取窗口的样本，保持原始 train/dev/test 划分，按任务与 X 长度档均衡选择。跨来源按任务及 X/Y 去重，train 去除 19 条、test 去除 1 条重复内容。

| 内容 | 位置 |
|---|---|
| 数据选择配置 | `configs/data_preparation/fineweb-4096-doc100k_pretrain-data-comparison-2048.json` |
| 训练配置 | `configs/v1/pretrain-data-comparison_fineweb-2048-doc100k.json` |
| 派生数据 | 来源目录下的 `derived/pretrain-data-comparison-2048_20260910/` |

派生目录归属于来源数据，名称记录实验用途、长度上限和创建日期；实际数量及来源数据身份保存在 `preparation.json` 和 `selection.json`。

```text
fineweb-4096-doc100k_20260910/
├── semantic/                       # 4096 上限的来源样本
├── random/
└── derived/
    └── pretrain-data-comparison-2048_20260910/
        ├── semantic/               # train/dev/test.jsonl、preparation.json
        ├── random/                 # train/dev/test.jsonl、preparation.json
        ├── mixed/                  # train.jsonl、preparation.json
        └── selection.json
```

### 训练组与样本配额

| 训练组 | semantic 占比 | random 占比 | 训练样本数 |
|---|---:|---:|---:|
| semantic | 100% | 0% | 157,320 |
| random | 0% | 100% | 157,320 |
| mixed | 50% | 50% | 157,320 |

三组共享 semantic/random 两套 dev/test。AE 的 X、LM 的 X 与 Y 均为 32–2048 tokens，LM 满足 0.3 ≤ |X| / (|X| + |Y|) ≤ 0.7。

| 数据划分 | 每组／每套总数 | 每个 X 长度档、每个任务的样本数 |
|---|---:|---:|
| train | 157,320 | 13,110；mixed 的两来源各 6,555 |
| dev | 3,468 | 289 |
| test | 3,108 | 259 |

semantic/random 各包含 163,896 条样本；mixed 保存训练划分，评估复用两套来源数据。

## 2. 训练与评估设置

### 模型与优化

| 配置项 | 设置 |
|---|---|
| 语言模型基座 | Llama-2-7B-Chat |
| 写入／读取窗口 | 各 4096 tokens |
| 记忆维度／容量上限 | 512／4096 个位置 |
| 记忆更新器 | 3 层，8 个注意力头，前馈维度 2048 |
| 读取 LoRA | rank=16，alpha=32，dropout=0；作用于 q_proj/v_proj |
| 名义压缩率 | 2、4、8，各以 1/3 概率采样 |
| 训练步数 | 20,000 次 optimizer 更新 |
| 全局 batch | 8：双卡 × 每卡 microbatch 2 × 梯度累积 2 |
| 数值精度／梯度检查点 | BF16／关闭 |
| 优化器 | AdamW，weight decay=0.01，梯度裁剪阈值 1.0 |
| 学习率 | 峰值 3e-5；600 步 warmup，余弦衰减至 3e-6 |
| AE／LM 损失权重 | 各 1 |
| 模型／训练采样／数据选择 seed | 42／20260907／20260910 |

长度采样概率在前 6,000 步线性过渡，之后保持目标分布：

| X 长度（tokens） | 起始概率 | 目标概率 |
|---|---:|---:|
| 32–64 | 25% | 1/6 |
| 65–128 | 25% | 1/6 |
| 129–256 | 20% | 1/6 |
| 257–512 | 15% | 1/6 |
| 513–1024 | 10% | 1/6 |
| 1025–2048 | 5% | 1/6 |

每组累计采样 160,000 次，约为训练集数量的 1.02 倍；各长度池的实际遍历次数由课程采样决定。

### 保存与评估

| 项目 | 设置 |
|---|---|
| dev 条件预测 | 初始化、每 1,000 步及最后一步；每套固定 120 条样本 |
| dev AE 自由生成 | 初始化、每 2,000 步；每套 12 条 AE 样本 |
| checkpoint | 每 1,000 步保存可训练参数、optimizer 与恢复状态 |
| 最终 test | step 20,000；semantic 实际 120 条、random 实际 117 条，均来自独立文档 |
| test AE 自由生成 | 每套 12 条 AE 样本，按三种压缩率展开 |
| AE 对照 | memory、wrong_memory、full_context、base_full_context |
| LM 对照 | AE 四种条件及 no_memory |
| AE 指标 | NLL、PPL、BLEU-4、正确前缀比例 |
| LM 指标 | NLL、PPL |

单模型展示总体六项指标及 memory 条件的长度 × 压缩率分组，共 12 张图；跨模型比较展示总体六项指标。三个模型与一个比较 run 共 42 张图。训练来源 semantic/random/mixed 分别使用蓝／橙／绿色系，测试来源使用同色系深浅色，图例显示完整来源名称。

## 3. 复现命令与产物

命令在服务器项目根目录 `/data/bywei/projects/latent_working_memory` 执行，使用根目录 `.venv`。重新运行使用新的输出目录。

### 选择实验数据

```bash
.venv/bin/python -m latent_working_memory.data_preparation.experiment \
  --spec configs/data_preparation/fineweb-4096-doc100k_pretrain-data-comparison-2048.json \
  --config configs/v1/pretrain-data-comparison_fineweb-2048-doc100k.json \
  --output-dir data/v1/fineweb-4096-doc100k_20260910/derived/pretrain-data-comparison-2048_20260910
```

`selection.json` 保存来源、配比、配额及数据目录映射；其中的 `evaluation_dirs` 用于训练与评估，本轮另存为实验产物下的 `plan/evaluation-dirs.json`。

### 训练与完整评估

本系列产物根目录为 `artifacts/v1/pretrain-data-comparison-2048_20260911/`。以 semantic 训练为例：

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m latent_working_memory.v1.train \
  --phase pretrain \
  --config configs/v1/pretrain-data-comparison_fineweb-2048-doc100k.json \
  --data-dir data/v1/fineweb-4096-doc100k_20260910/derived/pretrain-data-comparison-2048_20260910/semantic \
  --evaluation-dirs artifacts/v1/pretrain-data-comparison-2048_20260911/plan/evaluation-dirs.json \
  --output-dir artifacts/v1/pretrain-data-comparison-2048_20260911/train/pretrain-semantic-157k-20260911 \
  --max-steps 20000 --save-every 1000 \
  --swanlab-mode online --swanlab-project latent-working-memory-v1 \
  --swanlab-group lwm-boundary-comparison-2048-20260911 --swanlab-tag study:boundary-comparison
```

random、mixed 使用各自的数据与输出目录，其余训练参数相同。完整评估以 random 模型为例：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m latent_working_memory.v1.evaluate \
  --checkpoint artifacts/v1/pretrain-data-comparison-2048_20260911/train/pretrain-random-157k-20260911/checkpoints/pretrain-step-020000.pt \
  --evaluation-dirs artifacts/v1/pretrain-data-comparison-2048_20260911/plan/evaluation-dirs.json \
  --output-dir artifacts/v1/pretrain-data-comparison-2048_20260911/eval/pretrain-random-157k-eval-20260912 \
  --split test --examples 120 --generation-examples 12 \
  --swanlab-mode online --swanlab-project latent-working-memory-v1 \
  --swanlab-group lwm-boundary-comparison-2048-20260911 --swanlab-tag study:boundary-comparison
```

### 产物与 SwanLab

SwanLab project 为 `latent-working-memory-v1`，本轮 group 为 `lwm-boundary-comparison-2048-20260911`。训练 run 的 157k 表示训练样本数。以下路径相对本系列产物根目录：

| 子目录 | 内容 |
|---|---|
| `train/pretrain-{semantic,random,mixed}-157k-20260911/` | 训练记录及 `checkpoints/pretrain-step-020000.pt` |
| `eval/pretrain-{semantic,random,mixed}-157k-eval-20260912/` | 完整评估，各测试来源保存 `test-step-020000.json` 与 `.jsonl` |
| `compare/pretrain-157k-compare-20260912/` | 跨模型比较 |
| `plan/` | 启动命令、评估目录映射和 `complete-ae-reports.json` 报告清单 |

| 运行 | SwanLab |
|---|---|
| `pretrain-semantic-157k-eval-20260912` | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/19fgzh1r) |
| `pretrain-random-157k-eval-20260912` | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/81yl0lnd) |
| `pretrain-mixed-157k-eval-20260912` | [评估](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/d3z6hkfd) |
| `pretrain-157k-compare-20260912` | [比较](https://swanlab.cn/@percyWeeeeei/latent-working-memory-v1/runs/xltjprbj) |

## 4. 运行结果

三组均完成 20,000 步训练。正式训练代码提交为 `5b32bb4`，完整评估提交为 `5ded1be`。本轮 semantic/random/mixed 分别使用 GPU 0、1／4、5／6、7；训练及首次测试于 20260912 00:39:22 UTC+08:00 全部结束，完整评估于当日 14:13:26 完成。

以下采用完整评估报告的 memory 条件。NLL 按目标 token 加权且不含 EOS，越低越好；BLEU-4 为 0–100 标度，正确前缀比例以百分数表示，二者越高越好。

| 训练来源 | 测试来源 | AE NLL | LM NLL | AE BLEU-4 | AE 正确前缀比例 |
|---|---|---:|---:|---:|---:|
| semantic | semantic | 2.1312 | 1.9941 | 0.932 | 0.535% |
| semantic | random | 2.0756 | 2.2444 | 0.243 | 0.000% |
| random | semantic | 2.1584 | 2.0163 | 0.749 | 0.000% |
| random | random | 2.0627 | 2.2408 | 0.408 | 0.473% |
| mixed | semantic | 2.1232 | 1.9852 | 0.912 | 0.596% |
| mixed | random | 2.0440 | 2.2173 | 0.518 | 0.679% |

mixed 在两套测试的 AE/LM NLL 上均最低。三组 LM 的正确记忆均优于空记忆和错误记忆，但仍落后于各自的完整原文对照。AE 自由生成的 BLEU-4 和正确前缀比例仍很低，条件预测的改善尚未转化为忠实重建。结果来自单一模型 seed 和小规模固定测试面板，尚不足以判断差异的统计显著性。

| 训练来源 | 累计压缩输入 tokens | 累计目标 tokens |
|---|---:|---:|
| semantic | 64,634,170 | 56,669,206 |
| random | 65,695,227 | 55,479,358 |
| mixed | 65,114,675 | 56,028,257 |
