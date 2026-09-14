# 20260912_latent_working_memory

最后修订时间：20260914 10:16:54 UTC+08:00

本项目用于研究 streaming mutable latent working memory，并开展 matched-budget context compression 实验。论文复现与新方法分开管理；ICAE v1 和 C-DIC 分别位于 `reproductions/icae/` 与 `reproductions/cdic/`，各自使用独立的 `uv` 环境。

## 主环境与开发

数据构造、主实验训练与评估统一使用项目根目录 `.venv/`，依赖由根目录 `pyproject.toml` 与 `uv.lock` 管理。通用数据构造入口位于 `src/latent_working_memory/data_preparation/`，训练与实验代码按阶段放在 `src/latent_working_memory/v1/` 的对应子目录，正式配置位于 `configs/`；执行记录与日志写入 `artifacts/`。

数据构造按流程分包：[pretrain/](src/latent_working_memory/data_preparation/pretrain/) 负责 FineWeb 预训练文本构造、质量审查与恢复，[personamem/](src/latent_working_memory/data_preparation/personamem/) 负责事实 QA 构造。预训练的来源、句界、截断、文本格式、审查和恢复模块均位于 `pretrain/`；训练时的数据读取与实验选样继续使用 `v1/pretrain/prepared_data.py` 和 `v1/pretrain/data_selection.py`。

```bash
uv sync --frozen
uv run pytest
```

通用预训练构造入口为：

```bash
.venv/bin/python -m latent_working_memory.data_preparation.pretrain \
  --config configs/data_preparation/fineweb-4096-doc100k.json \
  --dataset-dir data/raw/HuggingFaceFW-fineweb \
  --output-dir /path/to/new-data \
  --stage all
```

`--stage` 支持 `sources`、`semantic`、`random` 和默认的 `all`；恢复未完成的 `semantic` 或 `random` 时使用原配置、原输出目录并追加 `--resume`。独立抽查入口为 `latent_working_memory.data_preparation.pretrain.inspection`，既有 Episode 语料的显式迁移工具为 `latent_working_memory.data_preparation.pretrain.migrate_text_samples`。原 `python -m latent_working_memory.data_preparation` 入口和根层预训练模块已迁入 `pretrain/`，调用方应更新模块路径；配置字段、文本格式、构造产物与恢复协议保持一致。

## 可增长记忆 v1

第一版新方法位于 `src/latent_working_memory/v1/`，按阶段分为 [pretrain/](src/latent_working_memory/v1/pretrain/)、[dynamic/](src/latent_working_memory/v1/dynamic/) 和 [capacity/](src/latent_working_memory/v1/capacity/)。根层保存模型、状态、checkpoint、通用数据与损失、训练辅助函数、SwanLab 会话、图表和命令执行等跨阶段能力，不导入阶段模块。阶段内部直接保存训练、评估与各实验入口文件，实验形成多份专用模块后再考虑子目录。容量阶段目前包含资源代价和策略运行基础实现，尚无完整训练入口。

动态阶段使用 `python -m latent_working_memory.v1.dynamic.prepare` 准备实验，`python -m latent_working_memory.v1.dynamic.run train` 和 `evaluate` 执行训练与评估；QA 报告入口为 `latent_working_memory.v1.dynamic.reporting`。SQuAD 与 PersonaMem 的运行时读取器同属 `v1/dynamic/`。阶段重构仅改变源码与入口路径，既有配置字段、checkpoint、训练日志和数据格式保持一致；旧入口不保留转发层。

配置位于 `configs/v1/`，测试位于 `tests/v1/`，数据与运行产物分别使用 Git-ignored 的 `data/` 和 `artifacts/v1/`。实验产物按系列组织为 `artifacts/v1/<实验系列>/`，其中 `train/` 保存训练及 checkpoint，`eval/` 保存独立评估，`compare/` 保存跨运行比较，`plan/` 保存调度与清单。当前预训练数据类型对比系列位于 `artifacts/v1/pretrain-data-comparison-2048_20260911/`，完整记录及路径见 [预训练数据类型对比实验](notes/v1/20260911_pretraining_data_comparison.md)。数据准备与工程验证产物仍分别位于 `artifacts/v1/data-preparation/` 和 `artifacts/v1/validation/`。

FineWeb 基础语料的 `semantic/`、`random/` 各只保存 `train.jsonl`、`dev.jsonl`、`test.jsonl` 和 `preparation.json`。样本保存原始 X、LM 后续 Y、来源与字符跨度，以及构造 tokenizer 的参考长度；AE 不重复保存目标，不落盘 token IDs、训练提示词或读写位置。父目录的 `source-pool.json` 只记录原始 Parquet 文件路径、随机种子、构造规则和统计，不保存来源正文副本。构造、恢复构造及原文边界检查按该记录重新读取原始文件，在内存中重建候选来源与划分；训练和评估不需要原始文件。迁移原始文件位置后需相应更新 `source_files` 路径。

训练和评估可直接读取基础目录：同 tokenizer 复用参考长度筛选，不同 tokenizer 重新分词计算长度，采样时按当前提示词构造 Episode，不生成分词副本。构造未完成时用 `progress.json` 和临时接收记录支持恢复，完成后自动清理。已有独立构造实验仍可读取原 Episode 数据。仅筛选与混合时使用 `--data-selection <配置> --data-run <训练集名>`，评估使用相同的 `--data-selection`；来源、seed、配额及比例由配置指定，混合仅组合内存索引。`balance_task_lengths` 决定是否均衡任务与长度档，`samples_per_split` 指定各划分数量，非均衡模式可用 null 表示全部取用（混合训练须指定数量以保证比例）。run 中的 `data-selection.json` 保存可直接复用的选择配置，实际数量与来源身份保存在 provenance 中，不生成样本清单或 token 副本。

SQuAD 的 `data/squad/` 只保存 `train.jsonl`、`dev.jsonl`、`test.jsonl` 和 `preparation.json`。划分文件每行对应一篇文章，保存来源位置、来源组、段落／问题数量与 `reference_*` 长度，不复制原文和问答。准备信息集中记录来源、划分规则、排除文章和参考 tokenizer。构造配置为 `configs/data_preparation/squad.json`。动态准备、训练和评估统一使用 `--dataset squad --dataset-dir data/squad`，tokenizer 由实验 checkpoint 提供；匹配参考分词行为时复用长度，否则在内存中重新计算。

短文本目标对比使用独立构造的 `data/fineweb-128-doc100k_20260912/{semantic,random}/` 文本基础数据，复用原 FineWeb 来源池。构造配置为 `configs/data_preparation/fineweb-128-doc100k.json`，选择配置为 `configs/data_preparation/fineweb-128-doc100k_pretrain-objective-comparison.json`；不再保存该实验的派生目录和 mixed 副本。

当前已实现 FineWeb 完整句界／随机截断双版本数据与多容量 AE/LM 预训练链路：空记忆首次分配、完整自然单元前向、变长 batch 与 mask、独立 AE/LM 样本的重建和续写、按长度与文档采样、容量课程、checkpoint/resume，以及独立文档的多容量 memory/no-memory/wrong-memory 评估。语言模型基座冻结，联合训练写入投影、记忆更新器、读取投影与读取 LoRA。

v1 测试覆盖损失与梯度、原文边界、规则句界、任务配额与长度区间、来源隔离、独立抽查、分层评估、自由生成、原文对照的因果位置、BLEU 聚合、精确恢复和 SwanLab 记录。FineWeb `sample-10BT` 已下载到服务器；512 篇真实文档的多粒度准备、Llama-2-7B-Chat 单卡训练、保存恢复和多容量评估已跑通。后续实现顺序见 [框架设计与后续阶段](notes/20260907_growing_latent_working_memory_framework_v1.md)。

固定 16 样本的 100 步试验完成训练集拟合验证；独立文档表现呈现过拟合，该 checkpoint 用于工程验证，扩大数据试验从统一初始化开始验证泛化收益。

历史扩大数据预训练 已准备 10,000／512／512 篇训练、验证和测试文档，训练集包含 126,278 个 AE/LM 样本对。1e-4 与 3e-5 两组均已完成 2000 步实验，每组访问 10,000 篇不同文档和约 140 万输入 tokens。dev 选择 3e-5 第 2000 步 checkpoint，512 文档独立 test 已完成：AE/LM NLL 为 2.2532/2.3914，相对空记忆收益为 0.1409/0.1242；64 文档、三个容量的自由重建仍为 0/192 完整匹配，平均归一化 token 编辑距离为 0.9391。记忆已辅助条件预测，原文重建能力仍需改进。该实验增加句段质量筛选、基座流畅度过滤和来源抽查，记录初始化验证、128 文档的周期性 NLL 与 64 文档的自由重建。

当前预训练数据使用 FineWeb 的独立 AE/LM 任务与 semantic/random 两种边界，按 32–4096 tokens 的七个输入长度区间均衡构造。数据来源、规则、代码、运行命令和本轮 408,800 条目标配额统一见 [预训练数据构建](notes/v1/20260910_pretraining_data_construction.md)。正式训练优先使用 semantic；最终产量及抽查结果在构建完成后补全。

数据准备仅检查数据长度与来源契约；训练时检查实际基座窗口、输入/目标预算和合法 memory 容量。当前默认 memory 上限为 4096，压缩率为 2、4、8。4k 数据的训练需要配置可覆盖 4096-token 写入及 `memory + target + prompt + special tokens` 读取预算的基座；下面的训练入口用于已完成相应预算适配的数据和训练配置。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m latent_working_memory.v1.pretrain.train \
  --phase pretrain \
  --config configs/v1/pretrain_a800.json \
  --data-dir data/fineweb-independent/semantic \
  --output-dir artifacts/v1/pretrain-pilot/train/pretrain-pilot \
  --max-steps 1000
```

`--train-example-limit 16` 固定一个优先覆盖不同文档的小样本池，供过拟合检查使用；常规训练使用全部已准备样本。恢复时增加 `--resume artifacts/v1/pretrain-pilot/train/pretrain-pilot/checkpoints/pretrain-step-000100.pt`，并把 `--max-steps` 设为新的总 step 上限。checkpoint 保存完整可训练模块、优化器、文档采样与容量课程进度及随机状态。运行产物包含逐样本容量和损失、文档覆盖、token 监督量、吞吐、显存及分层验证指标。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m latent_working_memory.v1.pretrain.evaluate \
  --checkpoint artifacts/v1/pretrain-pilot/train/pretrain-pilot/checkpoints/pretrain-step-001000.pt \
  --data-dir data/fineweb-independent/semantic \
  --output-dir artifacts/v1/pretrain-pilot/eval/pretrain-pilot-test \
  --split test
```

评估使用固定的独立文档面板，每篇选一个自然片段，遍历各压缩率对应的唯一合法容量。结果按实际划分写入 `{dev|test}-step-XXXXXX.json` 与 `.jsonl`，分别保存聚合指标和逐次读取记录；摘要包含 step、累计训练输入 tokens、评估口径和 SacreBLEU 签名。

| 评估项 | 口径 |
|---|---|
| AE/LM 条件预测 | 正确、空、错误记忆的 NLL、PPL、teacher-forcing token accuracy；NLL 按目标正文 token 数加权，另存包含 EOS 的 NLL |
| AE 自由重建 | 贪心生成，最多生成参考正文长度加一个 EOS 位置；完整序列匹配包含 EOS，连续正确前缀比例与归一化编辑距离使用正文 tokens，并按生成读取求均值 |
| AE BLEU-4 | SacreBLEU 语料级统计，范围 0–100，13a 分词、区分大小写、指数平滑；短句组启用 effective order，实际配置签名随结果保存 |
| LM 原文对照 | `full_context` 读取完整 X，`recent_context` 读取 X 最后的 K 个 tokens；二者使用当前 reader LoRA，`base_full_context` 单独关闭 LoRA |
| LM 对照差值 | `gain_vs_* = NLL(对照) − NLL(memory)`，正值表示记忆收益；`nll_gap_to_* = NLL(memory) − NLL(完整原文)`，同时记录对应 PPL 比值 |

LM 的全部条件预测相同的 Y，并使用同一任务提示。最近文本与记忆按读取时的 K 个上下文位置对齐，原文不足 K 时使用全部 X；`memory_bytes` 单列记忆张量的字节数。完整原文对照每篇计算一次，在各容量上按相同的文档、片段与目标 token 权重参与比较。完整原文与提示、目标共同受基座上下文上限约束。分层结果覆盖粒度、输入长度、容量、实际压缩率和句界来源。

补齐后的评估已使用上轮 3e-5 第 2000 步 checkpoint 在物理 GPU 0 验证：16 篇 dev 文档、每篇三个容量，共完成 432 次条件读取与 24 次自由重建，退出码为 0。SwanLab 离线记录完成；本次结果位于 `artifacts/v1/validation/evaluation-smoke-20260909/`。

### SwanLab 可视化

训练或独立评估命令增加 `--swanlab-mode online`，使用服务器已有登录。`--swanlab-mode offline` 将记录保存在运行目录；默认值为 `disabled`。项目名由 `--swanlab-project` 指定，默认 `latent-working-memory`，新建项目为私有。

看板记录 AE/LM 损失、梯度范数、吞吐和显存、输入长度与记忆容量、文档覆盖，以及各评估条件的 NLL、PPL、准确率和对照差值。长度、粒度、容量与压缩率分别提供分层 NLL、自由重建指标及读取数量。`progress/input_tokens` 与评估 step 同步记录，支持按训练曝光量分析学习曲线。

自由生成记录 BLEU-4、连续正确前缀比例、完整匹配和归一化 token 编辑距离；文本样例按每页 100 条记录。`eval_generation_every` 控制生成评估频率；独立评估可用 `--examples 512 --generation-examples 128` 扩大面板。SwanLab 配置包含本次数据准备与质量筛选记录，独立评估额外记录 checkpoint 路径与实际划分。

`swanlab.json` 保存实验 ID 和链接；在同一输出目录恢复训练时沿用该实验。独立评估可通过 `--swanlab-run-id` 写入对应训练实验，并使用 checkpoint 的 step 作为横轴。可视化参数由命令行管理，与模型配置分开保存。[SwanLab 初始化与续接接口](https://docs.swanlab.cn/api/py-init.html)

已有 100 步小样本试验可在 [SwanLab 看板](https://swanlab.cn/@percyWeeeeei/latent-working-memory/runs/hzg2z87k/chart) 查看。历史日志按 optimizer step 导入，时间轴显示本次导入时间。

## 服务器目录

项目相关的数据、checkpoint 和实验产物均放在服务器项目根目录 `/data/bywei/projects/latent_working_memory` 下：

```text
data/raw/pwc/          PwC 原始数据
data/raw/msc/          MSC 原始数据与归档
data/raw/HuggingFaceFW-fineweb/  FineWeb 原始 Parquet，子目录为 sample-10BT
data/                 共享数据集与原始来源，不按方法版本分层
checkpoints/icae/v1/  ICAE v1 公开 checkpoint
checkpoints/cdic/      C-DIC pilot 与完整训练 checkpoint
artifacts/             生成结果、测试报告和日志
```

上述目录均已由 `.gitignore` 排除。Llama-2-7B-Chat 基础模型仍保存在共享目录 `/data/bywei/models/`，Hugging Face 与 `uv` cache 仍保存在 `/data/bywei/cache/`。

迁移前位于 `/data/bywei/datasets/`、`/data/bywei/checkpoints/` 和 `/data/bywei/logs/cdic/` 下的相关旧路径暂时保留为 symlink，以兼容已有 checkpoint 中记录的绝对路径；新配置统一使用项目内路径。

## ICAE 环境

ICAE lock 面向 Linux x86-64，使用与原始代码依赖栈兼容的 PyTorch CUDA 11.8 wheel。服务器驱动支持 CUDA 13.0，并可向后兼容该 runtime。环境安装不会自动下载模型或数据。

```bash
uv sync --project reproductions/icae --frozen
uv run --project reproductions/icae icae-check-environment
```

在 GPU 服务器运行前，参见 `reproductions/icae/README.md`。

## C-DIC 复现

C-DIC 的论文实现位于 `reproductions/cdic/`，复现实验统一记录在 SwanLab 的 `latent-working-memory-cdic-repro` project 中。当前已完成 retrieval、recency、write-back、state lineage、trace、one-hop credit assignment、ICAE adapter、MSC loader、双卡训练与 checkpoint/resume。A800 多轮 GPU smoke test、双卡 MSC pilot 和 seed 42 的两 epoch 完整训练均已完成。

[20260906 评估口径对齐](notes/reproduction_results/20260906_c_dic_evaluation_alignment_record.md) 已在相同 32 个 validation episodes 上比较初始化与训练后模型，并分列轮次范围、EOS 分母和 ROUGE recall/F1。全部轮次 PPL 从 16.9929 升至 23.0159，session 5 最后一轮从 22.5589 升至 28.2935；当前结果不支持论文效果已复现。

```bash
PYTHONPATH=reproductions/cdic/src uv run pytest -q reproductions/cdic/tests
uv sync --project reproductions/cdic --frozen
```

### 预训练实验入口

不同实验的入口文件直接放在 `v1/pretrain/`，共用该阶段的 `train.py`、`evaluate.py`、`publish_reports.py` 和 `data_selection.py`。根层的 `v1/experiment_execution.py` 只负责命令执行、日志、状态和 GPU 分配检查，不决定实验步骤。

- 数据类型对比：`v1/pretrain/data_comparison.py`，分别训练 semantic、random、mixed。
- 目标对比：`v1/pretrain/objective_comparison.py`，分别训练 AE-only、联合训练和 AE warm-up；专用短文本构造入口为同目录的 `prepare_objective_data.py`。

```bash
.venv/bin/python -m latent_working_memory.v1.pretrain.data_comparison \
  --spec configs/experiments/pretrain-data-comparison-2048.json \
  --output-dir artifacts/v1/<新的数据对比实验目录>

.venv/bin/python -m latent_working_memory.v1.pretrain.objective_comparison \
  --spec configs/experiments/pretrain-objective-comparison-128.json \
  --output-dir artifacts/v1/<新的目标对比实验目录>
```

两者均串行执行双卡训练，按来源单卡并行测试，将最终 test 追加到训练 run，并单独发布跨组比较。正式配置显式指定 GPU、保存间隔与测试数量；目标对比另行指定 warm-up 继承步数和 prefix 诊断。旧 `artifacts/` 下的调度脚本仅为历史记录，不作为启动入口。
