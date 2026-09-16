# 20260912_latent_working_memory

最后修订时间：20260916 21:38:59 UTC+08:00

本项目用于研究 streaming mutable latent working memory，并开展 matched-budget context compression 实验。论文复现与新方法分开管理；ICAE v1 和 C-DIC 分别位于 `reproductions/icae/` 与 `reproductions/cdic/`，各自使用独立的 `uv` 环境。

## 主环境与开发

通用执行约定见项目 `AGENTS.md`；命名、SwanLab、实验产物及研究笔记规则见 [项目规范索引](docs/README.md)。

数据构造、主实验训练与评估统一使用项目根目录 `.venv/`，依赖由根目录 `pyproject.toml` 与 `uv.lock` 管理。通用数据构造入口位于 `src/latent_working_memory/data_preparation/`，训练与实验代码按阶段放在 `src/latent_working_memory/v1/` 的对应子目录，正式配置位于 `configs/`；执行记录与日志写入 `artifacts/`。

创建和整理配置遵守 [配置组织规则](configs/README.md)：基础数据准备与实验选样分开，每个配置目录只描述一个具体实验，实验组通过运行记录关联。

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

## GMSA 底座与 v2

v2 位于 [src/latent_working_memory/v2/](src/latent_working_memory/v2/)。当前统一读写框架为 `memory_codec.py`：旧记忆与完整新文本联合编码，首次压缩和后续更新共享参数，支持均值、加权和谱域三种压缩模块。`pretrain/` 使用 verl replicated/DDP 执行固定容量 AE／AE＋LM 与 warm-up／直接多次压缩四组训练，数据在启动时构造一次，各 epoch 完整复用。初始 GMSA 静态迁移及 `working_memory.py` 更新原型保留用于对照；容量策略尚未实现。

当前完成小型随机 Llama／Qwen3 工程验证与上游数值比较，尚未复现真实 GMSA 指标。训练命令、配置、数据契约和开发范围见 [v2 开发记录](src/latent_working_memory/v2/README.md)，上游版本及适配区别见 [迁移说明](src/latent_working_memory/v2/UPSTREAM.md)。v2 使用根目录环境；现有 v1 和 C-DIC 路径保持独立。

## 可增长记忆 v1

第一版新方法位于 `src/latent_working_memory/v1/`，按阶段分为 [pretrain/](src/latent_working_memory/v1/pretrain/)、[dynamic/](src/latent_working_memory/v1/dynamic/) 和 [capacity/](src/latent_working_memory/v1/capacity/)。根层保存模型、状态、checkpoint、通用数据与损失、训练辅助函数、SwanLab 会话、图表和命令执行等跨阶段能力，不导入阶段模块。阶段内部直接保存训练、评估与各实验入口文件，实验形成多份专用模块后再考虑子目录。容量阶段目前包含资源代价和策略运行基础实现，尚无完整训练入口。

v1 预训练、完整 BPTT 与 TBPTT 统一使用 verl 0.8.0 自定义 engine，当前多卡后端为 DDP。循环 memory 的计算图、截断边界和样本归一化由阶段实现管理，optimizer 生命周期由 verl engine 统一执行。配置、精度与恢复规则见 [v1 训练执行框架](configs/README.md#v1-训练执行框架)。重构验证结论见 [重构记录](notes/v1/20260915_verl_v1_refactor.md)。

动态阶段使用 `python -m latent_working_memory.v1.dynamic.prepare` 准备实验，`python -m latent_working_memory.v1.dynamic.run train` 和 `evaluate` 执行训练与评估；QA 报告入口为 `latent_working_memory.v1.dynamic.reporting`。SQuAD 与 PersonaMem 的运行时读取器同属 `v1/dynamic/`。阶段重构仅改变源码与入口路径，既有配置字段、checkpoint、训练日志和数据格式保持一致；旧入口不保留转发层。

正式配置位于 `configs/v1/<阶段>/<具体实验名>/`，测试位于 `tests/v1/`，数据与运行产物分别使用 Git-ignored 的 `data/` 和 `artifacts/v1/`。当前 v1 训练实验入口的产物按系列组织为 `artifacts/v1/<实验系列>/`，其中 `train/` 保存训练及 checkpoint，`eval/` 保存独立评估，`compare/` 保存跨运行比较，`plan/` 保存调度与清单。当前预训练数据类型对比系列位于 `artifacts/v1/pretrain-data-comparison-2048_20260911/`，完整记录及路径见 [预训练数据类型对比实验](notes/v1/20260911_pretraining_data_comparison.md)。数据准备与工程验证产物仍分别位于 `artifacts/v1/data-preparation/` 和 `artifacts/v1/validation/`。

FineWeb 基础语料的 `semantic/`、`random/` 各只保存 `train.jsonl`、`dev.jsonl`、`test.jsonl` 和 `preparation.json`。样本保存原始 X、LM 后续 Y、来源与字符跨度，以及构造 tokenizer 的参考长度；AE 不重复保存目标，不落盘 token IDs、训练提示词或读写位置。父目录的 `source-pool.json` 只记录原始 Parquet 文件路径、随机种子、构造规则和统计，不保存来源正文副本。构造、恢复构造及原文边界检查按该记录重新读取原始文件，在内存中重建候选来源与划分；训练和评估不需要原始文件。迁移原始文件位置后需相应更新 `source_files` 路径。

训练与评估通过 `--data-selection <具体实验的 selection.json>` 引用共享数据，在内存中筛选与混合。预训练选样每次用当前 tokenizer 批量计算长度；训练按 batch 即时分词并有限预取，不保存派生文本或 token 副本。每个 epoch 根据来源、任务和长度分布，选择满足精确比例与全局 batch 整除的最大无放回样本量，打乱后组织 batch。dev/test 由独立选择规则固定。配置格式见 [配置组织规则](configs/README.md)，完整协议见 [预训练与评估](notes/v1/20260910_pretraining_and_evaluation.md)。

SQuAD 的 `data/squad/` 只保存 `train.jsonl`、`dev.jsonl`、`test.jsonl` 和 `preparation.json`。划分文件每行对应一篇文章，保存来源位置、来源组、段落／问题数量与 `reference_*` 长度，不复制原文和问答。准备信息集中记录来源、划分规则、排除文章和参考 tokenizer。构造配置为 `configs/data_preparation/squad.json`。动态准备、训练和评估统一使用 `--dataset squad --dataset-dir data/squad`，tokenizer 由实验 checkpoint 提供；匹配参考分词行为时复用长度，否则在内存中重新计算。

短文本目标对比使用独立构造的 `data/fineweb-128-doc100k_20260912/{semantic,random}/` 文本基础数据，复用原 FineWeb 来源池。构造配置为 `configs/data_preparation/fineweb-128-doc100k.json`，选择配置位于各 `configs/v1/pretrain/llama-<目标>_mixed-128/selection.json`；不再保存该实验的派生目录和 mixed 副本。

当前已实现 FineWeb 完整句界／随机截断双版本数据与多容量 AE/LM 预训练链路：空记忆首次分配、完整自然单元前向、变长 batch 与 mask、独立 AE/LM 样本的重建和续写、epoch 最大配额选样、容量抽样或加权平均、checkpoint/resume，以及独立文档的多容量 memory/no-memory/wrong-memory 评估。语言模型基座冻结，联合训练写入投影、记忆更新器、读取投影与读取 LoRA。

v1 测试覆盖损失与梯度、原文边界、规则句界、任务配额与长度区间、来源隔离、独立抽查、分层评估、自由生成、原文对照的因果位置、BLEU 聚合、精确恢复和 SwanLab 记录。FineWeb `sample-10BT` 已下载到服务器；512 篇真实文档的多粒度准备、Llama-2-7B-Chat 单卡训练、保存恢复和多容量评估已跑通。后续实现顺序见 [框架设计与后续阶段](notes/20260907_growing_latent_working_memory_framework_v1.md)。

固定 16 样本的 100 步试验完成训练集拟合验证；独立文档表现呈现过拟合，该 checkpoint 用于工程验证，扩大数据试验从统一初始化开始验证泛化收益。

历史扩大数据预训练 已准备 10,000／512／512 篇训练、验证和测试文档，训练集包含 126,278 个 AE/LM 样本对。1e-4 与 3e-5 两组均已完成 2000 步实验，每组访问 10,000 篇不同文档和约 140 万输入 tokens。dev 选择 3e-5 第 2000 步 checkpoint，512 文档独立 test 已完成：AE/LM NLL 为 2.2532/2.3914，相对空记忆收益为 0.1409/0.1242；64 文档、三个容量的自由重建仍为 0/192 完整匹配，平均归一化 token 编辑距离为 0.9391。记忆已辅助条件预测，原文重建能力仍需改进。该实验增加句段质量筛选、基座流畅度过滤和来源抽查，记录初始化验证、128 文档的周期性 NLL 与 64 文档的自由重建。

当前预训练数据使用 FineWeb 的独立 AE/LM 任务与 semantic/random 两种边界，按 32–4096 tokens 的七个输入长度区间均衡构造。数据来源、规则、代码、运行命令和本轮 408,800 条目标配额统一见 [预训练数据构建](notes/v1/20260910_pretraining_data_construction.md)。正式训练优先使用 semantic；最终产量及抽查结果在构建完成后补全。

数据准备仅检查数据长度与来源契约；训练时检查实际基座窗口、输入/目标预算和合法 memory 容量。当前默认 memory 上限为 4096，压缩率为 2、4、8。4k 数据的训练需要配置可覆盖 4096-token 写入及 `memory + target + prompt + special tokens` 读取预算的基座；下面的训练入口用于已完成相应预算适配的数据和训练配置。

```bash
CUDA_VISIBLE_DEVICES=4 uv run --frozen python -m latent_working_memory.v1.pretrain.train \
  --phase pretrain \
  --config configs/v1/pretrain/llama-semantic-2048/model.json \
  --data-selection configs/v1/pretrain/llama-semantic-2048/selection.json \
  --output-dir artifacts/v1/pretrain-pilot/train/pretrain-pilot \
  --epochs 3
```

每轮样本量、实际配额与步数保存在 `epoch-plan.json`。临时限制执行可用 `--stop-after-steps <累计步数>`，恢复时保持原 `--epochs`、配置、输出目录与进程数，增加 `--resume <该运行 checkpoint>`，并删除或提高临时步数上限。checkpoint 保存模型、optimizer、epoch 顺序与游标、选样及容量随机状态。旧 step 协议 checkpoint 不用于新协议精确续训。

```bash
CUDA_VISIBLE_DEVICES=4 uv run --frozen python -m latent_working_memory.v1.pretrain.evaluate \
  --training-result artifacts/v1/pretrain-pilot/train/pretrain-pilot/training-result.json \
  --data-selection configs/v1/pretrain/llama-semantic-2048/selection.json \
  --output-dir artifacts/v1/pretrain-pilot/eval/pretrain-pilot-test \
  --split test
```

`training-result.json` 记录实际最终 checkpoint；该入口仅对完整训练执行最终 test。阶段性诊断可直接传入 `--checkpoint <路径>`。

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

项目选择、运行组织和展示细则见 [SwanLab 使用规范](docs/swanlab.md)。

训练或独立评估命令增加 `--swanlab-mode online`，使用服务器已有登录。`--swanlab-mode offline` 将记录保存在运行目录；默认值为 `disabled`。项目名由 `--swanlab-project` 指定，默认 `latent-working-memory-v1`，新建项目为私有；创建新项目遵循项目约定。

看板记录 AE/LM 损失、梯度范数、吞吐和显存、输入长度与记忆容量、文档覆盖，以及各评估条件的 NLL、PPL、准确率和对照差值。长度、粒度、容量与压缩率分别提供分层 NLL、自由重建指标及读取数量。`progress/input_tokens` 与评估 step 同步记录，支持按训练曝光量分析学习曲线。

自由生成记录 BLEU-4、连续正确前缀比例、完整匹配和归一化 token 编辑距离；文本样例按每页 100 条记录。`eval_generation_every` 控制生成评估频率；独立评估可用 `--examples 512 --generation-examples 128` 扩大面板。SwanLab 配置包含本次数据准备与质量筛选记录，独立评估额外记录 checkpoint 路径与实际划分。

`swanlab.json` 保存实验 ID 和链接；在同一输出目录恢复训练时沿用该实验。恢复已有 run 时显式使用其原项目和 group；独立评估可通过 `--training-run` 指定原训练目录并追加结果，使用 checkpoint 的 step 作为横轴。可视化参数由命令行管理，与模型配置分开保存。[SwanLab 初始化与续接接口](https://docs.swanlab.cn/api/py-init.html)

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

### 独立实验入口

配置规则与完整目录清单见 [configs/README.md](configs/README.md)。预训练与动态阶段分别使用 `v1.pretrain.experiment` 和 `v1.dynamic.experiment`，每个实验读取自身 `experiment.json`，独立完成训练及最终 test。预训练目标对比的短文本构造入口仍为 `v1.pretrain.prepare_objective_data`，只读取数据准备配置。

```bash
.venv/bin/python -m latent_working_memory.v1.pretrain.experiment \
  --experiments configs/v1/pretrain/llama-semantic-2048/experiment.json \
                configs/v1/pretrain/llama-random-2048/experiment.json \
                configs/v1/pretrain/llama-mixed-2048/experiment.json \
  --output-dir artifacts/v1/<新系列> \
  --swanlab-group <本次分组> --swanlab-mode online \
  --swanlab-tag study:boundary-comparison
```

只选一个配置即可单独运行；AE-only、joint、AE warm-up 各自训练，warm-up 不依赖 AE-only。动态实验的预训练 checkpoint 依赖在自身配置中声明。增加 `--plan-only` 可先生成命令与配置快照，不执行训练或连接 SwanLab。计划检查不代表真实数据、显存与训练行为已在服务器验证。

最终 test 在 online 模式追加到对应训练 run。跨实验比较由报告入口读取 `<系列>/plan/reports.json` 显式发布；分组和比较不需要额外组配置目录。旧试跑配置统一放在 [configs/archive/](configs/archive/README.md)。
