# latent_working_memory（20260909 20:13:30 UTC+08:00）

最后修订时间：20260909 20:13:30 UTC+08:00

本项目用于研究 streaming mutable latent working memory，并开展 matched-budget context compression 实验。论文复现与新方法分开管理；ICAE v1 和 C-DIC 分别位于 `reproductions/icae/` 与 `reproductions/cdic/`，各自使用独立的 `uv` 环境。

## 本地开发

```bash
uv sync --frozen
uv run pytest
```

## 可增长记忆 v1

第一版新方法直接位于 `src/latent_working_memory/v1/`；未来版本使用同级目录，不增加额外的方法族目录。配置位于 `configs/v1/`，测试位于 `tests/v1/`，数据、checkpoint 与运行产物分别使用 Git-ignored 的 `data/v1/`、`checkpoints/v1/` 和 `artifacts/v1/`。本地运行产物按数据准备、工程验证、训练实验、独立评估和历史辅助文件分类，目录索引见 [产物整理记录](notes/v1/20260909_artifact_organization.md)。

当前已实现 FineWeb 完整句界／随机截断双版本数据与多容量 AE/LM 预训练链路：空记忆首次分配、完整自然单元前向、变长 batch 与 mask、独立 AE/LM 样本的重建和续写、按长度与文档采样、容量课程、checkpoint/resume，以及独立文档的多容量 memory/no-memory/wrong-memory 评估。语言模型基座冻结，联合训练写入投影、记忆更新器、读取投影与读取 LoRA。

v1 的 65 项测试通过，覆盖损失与梯度、原文边界、模型评分接口、任务配额与长度区间、来源隔离、独立抽查、分层评估、自由生成、原文对照的因果位置、BLEU 聚合、精确恢复和 SwanLab 记录。FineWeb `sample-10BT` 已下载到服务器；512 篇真实文档的多粒度准备、Llama-2-7B-Chat 单卡训练、保存恢复和多容量评估已跑通。实验结果见 [FineWeb 预训练记录](notes/v1/20260908_fineweb_pretraining_pilot_record.md)，后续实现顺序见 [v1 实施计划](notes/v1/20260907_growing_latent_working_memory_implementation_plan.md)。

固定 16 样本的 100 步试验完成训练集拟合验证；独立文档表现呈现过拟合，该 checkpoint 用于工程验证，扩大数据试验从统一初始化开始验证泛化收益。

[扩大数据预训练](notes/v1/20260908_fineweb_generalization_pretraining_record.md) 已准备 10,000／512／512 篇训练、验证和测试文档，训练集包含 126,278 个 AE/LM 样本对。1e-4 与 3e-5 两组均已完成 2000 步实验，每组访问 10,000 篇不同文档和约 140 万输入 tokens。dev 选择 3e-5 第 2000 步 checkpoint，512 文档独立 test 已完成：AE/LM NLL 为 2.2532/2.3914，相对空记忆收益为 0.1409/0.1242；64 文档、三个容量的自由重建仍为 0/192 完整匹配，平均归一化 token 编辑距离为 0.9391。记忆已辅助条件预测，原文重建能力仍需改进。该实验增加句段质量筛选、基座流畅度过滤和来源抽查，记录初始化验证、128 文档的周期性 NLL 与 64 文档的自由重建。

数据准备代码位于 `src/latent_working_memory/data_preparation/`。当前流程先执行简单属性检查和文档去重，固定共享来源划分，再独立构造 `semantic` 和 `random`，由 Qwen3.8-27B 判定最终 X/Y 的质量。两套数据各自按 split 补足等量 AE/LM；random 根据 semantic 入选后的各任务输入长度区间分布构造。流程、代码职责与命令见 [独立 AE/LM 数据构造实现](notes/v1/20260909_independent_ae_lm_data_preparation.md)。

此前完成的两套逐样本配对数据，每套包含 226,022／11,610／11,685 个 train/dev/test 样本，作为历史产物保留，见 [配对数据构造记录](notes/v1/20260909_paired_boundary_pretraining_data.md)。本次独立构造流程已通过本地接口测试，正式数据生成在评分服务验证后执行。

`configs/v1/pretrain_a800.json` 保存模型与训练采样设置，`configs/data_preparation/fineweb.json` 保存准备配方。`samples_per_task=[100000,2000,2000]` 表示每个版本的 train/dev/test 中，AE 与 LM 分别达到对应数量；`length_bounds` 默认定义 1–64、65–128、129–256、257–512、513–1024 tokens 五档。评分服务就绪后执行：

```bash
uv run python -m latent_working_memory.data_preparation \
  --config configs/v1/pretrain_a800.json \
  --recipe configs/data_preparation/fineweb.json \
  --dataset-dir data/raw/HuggingFaceFW-fineweb \
  --output-dir data/v1/fineweb-independent \
  --score-cache data/v1/quality-cache/sample-scores.jsonl
```

配方通过 `review_base_url` 与 `review_model` 连接独立评分服务，默认使用直连接口 `http://127.0.0.1:8000/v1` 和模型名 `Qwen/Qwen3.8-27B`。该模型权重已下载到服务器 `/data/bywei/models/Qwen/Qwen3.8-27B`。数据准备入口调用 HTTP 服务，tokenizer 从已有本地模型加载。

完整自然片段 S 用于 AE；在 S 的合法内部句界抽取切点，前缀 X 写入记忆、后缀 Y 作为独立 LM 样本的全部监督目标。单句片段提供 AE。random 独立选择文档和原文跨度，按任务及输入区间补足模型判定后的配额。两类提示词使用共同内容标准，并分别处理完整句界与随机截断的边界要求。

`--stage sources` 建立共享来源池；`--stage semantic` 与 `--stage random` 分别执行构造、模型判定和检查，后者读取已完成 semantic 的分布。根目录保存来源登记和 `comparison.json`，各版本保存三个 split、入选原文、候选判定、`audit.json` 与完成记录 `preparation.json`。质量抽查通过独立模块 `latent_working_memory.data_preparation.inspection` 的 `sample`、`judge`、`summarize` 执行，支持模型及人工复核。正式训练验证的数据入口为 `semantic/`。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m latent_working_memory.v1.train \
  --phase pretrain \
  --config configs/v1/pretrain_a800.json \
  --data-dir data/v1/fineweb-independent/semantic \
  --output-dir artifacts/v1/experiments/pretrain-pilot \
  --max-steps 1000
```

`--train-example-limit 16` 固定一个优先覆盖不同文档的小样本池，供过拟合检查使用；常规训练使用全部已准备样本。恢复时增加 `--resume artifacts/v1/experiments/pretrain-pilot/checkpoints/pretrain-step-000100.pt`，并把 `--max-steps` 设为新的总 step 上限。checkpoint 保存完整可训练模块、优化器、文档采样与容量课程进度及随机状态。运行产物包含逐样本容量和损失、文档覆盖、token 监督量、吞吐、显存及分层验证指标。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m latent_working_memory.v1.evaluate \
  --checkpoint artifacts/v1/experiments/pretrain-pilot/checkpoints/pretrain-step-001000.pt \
  --data-dir data/v1/fineweb-independent/semantic \
  --output-dir artifacts/v1/evaluations/pretrain-pilot-test \
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
data/v1/              完整句界／随机截断 AE/LM 数据与准备记录
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

C-DIC 的论文实现位于 `reproductions/cdic/`。当前已完成 retrieval、recency、write-back、state lineage、trace、one-hop credit assignment、ICAE adapter、MSC loader、双卡训练与 checkpoint/resume。A800 多轮 GPU smoke test、双卡 MSC pilot 和 seed 42 的两 epoch 完整训练均已完成。

[20260906 评估口径对齐](notes/reproduction_results/20260906_c_dic_evaluation_alignment_record.md) 已在相同 32 个 validation episodes 上比较初始化与训练后模型，并分列轮次范围、EOS 分母和 ROUGE recall/F1。全部轮次 PPL 从 16.9929 升至 23.0159，session 5 最后一轮从 22.5589 升至 28.2935；当前结果不支持论文效果已复现。

```bash
PYTHONPATH=reproductions/cdic/src uv run pytest -q reproductions/cdic/tests
uv sync --project reproductions/cdic --frozen
```
