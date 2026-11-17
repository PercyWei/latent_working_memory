# 20261108_LaCT 第一阶段复现

创建时间：20261108 17:03:39 Asia/Shanghai

最后修订时间：20261117 15:35:23 Asia/Shanghai

本目录使用 [Test-Time Training Done Right](https://arxiv.org/abs/2505.23884v1) 的官方语言模型实现和作者公开 checkpoint。Momentum 760M 已于 20261110 完成下表全部 12,000 条 RULER 样本的评估，结果查看方式见[查看结果](#查看结果)。

源码与本地优化边界见 [UPSTREAM.md](UPSTREAM.md)，验证结论见[复现状态](../20261110_ttt_reproduction_status.md)。

## 评估目标与数据

首批评估作者公开重训的 **760M Momentum／RoPE／PreNorm**，两个 Muon 变体按需补充。这些模型按论文训练设置重训，其权重不是论文表格对应的同一次训练产物。

三个公开模型的 tokenizer 和特殊 token 配置一致，可共用下表所列的 **12,000 条评估样本**。下载和构造命令见[评估数据准备](../20261108_ttt_data_preparation.md)。数据固定从下表路径读取。

| benchmark | 任务与长度 | 样本数 | 固定数据路径 |
|---|---|---:|---|
| RULER | S-NIAH-1/2/3、MK-NIAH-1、MQ-NIAH、MV-NIAH；4K／8K／16K／32K；每项任务、每个长度 500 条 | 12,000 | `data/benchmarks/ruler-v1/reproductions/lact-760m` |

论文 [Table 8](https://arxiv.org/html/2505.23884v1#A3.T8) 和 [Table 9](https://arxiv.org/html/2505.23884v1#A3.T9) 中 **760M 参数规模模型**的平均准确率如下，单位为 %。原论文默认使用 4 个 fast-weight heads，当前公开 Momentum 的配置与权重均为两头，并且其是作者重训的 PreNorm 模型，并非原表格对应的同一次训练产物。

| 指标 | 更新方式 | 4K | 8K | 16K | 32K |
|---|---|---:|---:|---:|---:|
| S-NIAH-1/2/3 平均准确率 | Momentum（两头） | 70.9 | 35.2 | 14.9 | 7.7 |
| S-NIAH-1/2/3 平均准确率 | Momentum（四头） | 89.9 | 64.3 | 40.9 | 33.1 |
| S-NIAH-1/2/3 平均准确率 | Muon（四头） | 77.9 | 60.5 | 40.0 | 35.1 |
| MK-NIAH-1／MQ-NIAH／MV-NIAH 平均准确率 | Momentum（两头） | 40.3 | 25.1 | 14.4 | 7.7 |
| MK-NIAH-1／MQ-NIAH／MV-NIAH 平均准确率 | Momentum（四头） | 41.7 | 29.7 | 18.5 | 9.1 |
| MK-NIAH-1／MQ-NIAH／MV-NIAH 平均准确率 | Muon（四头） | 42.9 | 32.1 | 17.1 | 7.5 |

## 模型下载

作者公开权重见 [YunjinZhang/lact_llm](https://huggingface.co/YunjinZhang/lact_llm/tree/main)，`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
# Momentum／RoPE／PreNorm
hf download YunjinZhang/lact_llm \
  --include 'lact-momentum-rope-prenorm-nheads4-chunk2048-760m/*' \
  --local-dir ~/rivermind-data/models/reproductions/lact

# 可选：Muon／NoPE 的 PreNorm 和 PostNorm
hf download YunjinZhang/lact_llm \
  --include 'lact-muon-*/*' \
  --local-dir ~/rivermind-data/models/reproductions/lact
```

需要注意的是，当前目录名和实际配置存在差异，加载时以 checkpoint 自带配置为准，差异见[LaCT 公开 checkpoint 权重说明](UPSTREAM.md#公开语言模型权重)。

## 环境与运行命令

从服务器项目根目录安装并激活独立评估环境，版本见[评估环境](../20261109_environment_setup.md)：

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh lact
source ~/rivermind-data/envs/ttt-lact/bin/activate
```

Momentum 和 Muon 均支持 `optimized` 运行，验证范围与批量计算的数值差异见[复现状态记录](../20261110_ttt_reproduction_status.md)：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/lact/scripts/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --output-id lact-momentum-rope-prenorm-nheads4-chunk2048-760m_20261116 \
  --inference-mode optimized \
  --batch-size 16 \
  --batch-size-by-length 4096=64 8192=32 16384=32 32768=16
```

| 参数 | 默认值与约定 |
|---|---|
| `--resume` | 沿用原评估协议补齐未完成样本的评估，默认不开启 |
| `--checkpoint` | 必需，本地完整 checkpoint 目录 |
| `--output-root` | 产物父目录，默认 `artifacts/reproductions/lact` |
| `--output-id` | 默认 `<checkpoint_name>_<YYYYMMDD-HHMM>` |
| `--inference-mode` | 推理模式，支持 `original`（默认）和 `optimized` |
| `--batch-size` | 每卡样本数上限，默认 `1`；只有 `optimized` 支持大于 1 |
| `--batch-size-by-length` | 按评估长度覆盖每卡上限，未指定的长度使用 `--batch-size` |
| `--lengths` | 默认 `4096 8192 16384 32768` |

- 评估产物保存在 `<output-root>/eval/<output-id>/ruler/`；新运行在已有同名目录时拒绝覆盖；原评估中断后，指定原 `--output-id` 并追加 `--resume` 可续评估，checkpoint、长度及推理模式保持原计划不变，支持调整 GPU 数量、顺序及 batch 设置。
- 输入为 `input + answer_prefix`，greedy 生成最多 128 tokens，保留 EOS；输入加生成预算不得超过所选长度，不截断。评分沿用原 RULER。
- 每批左 padding 到该批最长输入，最后一批保留实际样本数。`input_tokens` 记录各样本的真实输入长度，生成从补齐后的批次宽度之后截取。padding 不推进真实位置或 TTT 更新；TTT 保持每条样本状态独立，按真实 chunk 进度批量读出和提交。验证范围和 Muon 数值限制见[复现状态](../20261110_ttt_reproduction_status.md)。

## 查看结果

```bash
uv run reproductions/lact/evaluation/compare_results.py \
  --eval-dir artifacts/reproductions/lact/eval/lact-momentum-rope-prenorm-nheads4-chunk2048-760m_20261116
```

`--eval-dir` 指向 `<output-root>/eval/<output-id>`，报告保存为 `<eval-dir>/comparison.html`。

Momentum 和两个 Muon 均自动匹配论文参照；需完成六任务、四长度 RULER 评估。

## 来源记录与输出

以下评估记录位于 `<output-root>/eval/<output-id>/ruler/`，对比报告 `comparison.html` 位于父目录 `<output-id>/`。`scripts/eval.sh` 调用 `evaluation/evaluate_parallel.py` 统一调度，[进度展示](../README.md#进度展示)采用三方法的共同规则；续评复用历史逐样本预测，按当前 GPU 数量重新分配未完成的样本。

| 文件 | 内容 |
|---|---|
| `summary.json` | 所选样本全部完成后的任务／长度分数 |
| `evaluation_plan.json` | 固定 checkpoint、数据及生成／评分协议 |
| `launcher_execution.json` | 各次启动状态与执行参数 |
| `attempts/attempt-<编号>/shards/shard-<i>/progress.json` | 当前 worker 的独立进度快照，仅用于展示 |
| `attempts/attempt-<编号>/shards/shard-<i>/predictions.jsonl` | 本轮各卡实际完成的逐样本预测与评分 |
| `attempts/attempt-<编号>/merged/predictions.jsonl` | 本轮完成后按样本顺序汇总的全部预测 |
| `comparison.html` | 执行对比脚本后在父目录生成的图表与表格 |

最终分数读取 RULER 目录内的 `summary.json`，其中 `merged_summary_path` 指向本轮完整汇总。

## 备用训练入口

本轮使用公开权重进行模型评估。备用训练入口为 [train_760m_muon.sh](scripts/train_760m_muon.sh)，训练环境与数据按[官方说明](source/lact/README.md)准备；该入口尚未运行验证，设置及源码差异见 [UPSTREAM.md](UPSTREAM.md)。
