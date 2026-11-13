# 20261108_LaCT 第一阶段复现

创建时间：20261108 17:03:39 Asia/Shanghai

最后修订时间：20261113 18:44:22 Asia/Shanghai

本目录使用 [Test-Time Training Done Right](https://arxiv.org/abs/2505.23884v1) 的官方语言模型实现和作者公开 checkpoint。Momentum 760M 已于 20261110 完成下表全部 12,000 条 RULER 样本的评估，结果查看方式见[查看结果](#查看结果)。

源码与本地优化边界见 [UPSTREAM.md](UPSTREAM.md)，验证结论见[复现状态](../20261110_ttt_reproduction_status.md)，batch 设置与耗时依据见[运行参数测试](../20261110_ttt_evaluation_tuning.md)。

## 评估目标与数据

首批评估作者公开重训的 **760M Momentum／RoPE／PreNorm**，两个 Muon 变体按需补充。这些模型按论文训练设置重训，其权重不是论文表格对应的同一次训练产物。

三个公开模型的 tokenizer 和特殊 token 配置一致，可共用下表所列的 **12,000 条评估样本**。下载和构造命令见[评估数据准备](../20261108_ttt_data_preparation.md)。

| benchmark | 任务与长度 | 样本数 | 默认数据目录 |
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

Momentum 使用 `incremental`，每卡 batch 上限 16（经测试，16 是支持最长评估任务正常运行的最大 batch 设置）：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/lact/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --output-dir artifacts/reproductions/lact/eval/momentum-rope-prenorm-nheads4-chunk2048-760m_ruler \
  --inference-mode incremental \
  --batch-size 16
```

| 参数 | 默认值与约定 |
|---|---|
| `--ruler-data` | `data/benchmarks/ruler-v1/reproductions/lact-760m` |
| `--lengths` | `4096 8192 16384 32768` |
| `--inference-mode` | `original`（默认）或 `incremental` |
| `--batch-size` | 每卡上限，默认 `1`；大于 1 仅用于 `incremental`，按实际 token 数分组 |
| `--resume` | 同一设置与已有输出目录续评 |

首次运行的输出目录须不存在；中断后以相同 GPU 顺序、输出目录与参数追加 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash reproductions/lact/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --output-dir artifacts/reproductions/lact/eval/momentum-rope-prenorm-nheads4-chunk2048-760m_ruler \
  --inference-mode incremental \
  --batch-size 16 \
  --resume
```

Muon 支持 `incremental` 运行但尚未完成其适用性验证，详见[复现状态](../20261110_ttt_reproduction_status.md)。

输入为 `input + answer_prefix`，greedy 生成最多 128 tokens，保留 EOS；输入加生成预算不得超过所选长度，不截断。评分沿用原 RULER。

## 查看结果

```bash
uv run reproductions/lact/scripts/compare_results.py \
  --eval-dir artifacts/reproductions/lact/eval/momentum-rope-prenorm-nheads4-chunk2048-760m_ruler
```

`--eval-dir` 指向完整评估根目录，输出同目录的 `comparison.html`。

## 来源记录与输出

| 文件 | 内容 |
|---|---|
| `summary.json` | 所选样本全部完成后的任务／长度分数 |
| `predictions.jsonl` | 逐样本预测与评分 |
| `plan.json`、`execution.json` | 运行配置、代码来源、状态与耗时 |
| `comparison.html` | 执行对比脚本生成的图表与表格 |

最终分数读取输出根目录的 `summary.json`；逐卡预测位于 `shards/shard-<i>/`。增量与批量验证入口见 [validate_incremental.py](scripts/validate_incremental.py) 和 [validate_batch.py](scripts/validate_batch.py)。

## 备用训练入口

本轮使用公开权重进行模型评估。备用训练入口为 [train_760m_muon.sh](scripts/train_760m_muon.sh)，训练环境与数据按[官方说明](upstream/README.md)准备；该入口尚未运行验证，设置及源码差异见 [UPSTREAM.md](UPSTREAM.md)。
