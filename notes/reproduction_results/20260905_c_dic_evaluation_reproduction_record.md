# 20260905 C-DIC 评估复现记录（20260906 17:05:27 CST）

创建时间：20260905 10:50:23 CST（UTC+08:00）

最后修订时间：20260906 17:05:27 CST（UTC+08:00）

状态：训练后 checkpoint 加载与三组 pilot 已完成；当前结果未复现论文效果

20260906 补充：下述历史数值保持不变。8 × 8 held-out pilot 实际覆盖 session 1 的 59 turns 和 session 2 的 5 turns，不能作为 session 4 后期效果的证据；对齐样本、轮次与指标后的对照见 [20260906 评估口径对齐记录](20260906_c_dic_evaluation_alignment_record.md)。

## 公共设置

- backbone：Llama-2-7B-Chat；
- ICAE initialization：公开 v1 checkpoint；
- C-DIC final：seed 42、2 epochs、step 1002；
- compression tokens：128；
- retrieval：cosine similarity、decay `0.05`；
- GPU：NVIDIA A800-SXM4-80GB，物理 GPU 0。

`final.pt` 的 129 个 trainable tensors 均与 initialization 对应且发生非零变化，包括 64 个 LoRA A、64 个 LoRA B 和 1 个 compression-token embedding。训练参数已被正确加载。

## 合成多轮测试

测试序列为：写入旧 code → 无关 dolphin 话题 → 查询旧 code → 更新 code → 查询当前 code。

| 条件 | 无关 turn | 旧 code 查询 | code 更新 | 当前 code 查询 |
|---|---|---|---|---|
| initialization，threshold 0.80 | 0.695，insert | 正确，0.806，replace | 输出新 code，但 0.471，insert | 错误 |
| C-DIC final，threshold 0.80 | 0.834，错误 replace | 错误 | 错误，insert | 错误 |
| C-DIC final，threshold 0.85 | 0.834，insert | 正确，0.871，replace | 错误，0.578，insert | 错误 |

训练后 threshold `0.80` 将无关 turn 合并进 code thread，覆盖后导致后续失败。提高 threshold 可保留旧信息，但无法合并更新。在该轨迹中，拒绝无关 turn 需要 threshold 高于 `0.834`，接受更新却需要不高于 `0.578`，单一全局 threshold 无法同时满足两者。

两组训练后 pytest 均通过，耗时约 100 秒，峰值显存约 13.17 GiB。该样本不属于 MSC 分布，只用于暴露 routing 和路径依赖问题。

## MSC held-out diagnostic

使用 session 4 validation 的前 8 个 episodes，每个 episode 前 8 turns，共 64 turns、1053 response tokens；对 initialization 与 final 使用相同的 gold-response teacher forcing 路径。

| Response 指标 | initialization | final | final - initialization |
|---|---:|---:|---:|
| mean turn loss | 3.754708 | 3.732776 | -0.021932 |
| token-weighted loss | 3.658733 | 3.665768 | +0.007035 |
| token-weighted PPL | 38.8121 | 39.0861 | +0.2740 |

64 turns 中，loss 降低 30 次、升高 34 次；paired delta median 为 `+0.009651`。当前 pilot 不支持稳定的 response prediction 改善。

| Routing 指标 | initialization | final |
|---|---:|---:|
| same-episode mean score | 0.7883 | 0.8545 |
| cross-episode mean score | 0.7867 | 0.8487 |
| mean margin | 0.0016 | 0.0057 |
| pairwise accuracy | 50.0% | 75.0% |
| same-episode accept rate | 37.5% | 100.0% |
| cross-episode false accept rate | 12.5% | 100.0% |

训练使 positive 与 negative scores 同时升高；threshold `0.8` 下 8 个 cross-episode negatives 全部被接受。pairwise accuracy 仅基于 8 对样本，不足以抵消该风险。

## Table 1 MSC pilot

论文 Table 1 的 C-DIC MSC 指标为 PPL `8.431`、BLEU `0.023`、ROUGE-L `0.160`、ROUGE-1 `0.205`、ROUGE-2 `0.037`。

当前 protocol approximation 使用 session 5 test 的前 2 个 episodes：session 1 只构建 gold memory，sessions 2–5 生成并计分；共 48 turns、1703 response tokens。greedy generation 最多 128 tokens，运行约 49.74 秒，无失败记录。

| 指标 | 当前 pilot | 论文 |
|---|---:|---:|
| PPL | 18.838 | 8.431 |
| BLEU | 0.0084 | 0.023 |
| ROUGE-L F1 | 0.1545 | 0.160 |
| ROUGE-1 F1 | 0.1682 | 0.205 |
| ROUGE-2 F1 | 0.0247 | 0.037 |

部分 prediction 出现无关或重复的模板句。两个 episodes 的 final-turn PPL 为 `11.200`，说明计分范围会显著影响结果，但样本量不足以选择协议。

当前未确认项包括：evaluation split 聚合、计分 turn 范围、instruction memory、speaker/template、`[FT]` markers、generation length，以及 BLEU/ROUGE 的 tokenizer、smoothing 和 aggregation。因 PPL 与 BLEU 差距明显，全量 501-episode 运行暂缓。

## 复现命令

```bash
# 训练后合成多轮测试
CUDA_VISIBLE_DEVICES=0 uv run --project reproductions/cdic --no-sync \
  pytest -q -s reproductions/cdic/tests/test_gpu_multiturn_smoke.py \
  --cdic-gpu-config reproductions/cdic/configs/gpu_smoke_trained_a800.json

# MSC held-out diagnostic
CUDA_VISIBLE_DEVICES=0 uv run --project reproductions/cdic --no-sync \
  pytest -q -s reproductions/cdic/tests/test_gpu_msc_heldout.py \
  --cdic-msc-eval-config reproductions/cdic/configs/msc_heldout_pilot_a800.json

# Table 1 MSC pilot
CUDA_VISIBLE_DEVICES=0 uv run --project reproductions/cdic --no-sync \
  python -m cdic_repro.eval_table1_msc \
  --config reproductions/cdic/configs/table1_msc_pilot_a800.json
```

## 产物

| 测试 | 路径 |
|---|---|
| initialization 多轮对照 | `/data/bywei/projects/latent_working_memory/artifacts/cdic/20260904_multiturn_gpu_smoke/gpu_smoke_report.json` |
| final，threshold 0.80 | `/data/bywei/projects/latent_working_memory/artifacts/cdic/20260905_trained_multiturn_gpu_smoke/gpu_smoke_report.json` |
| final，threshold 0.85 | `/data/bywei/projects/latent_working_memory/artifacts/cdic/20260905_trained_multiturn_gpu_smoke_threshold085/gpu_smoke_report.json` |
| held-out diagnostic | `/data/bywei/projects/latent_working_memory/artifacts/cdic/20260905_msc_heldout_pilot/msc_heldout_report.json` |
| Table 1 pilot | `/data/bywei/projects/latent_working_memory/artifacts/cdic/20260905_table1_msc_pilot/` |

## 下一步

先用小样本核对 instruction initialization、prompt serialization 和计分 turn 范围，再扩大 validation/test。扩大后应保存原始 positive/negative scores，并报告 threshold-independent AUC、score quantiles 和 threshold sweep。
