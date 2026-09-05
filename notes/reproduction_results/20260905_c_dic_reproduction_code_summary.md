# 20260905 C-DIC 复现代码功能概览（20260905 16:37:07 CST）

创建时间：20260905 16:37:07 CST（UTC+08:00）

最后修订时间：20260905 16:37:07 CST（UTC+08:00）

状态：核心机制、MSC 训练、checkpoint 恢复和初步评估均已实现；论文指标尚未复现

本实现位于 `reproductions/cdic/`。由于作者尚未公开代码，它是依据论文公式和 Algorithm 1 编写的 paper-based reimplementation，未公开的实现选择记录在 `ASSUMPTIONS.md`。

## 核心机制

| 功能 | 代码 | 说明 |
|---|---|---|
| Memory state | `memory_state.py` | 管理 `thread_id`、revision、parent state 和 recency |
| Retrieval | `retrieval.py` | cosine similarity、exponential recency decay、threshold retrieval 和 top-1 fallback |
| Write-back | `writeback.py` | 根据 retrieval 结果执行 initialize、insert 或 replace |
| Credit assignment | `credit.py` | 构造 one-hop retrieval-aware TBPTT credit path |
| Inference loop | `engine.py` | 执行 `retrieve → generate → compress → write-back` |
| Trace | `trace.py` | 保存每轮 retrieval、credit 和 memory transition |

## ICAE 接入

`icae_adapter.py` 将 ICAE v1 接入 C-DIC，支持：

- 从公开 ICAE checkpoint strict load Llama-2-7B-Chat compressor；
- query encoding、latent retrieval key pooling、response generation；
- generated-response 与 gold-response compression；
- teacher-forced response loss；
- LoRA 和 compression-token 参数训练及训练后 state 恢复。

`checkpoint.py` 用于检查 ICAE checkpoint schema 和推断 LoRA rank；`training_checkpoint.py` 保存或恢复 trainable model state、optimizer、训练进度和各 rank RNG state。

## MSC 训练

- `msc.py`：解析官方 MSC `session_4/train.txt`，构造 episode 和 `(query, response)` turns；
- `training.py`：执行 teacher-forced loss、gold-response write-back 和 one-hop ra-TBPTT；
- `distributed.py`：实现双 GPU synchronous data parallel gradient averaging；
- `train_msc.py`：提供完整训练、checkpoint retention、resume 和指标写入入口；
- `training_config.py`：解析 JSON 配置并生成配置 fingerprint。

训练配置包括 `msc_pilot_a800.json` 和 `msc_paper_a800.json`。当前 seed 42 模型已完成 2 epochs、1002 optimizer steps。

## 评估

- `evaluation.py`：比较 ICAE initialization 与 C-DIC final 的 held-out response loss 和 routing score；
- `eval_table1_msc.py`：按 gold-history teacher forcing 生成 MSC predictions，支持 episode sharding 和按 turn 续跑；
- `generation_metrics.py`：计算 token-weighted PPL、BLEU-4 和 ROUGE-L/1/2；
- `test_gpu_multiturn_smoke.py`：验证训练前后多轮 retrieval、更新和 recall；
- `test_gpu_msc_heldout.py`：运行 MSC held-out GPU diagnostic。

## 入口与记录

主要命令：

```bash
uv run --project reproductions/cdic --no-sync pytest -q reproductions/cdic/tests
uv run --project reproductions/cdic --no-sync cdic-train-msc --config <config.json>
uv run --project reproductions/cdic --no-sync python -m cdic_repro.eval_table1_msc --config <config.json>
```

训练过程与产物见 [20260904_c_dic_training_reproduction_record.md](20260904_c_dic_training_reproduction_record.md)，评估结果与当前偏差见 [20260905_c_dic_evaluation_reproduction_record.md](20260905_c_dic_evaluation_reproduction_record.md)。
