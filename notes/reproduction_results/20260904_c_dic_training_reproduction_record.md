# 20260904 C-DIC 训练复现记录（20260905 11:38:24 CST）

创建时间：20260904 21:08:40 CST（UTC+08:00）

最后修订时间：20260905 11:38:24 CST（UTC+08:00）

状态：工程链路、单卡 autograd、双卡 pilot 和 seed 42 两 epoch 训练均已完成

## 环境与范围

- GPU：NVIDIA A800-SXM4-80GB，使用物理 GPU 0 和 1；
- Python：3.10；PyTorch：2.0.1+cu118；
- backbone：Llama-2-7B-Chat；
- initialization：ICAE v1，128 compression tokens；
- 训练目标：teacher-forced response loss、gold-response incremental compression 和 one-hop ra-TBPTT；
- 可训练参数：compressor LoRA 与 compression-token embeddings，generator frozen。

## MSC 数据

训练集为 `msc_v0.1` 的 `session_4/train.txt`：

| 项目 | 数值 |
|---|---:|
| episodes | 1001 |
| paired turns | 25,126 |
| source utterances | 50,374 |
| mean source utterances | 50.3237 |
| dropped unpaired utterances | 122 |
| empty utterances replaced by `__SILENCE__` | 1 |

episode 数与论文一致；utterance 均值低于论文报告的 53.3，未通过修改原始数据追齐。

## 工程验证

ICAE initialization 的五轮 GPU smoke test 验证了 strict load、`retrieve → generate → compress → write-back`、state lineage 和 trace。每轮 latent 为 `[128, 4096]`、bfloat16 且数值有限，峰值显存约 13.17 GiB。该测试只证明工程链路可运行，不代表语义效果复现。

单 episode、2 turns 的 autograd test 结果：

- mean loss：2.687118；gradient norm：0.808781；
- 128/128 LoRA tensors 和 compression-token tensor 均获得 gradient；
- checkpoint 写入与恢复通过；
- 峰值显存约 14.27 GiB。

双卡 pilot 使用 2 个真实 episodes、每个 episode 8 turns：两个 ranks 均完成 1 个同步 optimizer step，gradient norm 均为 0.102688，峰值显存约 14.3 GiB，checkpoint 写入成功。

## 完整训练

使用 synchronous data parallel：每个 rank 持有完整模型并处理 1 个 episode，NCCL 平均 trainable gradients，rank 0 保存 checkpoint。每卡 batch size 为 1，global batch size 为 2；这不同于论文报告的单卡 batch size 1。

```bash
nohup uv run --project reproductions/cdic --no-sync \
  torchrun --standalone --nproc-per-node=2 \
  -m cdic_repro.train_msc \
  --config reproductions/cdic/configs/msc_paper_a800.json \
  > /data/bywei/projects/latent_working_memory/artifacts/cdic/logs/20260904_msc_paper_seed42.log 2>&1 < /dev/null &
```

训练结果：

| 项目 | 数值 |
|---|---:|
| seed | 42 |
| epochs | 2 |
| optimizer steps | 1002 |
| 最后一批 mean loss | 3.607225 |
| 最后一批 gradient norm | 0.043376 |
| 最后一批 GPU 0 peak memory | 18.03 GiB |
| final checkpoint | 811,913,595 bytes |

## 产物位置

- 训练目录：`/data/bywei/projects/latent_working_memory/checkpoints/cdic/msc_paper_seed42/`；
- 最终 checkpoint：`checkpoints/final.pt`；
- 中间 checkpoint：`checkpoints/step-000950.pt`、`checkpoints/step-001000.pt`；
- 最新指针：`checkpoints/latest.json`；
- 配置与摘要：`config.resolved.json`、`data_summary.json`、`trainable_parameters.json`；
- 指标与 trace：`metrics.rank00.jsonl`、`metrics.rank01.jsonl`、`memory_trace.rank00.jsonl`、`memory_trace.rank01.jsonl`；
- 日志：`/data/bywei/projects/latent_working_memory/artifacts/cdic/logs/20260904_msc_paper_seed42.log`。

历史 `config.resolved.json` 保留训练时的旧绝对路径；旧路径已改为 symlink，不影响 checkpoint 恢复。

## 关键修复与边界

ICAE gradient-checkpoint branch 原先未向 `LlamaDecoderLayer` 转发 `enable_lora`，导致 compressor LoRA 无 gradient。修复后 128 个 LoRA tensors 均通过 autograd 检查。query routing 使用临时 eval mode 和 `inference_mode()`，gold-response compression 保持 train mode。

训练完成只证明当前 paper-based implementation 可运行。训练后的效果与论文指标见 [20260905_c_dic_evaluation_reproduction_record.md](20260905_c_dic_evaluation_reproduction_record.md)。
