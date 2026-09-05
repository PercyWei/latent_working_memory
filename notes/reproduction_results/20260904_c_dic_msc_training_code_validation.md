# 20260904 C-DIC MSC 训练代码验证（20260905 10:38:27 CST）

创建时间：20260904 22:00:30 CST（UTC+08:00）

最后修订时间：20260905 10:38:27 CST（UTC+08:00）

状态：训练代码、单卡 autograd/checkpoint smoke test、双卡官方 MSC pilot 和 seed 42 两 epoch 完整训练均已完成

## 验证范围

- MSC v0.1 `session_4/train.txt` 解析；
- teacher-forced response loss；
- gold-response incremental compression；
- one-hop ra-TBPTT gradient path；
- frozen generator 与可训练 LoRA/compression-token 参数边界；
- episode-level optimizer step；
- checkpoint/resume schema；
- JSON config、metrics 和 memory trace。

## 官方数据检查

归档：`msc_v0.1.tar.gz`

SHA256：`e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60`

`session_4/train.txt` 解析结果：

- episodes：1001；
- source utterances：50,374；
- mean source utterances：50.3237；
- paired turns：25,126；
- odd-length session 的无配对尾 utterance：122；
- 空 utterance：1，规范化为 `__SILENCE__`。

episode 数与论文一致；utterance 均值低于论文报告的 53.3，暂不通过隐式补样本或修改原始数据追齐。

## 真实 GPU autograd smoke test

环境：NVIDIA A800-SXM4-80GB、PyTorch 2.0.1+cu118、Llama-2-7B-Chat、ICAE v1 checkpoint。

输入：1 个合成 episode、2 turns；retrieval threshold 设为 `-1.0`，保证第二轮沿同一 thread 执行 replace 和 one-hop gradient。

最终结果：

- mean loss：2.687118；
- gradient norm：0.808781；
- LoRA：128/128 trainable tensors 获得 gradient；
- compression tokens：1/1 tensor 获得 gradient；
- final memory states：1；
- optimizer step：1；
- 单 episode runtime：2.2277 秒，不含模型加载；
- peak allocated GPU memory：15,322,347,520 bytes，约 14.27 GiB；
- 进程正常退出。

真实 checkpoint 写入与恢复测试通过：checkpoint 包含 trainable model state、AdamW state、CPU/CUDA RNG state 和训练位置；单步 checkpoint 约 775 MiB。训练配置默认只保留最近两个 step checkpoints，避免长训练无限占用磁盘。

## 双卡训练模式

采用 synchronous data parallel（同步数据并行），只训练一个 seed 42 模型：

- `torchrun` 启动两个进程，rank 0 使用 GPU 0，rank 1 使用 GPU 1；
- 每个 rank 持有相同模型副本并处理一个独立 episode；
- 每个 optimizer step 前，通过 NCCL 对 trainable gradients 求和并除以 active worker 数；
- rank 0 保存统一 checkpoint；checkpoint 同时保存两个 ranks 的 RNG state；
- metrics 和 memory trace 按 rank 分文件保存；
- 最后一批只有一个 episode 时，另一 rank 提供 zero gradients，按 active worker 数归一化。

该模式优先采用成熟的 replicated data-parallel 语义，不使用尚未验证的 layer sharding。每卡 batch size 为 1，global batch size 为 2；这与论文单卡 global batch size 1 不完全一致，必须作为复现偏差保留。

双卡官方 MSC pilot 使用两个真实 episodes、每个 episode 前八轮，结果为：

- 两个 ranks 均完成一个 optimizer step；
- 同步后的 gradient norm 均为 0.102688；
- rank 1 的 128/128 LoRA tensors 获得本地 gradient，rank 0 的 episode 全部发生 thread insert，因此本地 LoRA gradient 为 0；all-reduce 后两侧模型获得相同更新；
- GPU 0 peak allocated memory：15,364,981,760 bytes，约 14.31 GiB；
- GPU 1 peak allocated memory：15,350,328,320 bytes，约 14.30 GiB；
- 每个 rank 的八轮训练约 4.63 秒，不含模型加载；
- checkpoint 写入成功，大小约 775 MiB。

完整训练使用以下命令：

```bash
nohup uv run --project reproductions/cdic --no-sync \
  torchrun --standalone --nproc-per-node=2 \
  -m cdic_repro.train_msc \
  --config reproductions/cdic/configs/msc_paper_a800.json \
  > /data/bywei/projects/latent_working_memory/artifacts/cdic/logs/20260904_msc_paper_seed42.log 2>&1 < /dev/null &
```

训练对象只有一个 seed 42 模型，不并行运行多个 seeds。输出目录为 `/data/bywei/projects/latent_working_memory/checkpoints/cdic/msc_paper_seed42`。

## 训练结果保存位置

- 完整训练目录：`/data/bywei/projects/latent_working_memory/checkpoints/cdic/msc_paper_seed42/`；
- 最终 checkpoint：`checkpoints/final.pt`；
- 最新 checkpoint 指针：`checkpoints/latest.json`；
- 中间 checkpoint：`checkpoints/step-000950.pt`、`checkpoints/step-001000.pt`；
- 训练配置与数据摘要：`config.resolved.json`、`data_summary.json`、`trainable_parameters.json`；
- 分 rank 指标：`metrics.rank00.jsonl`、`metrics.rank01.jsonl`；
- 分 rank memory trace：`memory_trace.rank00.jsonl`、`memory_trace.rank01.jsonl`；
- 训练日志：`/data/bywei/projects/latent_working_memory/artifacts/cdic/logs/20260904_msc_paper_seed42.log`。

## 完整训练结果

- epoch：2；
- optimizer steps：1002；
- 最终进度：`epoch=2`、`next_episode_position=0`；
- 最终 checkpoint：`checkpoints/cdic/msc_paper_seed42/checkpoints/final.pt`；
- 保留的中间 checkpoint：`step-000950.pt`、`step-001000.pt`；
- 最终 checkpoint 大小：811,913,595 bytes；
- 最后一批 mean loss：3.607225；
- 最后一批 gradient norm：0.043376；
- 最后一批 GPU 0 peak allocated memory：19,357,192,192 bytes，约 18.03 GiB。

`latest.json` 已指向迁移后的项目内 `final.pt`。历史 `config.resolved.json` 保留训练时的旧绝对路径；旧路径现为 symlink，不影响 checkpoint 恢复。

## 修复记录

首次测试发现启用 gradient checkpointing 后 LoRA gradient 为 0。原因是 ICAE 的 checkpoint branch 未把 `enable_lora` 传给 `LlamaDecoderLayer`，导致 compressor forward 退化为 base weights。已在 `reproductions/icae/src/icae/base/modeling_llama_icae.py` 补充该参数，并再次验证全部 128 个 LoRA tensors 获得 gradient。

query routing 使用临时 eval mode 和 `inference_mode()`，避免无梯度 query encoding 进入 gradient-checkpoint branch；gold-response compression 保持 train mode。

## 下一步

使用 `final.pt` 运行训练后多轮 GPU 评估，并与 ICAE initialization、论文设置及必要 baseline 对照。训练完成本身不等于论文效果复现。
