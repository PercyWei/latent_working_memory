# 20261124_边界输入的待压缩长度探针

创建时间：20261124 11:23:40 Asia/Shanghai

最后修订时间：20261124 11:23:40 Asia/Shanghai

冻结 PPO checkpoint，检验边界判断器实际输入 `hidden[-1]` 能否预测当前待压缩长度。只训练轻量探针，不更新 Writer、压缩器或原边界判断器，不计算 QA。

## 实验设置

- checkpoint：`boundary-post_k256_ppo-decay0p1-gae1-cost0p03_20261122/train/checkpoints/global_step_252`。该点在已有 dev 诊断中仍有自主压缩，适合同时观察首次压缩前后的事件。
- 保留原 train 的 1,002 篇与 dev 的 118 篇文档，二者文档互斥。train 内按文档划分 80% 拟合、20% 调参；选定超参数后在全部 train 上重新拟合，最后报告 dev。
- 每篇文档采样一条轨迹，train/dev 均使用 checkpoint 配置对应的 `temperature=1.0`，`recent_size=256`、`pending_stride=256`、`pending_size=256`，随机种子 42。实际温度以 checkpoint 配置为准，不从运行名推断。
- 精确提取 `event_features[1:-1][decision_mask]`。标签为到达新单元之前的 `pending_tokens_before`；强制等待与 EOF 不作为样本。首次自主 compress 所在事件归入“首次压缩前”，之后的事件归入“首次压缩后”。
- 特征与目标仅用拟合文档计算均值和标准差。Ridge 在 `[0.1, 1, 10, 100, 1000, 10000]` 中选调参 MSE 最低值；目标为平方误差之和加 `alpha * ||w||²`，使用 float64 求解。
- MLP 为 `Linear(2560,128) → SiLU → Linear(128,1)`，AdamW，学习率 `1e-3`、weight decay `1e-4`、batch size 256，最多 100 epoch，patience 10。按最佳 epoch 数重新初始化并在全部 train 上拟合。
- 所有自主决策等权，报告 R²、MAE（tokens）、事件数、文档数和目标标准差。统一拟合的模型分别在全部、首次压缩前、首次压缩后统计；各组 R² 分母使用该组自己的标签均值。另报告训练均值与 `read_tokens` 单变量线性回归基线。

## 输入协议与产物

该历史 checkpoint 的 Writer 输入包含固定聊天提示与 `Document:` 前缀，当前训练代码已采用纯正文输入。本实验明确使用 checkpoint 原始输入协议，以其 tokenizer binding 和 train/dev 的原始 `tokenized_content` 校验还原结果，原 checkpoint 不作修改。实验结论对应这套历史输入分布。

实验目录为 `a100:/data/percyw/projects/latent_working_memory/artifacts/v3-1/pending-length-probe_20261124/`。先保存 `plan.json`，再采集特征；完整轨迹输入保存为 `documents.pt`，唯一格式如下：

```text
{checkpoint, tokenization, data_identity,
 documents: [{id, split, input_ids}]}
```

两个 GPU worker 按文档分片，分别保存 `features/shard_0.pt`、`features/shard_1.pt`，每个文档包含原 dtype 的特征矩阵、pending 长度、首次压缩后标记、已读长度和自主动作。采集过程保存 `features/progress_*.json`。探针目录 `probe/` 保存计划、训练日志、模型参数、逐位置预测和最终 `metrics.json`；大特征仅保存在服务器。

采集入口：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m latent_working_memory.v3_1_1.evaluation.pending_length_features \
  --checkpoint <checkpoint> --documents-file <experiment>/documents.pt \
  --output-dir <experiment> --batch-size 2 --shard-index 0 --num-shards 2
```

另一个 worker 使用物理 GPU 1 和 `--shard-index 1`。全部采集完成后：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m latent_working_memory.v3_1_1.evaluation.pending_length_probe \
  --feature-dir <experiment>/features --output-dir <experiment>/probe --device cuda --seed 42
```

## 结论口径

高 dev R² 支持长度信息在该 checkpoint 的边界输入中可读出；首次压缩后的结果更能检验对 pending 重置的适应。低 R² 只能说明本实验的样本分布和轻量探针未能准确读出长度，不能证明特征完全不包含长度信息。首次压缩前 pending 与已读位置近乎一致，单看该组的高分不足以证明能追踪压缩后的 pending。
