# 20260905 C-DIC 训练后多轮 GPU 测试记录（20260905 10:50:23 CST）

创建时间：20260905 10:50:23 CST（UTC+08:00）

最后修订时间：20260905 10:50:23 CST（UTC+08:00）

状态：训练后 checkpoint 加载和多轮工程链路通过；当前合成样本未证明效果提升，并暴露 fixed threshold 与路径依赖问题

## 测试目标

验证 seed 42 两 epoch 训练得到的 `final.pt` 能否严格恢复到 ICAE adapter，并在与训练前相同的五轮对话中检查 retrieval、write-back、旧信息召回和信息更新。

## 输入与方法

- 基础模型：`/data/bywei/models/meta-llama/Llama-2-7b-chat-hf`；
- ICAE initialization：`/data/bywei/projects/latent_working_memory/checkpoints/icae/v1/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt`；
- C-DIC checkpoint：`/data/bywei/projects/latent_working_memory/checkpoints/cdic/msc_paper_seed42/checkpoints/final.pt`；
- checkpoint 进度：2 epochs、1002 optimizer steps；
- GPU：物理 GPU 0；
- 解码：greedy，`max_new_tokens=16`；
- retrieval：cosine similarity、decay `0.05`；
- threshold：复现配置 `0.80`，附加控制组 `0.85`。

执行命令：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project reproductions/cdic --no-sync \
  pytest -q -s reproductions/cdic/tests/test_gpu_multiturn_smoke.py \
  --cdic-gpu-config reproductions/cdic/configs/gpu_smoke_trained_a800.json

CUDA_VISIBLE_DEVICES=0 uv run --project reproductions/cdic --no-sync \
  pytest -q -s reproductions/cdic/tests/test_gpu_multiturn_smoke.py \
  --cdic-gpu-config reproductions/cdic/configs/gpu_smoke_trained_threshold085_a800.json
```

两组测试均为 `1 passed`，运行时间分别约 100.19 秒和 100.54 秒。峰值显存约 13.17 GiB。

## Checkpoint 恢复检查

`final.pt` 包含 129 个 trainable tensors，名称与 ICAE initialization 完全对应。所有 tensors 均发生非零变化：64 个 LoRA A、64 个 LoRA B 和 1 个 compression-token embedding。该结果排除了“训练参数未加载”这一解释。

## 结果

| 条件 | 无关 dolphin turn | 旧 code 查询 | code 更新 | 当前 code 查询 |
|---|---|---|---|---|
| ICAE initialization，threshold 0.80 | score 0.695，insert | `ZETA-4827`，score 0.806，replace | 输出包含 `OMEGA-7319`，但 score 0.471，insert | 错误回答 `42010` |
| C-DIC final，threshold 0.80 | score 0.834，错误 replace | 错误回答 `I am a dog.` | 错误回答，insert | 错误回答 |
| C-DIC final，threshold 0.85 | score 0.834，insert | `Zeta-4827`，score 0.871，replace | 错误回答，score 0.578，insert | 错误回答 |

threshold `0.80` 下，训练后的表示把无关 dolphin query 判为 on-topic，并用 dolphin turn 覆盖了 code thread，导致后续旧 code 召回失败。提高到 `0.85` 后可以阻止该错误覆盖并恢复旧 code，但更新 query 与旧 code state 的得分仅为 `0.578`，更新仍被写成新 thread，最终无法回答当前 code。

在该单一样本轨迹中，拒绝无关 turn 需要 threshold 高于 `0.834`，而合并更新 turn 需要 threshold 不高于 `0.578`。因此，仅调整单一全局 threshold 无法同时满足两者，问题涉及 retrieval key 的判别结构和错误 write-back 引起的路径依赖，而不只是 threshold 数值选择。

该对话不属于 MSC 分布，不能据此判断论文结果是否复现。它只说明当前训练后模型在该合成更新任务上没有改善，且 fixed-threshold routing 存在明确反例。

## 产物

- threshold 0.80：`/data/bywei/projects/latent_working_memory/artifacts/cdic/20260905_trained_multiturn_gpu_smoke/gpu_smoke_report.json`；
- threshold 0.85：`/data/bywei/projects/latent_working_memory/artifacts/cdic/20260905_trained_multiturn_gpu_smoke_threshold085/gpu_smoke_report.json`；
- 训练前对照：`/data/bywei/projects/latent_working_memory/artifacts/cdic/20260904_multiturn_gpu_smoke/gpu_smoke_report.json`。

## 下一步

在 MSC held-out episodes 上比较 ICAE initialization 与 C-DIC final 的 teacher-forced response loss、同 episode 与跨 episode retrieval score 分布及 routing 错误率。只有先确认 in-distribution 指标，才能区分训练无效、合成样本分布外和 fixed-threshold 设计缺陷。
