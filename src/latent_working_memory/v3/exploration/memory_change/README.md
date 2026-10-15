# 20261012_memory_change：冻结写入器诊断与阈值校准

创建时间：20261012 15:01:54 Asia/Shanghai

最后修订时间：20261015 20:09:53 Asia/Shanghai

本阶段回答两个问题：原阈值是否与 `I` 的实际尺度不匹配，以及 `I` 能否反映覆盖相对追加造成的 QA 损失。先固定 warmup checkpoint，完成诊断和完整轨迹扫描，再决定修改阈值还是训练目标。

`inspect.py` 读取已有日志；`probe.py` 做同状态反事实比较；`sweep.py` 比较阈值策略与 always-append、always-overwrite。GPU 入口只使用 dev，不修改 checkpoint；结果保存在本地探索目录。

## 1. 检查原实验，选择 warmup checkpoint

将代码通过 GitHub 同步到服务器后，在服务器项目根目录执行。`MC_RUN` 填已有 memory_change 方法目录，目录下应有 `warmup/` 和 `policy/`。

```bash
MC_RUN="/替换为原实验的memory_change方法目录"
MC_DATA="$PWD/data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008"
MC_OUT="$PWD/artifacts/v3/memory-change-calibration_20261012-01"
MC_MODULE=latent_working_memory.v3.exploration.memory_change

uv run --frozen python -m "$MC_MODULE.inspect" \
  --run-dir "$MC_RUN" --output-dir "$MC_OUT/inspect"
```

观察 warmup 的 dev loss、old/new NLL 是否改善，有无持续恶化；对照 grad_norm 峰值发生的位置。grad_norm 是裁剪前数值，超过裁剪值本身不表示失败。工具列出仍保留的 warmup checkpoint 及同一步 dev 值：优先选 dev 正常的 warmup 末期 checkpoint，填入下面的 `MC_CKPT`。

## 2. 冻结写入器，诊断分数和损失

```bash
MC_CKPT="/替换为上一步选出的warmup/checkpoints/global_step_N"

CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m "$MC_MODULE.probe" \
  --checkpoint "$MC_CKPT" --dataset-dir "$MC_DATA" \
  --max-dev-samples 24 --output-dir "$MC_OUT/probe"
```

默认以固定 seed、50% append 概率推进轨迹。在每个更新点，从同一旧状态构造两个候选，读取已出现的 evaluation 题；旧题还读取更新前状态。gate 题和未来题不参与质量诊断。

查看终端摘要及 `probe/summary.json`，重点是 `by_tail_slots`：

- `I.p25/p50/p75`、`fraction_I_ge_original_threshold`：原阈值是否远低于真实分布；分别看 512-slot 和 32-slot 末块。
- `delta_old/new/all`：`Δ = Lrw − Lapp`，正值表示 append 的当前 QA NLL 更低；`all` 按旧、新题的实际题数合并。
- `damage_old`：旧题 `Lrw − L0`，正值表示覆盖损害了历史读取。
- `spearman_I_delta` 与 `I_bins`：较大的 `I` 是否对应更大的损失差。结合分层样本数、差值量级一起看，不能仅凭相关系数下结论。

逐更新点、逐题记录在 `updates.jsonl`；`thresholds.json` 包含原阈值及总体、各末块大小的 P25/P50/P75。它们只是候选值，尚未证明有效。

## 3. 在完整 dev 上扫描阈值

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m "$MC_MODULE.sweep" \
  --checkpoint "$MC_CKPT" --dataset-dir "$MC_DATA" \
  --thresholds-file "$MC_OUT/probe/thresholds.json" \
  --output-dir "$MC_OUT/sweep"
```

默认只算 NLL。每个阈值从头独立构造完整记忆，并自动加入 always-append / always-overwrite；不是在 probe 的随机轨迹上重新标记动作。

打开 `sweep/comparison.csv`，比较 `append_ratio`、`final_slots`、`all_nll`、`old_nll`、`new_nll` 和写入成本。目标是以较少容量保持 QA 质量，不是把 append_ratio 调成某个中间值。固定动作基线只执行实际需要的写入，所以写入调用数可能少于阈值策略。

## 4. 多 epoch 训练的验证与保存

本轮以 NLL 为质量指标，采用 warmup 3 epochs、policy 3 epochs、阈值 0.4，其他训练参数沿用原设置。从原动态预训练 checkpoint 重新开始 warmup，不恢复旧的一轮 warmup 或 policy。

训练中的 dev 验证固定使用采样 epoch 0，跨 epoch 保持相同的随机动作和分段；训练仍按当前 epoch 采样。固定的是评估条件，NLL 仍随模型权重变化。

代码通过 GitHub 同步到服务器后，在项目根目录执行。复用原动态预训练 checkpoint，模型、数据、batch 和 writer 等沿用预设与入口默认值：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3/scripts/run.sh \
  --train-only --method memory_change \
  --run-id memory-change-3ep-threshold04_20261013-01 \
  --checkpoint artifacts/v3/capacity-comparison_20261010-01/train/dynamic-pretrain-k512/pretrain/checkpoints/global_step_800 \
  --epochs 3 --threshold-i 0.4 \
  --save-every after_each_epoch --save-total-limit 6
```

`--train-only` 保留训练中的 dev NLL 验证，跳过最终生成评估。`--epochs 3` 同时应用于 warmup 和 policy，阈值 0.4 只影响 policy 的动作选择；warmup 仍以 50% 概率追加。

`after_each_epoch` 按实际训练 dataloader 长度确定保存间隔，无需预先计算步数。上例将总保留数设为 6，使 warmup 与 policy 各 3 epochs 的 checkpoint 均可用于后续比较；外部共享预训练不计入此限额。当前训练样本为 1006、全局 batch 为 16 时，两个阶段分别保留 `global_step_63`、`global_step_126`、`global_step_189`。样本数或 batch 改变时，步数相应变化。提前停止时保存的 epoch 中间 checkpoint 同样计入总数；若总保留数设为 3，policy 完成后 warmup checkpoint 会被淘汰。

在 SwanLab 观察 `dev/qa_loss`、`dev/qa_old_nll`、`dev/qa_new_nll` 是否随 epoch 改善，以及 train 持续改善而 dev 恶化的情况。warmup 的 `append_ratio` 应约为 0.5，固定 dev 的值跨 epoch 相同是预期；policy 重点结合 `append_ratio`、`slots_final` 与 NLL，判断是否重新趋向全追加，容量增长是否带来质量改善。`gate_I` 当前只记录在各阶段 `metrics.jsonl`，使用 `train/gate_I`、`dev/gate_I` 查看。

在 warmup 和 policy 结束后，分别将 `MC_CKPT` 指向该阶段 checkpoint，再用同一阈值计算最终记忆的 NLL：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m "$MC_MODULE.sweep" \
  --checkpoint "$MC_CKPT" --dataset-dir "$MC_DATA" \
  --thresholds 0.4 --output-dir "$MC_OUT/warmup-end"
```

policy 结束后替换 checkpoint，并将输出目录改为 `$MC_OUT/policy-end`。比较两次评估的 old/new NLL、append_ratio 和容量：

| 结果 | 下一步 |
|---|---|
| policy 训练后 NLL 降低，容量仍较小 | 继续验证当前阈值和训练预算 |
| policy 训练后 append_ratio 再次接近 1 | 检查训练后的 I 分布，判断阈值尺度是否漂移 |
| I 与损失差缺乏关系，扫描只是在损失质量和增加容量之间切换 | 修改评分信号；不继续靠阈值强行控制动作比例 |
| 多数更新点 overwrite 明显损害旧题，较高阈值也无法保持质量 | 优先研究 overwrite 训练，例如给未选中的覆盖候选加入 QA 监督 |
| warmup 本身已明显恶化 | 先检查 warmup 训练，再讨论 policy 门控 |

当前探索代码提供冻结诊断与校准，训练使用 v3 的现有入口；辅助训练目标根据诊断结果再决定。阈值在 dev 上选定，后续训练方案确定后再做最终 test。

输出目录需为新目录。若更换 checkpoint，重新运行 probe 和 sweep。诊断默认使用完整 dev，`--max-dev-samples` 接受正整数或 `null`，后者表示不限；按训练 seed 固定选择完整轨迹，不复制数据。
