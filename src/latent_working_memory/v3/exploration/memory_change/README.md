# 20261012_memory_change：冻结写入器诊断与阈值校准

创建时间：20261012 15:01:54 Asia/Shanghai

最后修订时间：20261012 17:06:19 Asia/Shanghai

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
  --max-trajectories 24 --output-dir "$MC_OUT/probe"
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

## 4. 用 EM/F1 确认入选阈值

把 `MC_THRESHOLD` 换成上一步选中的实际数值，再跑完整 dev 的生成评估；同时仍会评估两个固定动作基线。

```bash
MC_THRESHOLD="替换为选中的阈值"

CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m "$MC_MODULE.sweep" \
  --checkpoint "$MC_CKPT" --dataset-dir "$MC_DATA" \
  --thresholds "$MC_THRESHOLD" --generate \
  --output-dir "$MC_OUT/confirm"
```

查看 `confirm/comparison.csv` 的 old/new EM、F1 和容量。随后依据结果选择下一阶段：

| 结果 | 下一步 |
|---|---|
| 较高阈值节省容量，NLL 和 EM/F1 基本保持 | 从同一 warmup checkpoint 开始 policy，先只改变阈值做对照 |
| I 与损失差缺乏关系，扫描只是在损失质量和增加容量之间切换 | 修改评分信号；不继续靠阈值强行控制动作比例 |
| 多数更新点 overwrite 明显损害旧题，较高阈值也无法保持质量 | 优先研究 overwrite 训练，例如给未选中的覆盖候选加入 QA 监督 |
| warmup 本身已明显恶化 | 先检查 warmup 训练，再讨论 policy 门控 |

当前代码完成冻结诊断与校准，不包含尚未选定的辅助训练目标。阈值在 dev 上选定，后续训练方案确定后再做最终 test。

输出目录需为新目录。若更换 checkpoint，重新运行 probe 和 sweep。`--max-trajectories` 可用于缩小诊断集；它按训练 seed 固定选完整轨迹，不复制数据。
