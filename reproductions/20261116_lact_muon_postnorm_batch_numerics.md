# 20261116_LaCT Muon PostNorm 的 batch 数值差异

创建时间：20261116 00:35:43 Asia/Shanghai

最后修订时间：20261116 02:18:13 Asia/Shanghai

## 结论

对上轮评分变化的同一条完整输入，原有无 padding 路径在 B1 与 B3 下已经产生不同答案和得分；无实际 padding 时，同形状的旧／强制新 B1、B3 各自生成相同的 128 个 IDs；实际新混长 B3 也与旧等长 B3 一致。因此，本例没有发现 padding 的位置、状态或更新时序错误，原有 batch 数值差异已足以改变单题评分。不能将上轮的 100→0 单独归因为 padding。

这里只判断这个案例，不保证任意输入跨 batch 的 logits、生成或评分相同。本轮只新增诊断脚本和记录，没有修改模型计算、TTT 公式、更新时序、精度或生产组批策略。

## 控制条件

- 服务器 `zcy-memory-v4.2`，物理 GPU 0，L40S 48GB；LaCT 固定环境与上轮相同。
- checkpoint 为 `lact-muon-nope-postnorm-nheads2-chunk2048-760m`，实际按配置加载四个 fast-weight heads；float32 参数、BF16 autocast、greedy、128-token 预算不变。
- 焦点输入为完整 `ruler/8192/niah_single_2/62`，7,435 tokens，目标数字为 `5813613`。混合长度伙伴分别为 8,035、8,064 tokens，仍使用完整原文。
- 各模式顺序运行，每种单独启动新 Python 进程，第一次模型调用就是生成；不提前运行边界检查或预热。诊断从真实生成的前两次 forward 读取 logits，没有额外 teacher-forced 前向。
- 下表的“旧路径”是 padding 接入前已有的增量等长 batch 路径，不是官方完整前缀 `original` 模式。“强制新路径”传入空的 `PaddedLaCTCache`，即使没有实际 padding 也执行新路径。

等长 B3 使用焦点输入的三份复制，只控制 batch 形状，不用于宣称样本隔离。新混长 B3 使用上轮的三条真实输入。checkpoint 配置与上轮逐项相同；两次任务之间 LaCT 的模型、padding 和 TTT 代码没有变化。

## 完整输入结果

| 模式 | 物理输入形状 | 焦点第二个生成 token | 单题得分 |
|---|---:|---|---:|
| 旧 B1，无 padding | 1×7435 | `2` | 0 |
| 旧等长 B3，无 padding | 3×7435 | `5` | 100 |
| 强制新 B1，无 padding | 1×7435 | `2` | 0 |
| 强制新等长 B3，无 padding | 3×7435 | `5` | 100 |
| 新 B1，左 padding | 1×8064 | `2` | 0 |
| 新等长 B3，左 padding | 3×8064 | `5` | 100 |
| 新混长 B3 | 3×8064 | `5` | 100 |

所有行的焦点真实输入长度均为 7,435。关键配对：

- 旧 B1与强制新 B1：完整 128 个生成 IDs 相同，前两步 logits 也完全相同。
- 旧等长 B3与强制新等长 B3：完整 128 个 IDs 相同；logits 有数值差异，不能宣称逐元素相同。
- 旧等长 B3与实际新混长 B3：完整 128 个 IDs 相同。
- 强制新 B1与加 padding 的新 B1：完整 128 个 IDs 相同。
- 强制新等长 B3→左 padding 等长 B3，以及左 padding 等长 B3→真实混长 B3，两组均从第 33 个生成 token 开始不同，但答案和得分不变。旧等长 B3与真实混长 B3的完整 128 个 IDs 仍完全相同。

另外，在新进程中完全移除 logits hook，分别重复旧 B1、旧等长 B3和新混长 B3。得分仍为 **0、100、100**，且每组的完整 128 个 IDs 与记录 logits 时相同，排除了本次记录 hook 的干扰。

## 分歧位置

各组第一个生成 token 都是 ID `28705`，因此第二步具有相同的真实输入前缀，可以直接比较：

| 路径 | `2` 的 logit | `5` 的 logit | `5`−`2` |
|---|---:|---:|---:|
| 旧 B1 | 25.125 | 25.125 | 0 |
| 旧等长 B3 | 24.750 | 24.875 | 0.125 |
| 强制新 B1 | 25.125 | 25.125 | 0 |
| 强制新等长 B3 | 25.000 | 25.125 | 0.125 |
| 新 B1，左 padding | 25.000 | 24.875 | −0.125 |
| 新等长／混长 B3，左 padding | 25.000 | 25.125 | 0.125 |

旧 B1 的两个候选并列；greedy 的 `argmax` 选择 ID 较小的 `2`（28750），而 `5` 的 ID 为 28782。旧 B3 中 `5` 高出 0.125，排序改变，随后生成正确数字。后续[第二个 token 的逐层定位](20261116_lact_muon_postnorm_second_token_trace.md)确认，首 decode 第 0 层 QKV 与输出投影均可在相同输入、相同权重下，仅因 B1／B3 形状不同而产生差异，且独立重放与真实输出一致；尚未将最终排序变化唯一归因于其中一个元素或某个底层 CUDA 算法。

prefill 后全部路径均完成三次写入、保留 1,291 个待提交 token；喂入首个生成 token 后为 1,292，仍为三次写入。共检查前两步、24 层、七种模式的 **336 条逐层进度记录**，全部符合真实 token 时序。整个 128-token 预算内焦点都不会跨入下一次写入边界，所以本例的第二步分歧不是 decode 更新次数不同造成的。

## 与上轮记录的关系

上轮 `validate_padding.py` 先运行整套边界检查，再执行真实生成；本轮每种模式均是新进程首次生成。上轮 B1=100、新混长 B3=0，本轮 B1=0、新混长 B3=100，不能把不同调用历史中的 B1 当作固定不变的参照。

两轮的模型计算代码、checkpoint 配置、输入、精度和生成参数一致；本轮另固定 seed，并用无 hook 复测排除了当前记录的影响。上述证据支持结果对 batch 和执行条件敏感，但没有精确复放上轮所有调用，因此不将反转完全归因于 warmup 或 `torch.compile`，也不宣称编译器有错误。

当前应将这条差异归类为已有 batch 推理的数值敏感案例，而不是 padding 正确性检查失败。padding 仍可能改变浮点执行路径，后续评估应保留 B1 对照；本次单题结果不说明 B3 总体优于 B1。

## 运行与产物

新增入口 `lact/scripts/diagnose_muon_batch.py`，每次只运行一种模式：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=reproductions/lact:reproductions/lact/upstream/lact_llm:src \
  ~/rivermind-data/envs/ttt-lact/bin/python reproductions/lact/scripts/diagnose_muon_batch.py \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-muon-nope-postnorm-nheads2-chunk2048-760m \
  --ruler-data data/benchmarks/ruler-v1/reproductions/lact-760m \
  --source-plan artifacts/reproductions/padding-validation_20261115/lact-muon-postnorm/plan.json \
  --mode old_B1 \
  --output-dir artifacts/reproductions/lact-muon-postnorm-batch-diagnosis_20261116/compiled/old_B1
```

其他模式使用脚本 `--help` 所列名称，并传入 `--reference-dir` 指向同设置的旧 B1 目录；每个模式须独立进程、独立新目录。`--no-record-logits` 完全移除 hook，参照也须使用同一设置。复制输入仅用于控制实验，没有修改正式评估的数据或组批方式。

本地与服务器产物位于 `artifacts/reproductions/lact-muon-postnorm-batch-diagnosis_20261116/`：根 `summary.json` 汇总配对结果，`compiled/` 保存七模式，`without-hook/` 保存三组无 hook 复测。仅保存 JSON、日志和每模式两条小型 logits 向量，不保存整份 KV 或 fast weights。

七模式实际代码提交为 `27a3da6e`，无 hook 复测为 `412456ab`，GPU 运行时仓库均为 clean。通过 GitHub／Gitee 同步后从服务器项目仓库运行，原始实现和生产优化代码保留不变。
