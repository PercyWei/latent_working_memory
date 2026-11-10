# 20261110_E²-TTT 推理分项计时与缓存存储诊断

创建时间：20261110 10:59:11 Asia/Shanghai

最后修订时间：20261110 10:59:11 Asia/Shanghai

## 目标与范围

测量官方 MLP／SwiGLU 1.3B checkpoint 的 prefill、普通单 token decode 和跨 TTT chunk 更新的耗时，检查逻辑尾段缓存是否继续持有整个前缀的底层存储，确定显存与速度优化的优先级。本轮只添加独立诊断入口，不修改官方模型、依赖安装或正式 `eval.sh`，不实施缓存压紧、编译或融合优化。

使用 `zcy-memory` 的单张 A800、物理 GPU 0，显式设置 `CUDA_VISIBLE_DEVICES=0`；两个 checkpoint 顺序执行，沿用 `~/rivermind-data/envs/ttt-e2_ttt/`。代码通过 GitHub 同步后从服务器项目仓库运行；执行前保存计划、Git commit／未提交状态、checkpoint 配置及实际依赖来源。

## 测量设置

- 真实输入：RULER 的 `niah_single_1`、`niah_single_2`，长度 512／4K／16K，各取索引 1 的样本（第 2 条）。沿用原 HFLM 编码、`input + answer_prefix`、greedy、128-token 生成上限、EOS 与停止条件，逐条创建独立 cache。
- 边界输入：从真实输入的 token 序列截取 511／512 tokens，各生成最多 4 tokens，只诊断 chunk 边界，不作为 benchmark 分数。
- 每个模型先预热。分别执行无 hook 的原始生成及带诊断的同输入生成，比较完整输入／生成 IDs；速度参照使用无 hook wall，带 hook 的分项时间单独记录。
- 每次 forward 记录输入长度与 CUDA event；按调用前实际 TTT 缓冲长度区分 prefill、普通 decode 和填满 512-token chunk 的 decode。prefill 产生第一个输出 token，生成 128 tokens 通常对应 127 次 decode forward。
- CUDA event 的 device span 包含 CPU 未及时提交后续算子时的等待，不解释为纯 kernel 活动时间；分项记录用于阶段归因，不冒充正式 benchmark 实测。
- prefill 结束和首个 decode 结束记录各层 attention KV、TTT 缓冲、fast weights 与 momentum 的 shape／stride／逻辑字节／底层存储字节。按底层 storage 去重，只保存元数据，不保留额外 tensor 引用；底层存储与逻辑大小的差额不能直接视为可释放的峰值显存。

## 入口与产物

独立入口为 `e2_ttt/scripts/profile_inference.py`。产物放在服务器项目的 `artifacts/reproductions/e2_ttt/inference-profiling_20261110/{mlp,swiglu}/`，保存计划、逐输入生成与一致性、阶段计时、缓存元数据和执行状态；不保存大张量或完整 profiler trace，不计入正式评估。

本轮尚未执行 GPU 诊断。结果与运行命令在执行后补齐。

在服务器仓库根目录运行：

```bash
source ~/rivermind-data/envs/ttt-e2_ttt/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/reproductions/e2_ttt/upstream:$PWD/reproductions/e2_ttt/upstream/lm-evaluation-harness"

for variant in mlp swiglu; do
  python -u reproductions/e2_ttt/scripts/profile_inference.py \
    --checkpoint "$HOME/rivermind-data/models/reproductions/e2_ttt/${variant}-1.3b-15b" \
    --ruler-data data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b \
    --output-dir "artifacts/reproductions/e2_ttt/inference-profiling_20261110/$variant"
done
```
