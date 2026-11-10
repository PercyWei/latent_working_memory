# 20261110_TTT 正式评估运行参数测试

创建时间：20261110 11:51:20 Asia/Shanghai

最后修订时间：20261110 11:51:20 Asia/Shanghai

## 目标与约束

在 `zcy-memory` 的单张 A800-80GB、物理 GPU 0 上，测量正式评估的显存与吞吐，选择运行参数并估算完整评估耗时。只调整运行调度，不改变 checkpoint、生成预算、停止条件、任务或数据规模，不实施新的模型优化。

当前三种 TTT 路径的可用单进程 batch 均为 1：LaCT 增量 attention 明确限定单样本；E²-TTT 的多样本 mask、decode 归一化和动态裁剪不能直接保持单条语义；In-Place TTT 官方 `generate` 与 fast weights 也限定单样本。本轮区分每进程 batch 与同卡独立进程数，测试后者，不能将并发数称为 batch size。

## 测量方法

- 首批 checkpoint：LaCT Momentum 增量路径；E²-TTT MLP／SwiGLU 官方路径；In-Place TTT 的 TTT／baseline 原始完整 attention。
- 每个 worker 独立加载模型，逐条使用 fresh cache；CPU 线程设为 1，显式 `CUDA_VISIBLE_DEVICES=0`。不使用线程共享模型，不修改上游文件。
- 同一比较使用相同真实输入，单进程作为生成 IDs 参考。各 worker 完整预热后，在统一屏障后开始测量；模型载入和预热单独记录。
- 首轮用长输入比较 1／2／4 个进程，必要时扩展候选；随后覆盖全部正式长度及 E²-TTT 两个 recall benchmark。256K 根据已测显存只测试单进程。
- 通过整卡 `nvidia-smi` 采样记录并发总占用，另记录每进程 allocated／reserved 峰值。OOM 或生成不一致的配置不可选。显存占用只作为容量约束，最终以稳定吞吐和必要余量选择配置。
- 预计时间按完整样本数乘以测量池的每样本 wall，测量池耗时已包含并发收益，不再次除以 worker 数；小样本估算给出适用范围。

独立测试代码为 `tuning/run.py` 和 `tuning/worker.py`。计划、运行来源、候选结果统一放在 `artifacts/reproductions/evaluation-tuning_20261110/`，只保存小型 JSON 与日志，不生成 tensor dump、profiler trace 或正式 benchmark 分数。正式入口不在参数确定前改动。

在服务器仓库根目录执行，例如：

```bash
python3 reproductions/tuning/run.py \
  --method lact \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --dataset ruler --length 32768 --workers 1 2 4 --num-samples 8 \
  --output-dir artifacts/reproductions/evaluation-tuning_20261110/lact-32k
```

GPU 测量结果与最终参数在完成后补齐。
