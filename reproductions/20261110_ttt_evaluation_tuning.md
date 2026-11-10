# 20261110_TTT 正式评估运行参数测试

创建时间：20261110 11:51:20 Asia/Shanghai

最后修订时间：20261110 13:19:33 Asia/Shanghai

## 目标与约束

在 `zcy-memory` 的单张 A800-80GB、物理 GPU 0 上，测量正式评估的显存与吞吐，选择运行参数并估算完整评估耗时。只调整运行调度，不改变 checkpoint、生成预算、停止条件、任务或数据规模，不实施新的模型优化。

当前三种 TTT 路径的可用单进程 batch 均为 1：LaCT 增量 attention 明确限定单样本；E²-TTT 的多样本 mask、decode 归一化和动态裁剪不能直接保持单条语义；In-Place TTT 官方 `generate` 与 fast weights 也限定单样本。本轮区分每进程 batch 与同卡独立进程数，测试后者，不能将并发数称为 batch size。

## 测量方法

- 首批 checkpoint：LaCT Momentum 增量路径；E²-TTT MLP／SwiGLU 官方路径；In-Place TTT 的 TTT／baseline 完整 attention。In-Place TTT 先用原始路径比较短输入并发，再用仅尾部缓存压紧、MLP 不分块的路径测量 TTT 长输入；baseline 保持原始路径。
- 每个 worker 独立加载模型，逐条使用 fresh cache；CPU 线程设为 1，显式 `CUDA_VISIBLE_DEVICES=0`。不使用线程共享模型，不修改上游文件。
- 同一比较使用相同真实输入，单进程作为生成 IDs 参考。各 worker 完整预热后，在统一屏障后开始测量；模型载入和预热单独记录。
- 首轮比较 1／2／4 个进程，必要时扩展至 6／8；随后测量短输入及 E²-TTT 两个 recall benchmark。In-Place TTT 覆盖全部正式长度；LaCT、E²-TTT 用短、中、长输入估算未测长度。256K 只测试单进程。
- 通过整卡 `nvidia-smi` 采样记录并发总占用，另记录每进程 allocated／reserved 峰值。OOM 或生成不一致的配置不可选。显存占用只作为容量约束，最终以稳定吞吐和必要余量选择配置。
- 预计时间按完整样本数与实测速度计算；测量池耗时已包含并发收益，不再次除以 worker 数。LaCT 分片数贯穿全量输入，E²-TTT 则按八个原生 job 的实际排队估算，计入尾部并发不足。小样本估算不等同于正式评估耗时。

独立测试代码为 `tuning/run.py` 和 `tuning/worker.py`。计划、运行来源、候选结果统一放在 `artifacts/reproductions/evaluation-tuning_20261110/`，完成后合并为小型记录，不保留重复逐进程日志、屏障文件或完整显存采样序列。保留 `measurements.json`（34 组计划、53 个 trial、完整生成 IDs、逐进程环境／资源及验证结果）和 `selection.json`（最终设置与估时参数）。本轮没有正式 benchmark 分数。

在服务器仓库根目录执行，例如：

```bash
python3 reproductions/tuning/run.py \
  --method lact \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --dataset ruler --length 32768 --workers 1 2 4 --num-samples 8 \
  --output-dir artifacts/reproductions/evaluation-tuning_20261110/lact-32k
```

## 并发测试结果

下表为整池吞吐，单位为条／秒。各比较使用相同输入：通常 8 条，6 进程测试为 12 条。6 与 4 进程使用的样本数不同，约 10% 的差异只能作为本轮选参依据，不代表精确加速比。

| checkpoint／输入 | 1 进程 | 2 进程 | 4 进程 | 6 进程 | 8 进程 |
|---|---:|---:|---:|---:|---:|
| LaCT Momentum，32K | 0.0999 | 0.1879 | 0.3191 | — | **0.3513** |
| E²-TTT MLP，16K | 0.1053 | 0.1553 | **0.2191** | 0.2185 | — |
| E²-TTT SwiGLU，16K | 0.1048 | 0.1466 | 0.1978 | **0.2178** | — |
| In-Place TTT 原始，32K | 0.1755 | 0.1860 | 0.1873 | — | — |
| In-Place baseline，32K | 0.1798 | 0.1912 | 0.1939 | — | — |
| In-Place TTT 仅压紧尾部，128K | **0.02730** | 0.02482 | — | — | — |
| In-Place baseline，128K | **0.02730** | 0.02501 | — | — | — |

128K 比较各用两条长输入，双进程的 TTT／baseline 峰值分别为 75.8／77.4 GiB，整体耗时却增加约 10.0%／9.2%。256K 单进程均约 127 秒／条。In-Place TTT 的短输入并发收益小、长输入反而变慢，统一采用单进程。

LaCT 8 进程相对 4 进程仍提高约 10% 的吞吐，保留约 11 GiB 整卡余量。E²-TTT MLP 的 6 进程比 4 进程多占约 22.5 GiB，却没有可见提速，因此选 4；SwiGLU 本轮 6 进程的吞吐较高，暂选 6。显存占满本身不作为优化目标。

共完成 53 个计时 trial；19 组包含单进程对照的并发比较、累计 212 条生成序列的完整 token IDs 均一致（包含不同候选间重复的输入）。仅运行最终并发数的长度补测，其原记录一致性字段为 `null`。后补的 E²-TTT 512／4K 单进程各两条与并发结果的重合部分一致，不能扩展为整组 8／12 条均已配对；TTT 32K 仅压紧尾部与原始路径的重合两条也一致。

## 正式运行设置

| checkpoint | 每进程 batch | 同卡独立进程数 | 所测整卡峰值 | 路径 |
|---|---:|---:|---:|---|
| LaCT Momentum 760M | 1 | 8 | 68.8 GiB | `incremental`，八个互不重复的行分片 |
| E²-TTT MLP 1.3B | 1 | 4 | 47.9 GiB | 原始实现，八个原生 job 最多四个并发 |
| E²-TTT SwiGLU 1.3B | 1 | 6 | 71.9 GiB | 原始实现，八个原生 job 最多六个并发 |
| In-Place TTT 32K TTT | 1 | 1 | 67.8 GiB | `optimized --mlp-chunk-size 0`，仅压紧尾部 |
| In-Place TTT 32K baseline | 1 | 1 | 69.0 GiB | `original` |

峰值为 `nvidia-smi` 整卡占用，包含模型载入和预热，与单进程 PyTorch allocated 不同。E²-TTT 并发峰值来自所有进程同时处理 16K 的压力测试；正式八个 job 的输入长度不同，整卡峰值可能更低。三个方法的 worker 池顺序独占同一张卡，不同时运行不同方法。

生成设置保持原协议：LaCT／E²-TTT RULER 最多 128 tokens，E²-TTT recall 最多 48 tokens，In-Place TTT 最多 64 tokens；保留原 EOS／换行停止与长度边界，memory 状态不跨样本共享。TTT chunk、SWA、目标函数与 checkpoint 均未更改。

运行与续评的完整命令维护在 [LaCT](lact/README.md#环境与运行命令)、[E²-TTT](e2_ttt/README.md#环境与评估命令)、[In-Place TTT](in_place_ttt/README.md#ruler-评估)。LaCT 必须等待全部分片成功后执行 `merge_evaluation.py`；E²-TTT 使用 `eval_parallel.sh` 调度原有 `eval.sh`，保留原生子任务结果。

## 完整评估时间估算

LaCT 在 4K／32K 的八进程池每条耗时为 2.227／2.847 秒，对 8K／16K 按输入长度线性插值。每档 3,000 条，合计约 **8.24 小时**；完整加载、预热、评分与汇总另留余量。

E²-TTT 在 512／4K／16K 测量 RULER，1K／2K／8K 按输入长度插值；SQuAD／SWDE 各均匀抽取 24 条。不能把 10,095 条直接除以满并发吞吐：正式入口只有八个原生 job，队列结束后并发逐步减少。估算按固定 job 顺序模拟完成与补位，并以单进程和所选并发的实测服务时间插值剩余进程数的速度。混合长度之间的实际争用可能不同，此模型仅用于安排时间。

| checkpoint | 完整规模 | 外推计算时间 | 建议预留时间 |
|---|---:|---:|---:|
| LaCT Momentum | 12,000 | 8.24 小时 | 约 8–10 小时 |
| E²-TTT MLP | 10,095 | 5.42 小时 | 约 4–7 小时 |
| E²-TTT SwiGLU | 10,095 | 5.26 小时 | 约 4–7 小时 |
| In-Place TTT：TTT 权重 | 9,100 | 68.92 小时 | 约 59–80 小时 |
| In-Place TTT：baseline | 9,100 | 68.62 小时 | 约 58–79 小时 |

In-Place TTT 每档 1,300 条，七档分别按真实输入单进程均值外推。TTT 的 128K／256K 分别约需 13.23／45.91 小时，完整评估无法控制在半天内；增加进程未改善这一结果。

五个 checkpoint 顺序完成约 **156.5 小时，6.5 天**。E²-TTT 的队列顺序保留 RULER 16K→512，然后 SQuAD、SWDE；没有改变原生 job 或数据规模。

上述范围用于规划，不是统计置信区间（E²-TTT 约 ±25%、In-Place TTT 约 ±15% 的速度敏感性余量），假定单卡独占、当前环境和生成设置不变；样本 EOS 长度、任务分布、GPU 频率与编译／评分开销均可能影响结果。五个 checkpoint 顺序评估，参数测试没有替代正式全量准确率评估。

## 代码与验证范围

- `b33d7177`：独立调参控制器与 worker，初始短／长输入并发测试。
- `375baad4`：LaCT 行分片与精确汇总；补充模式参数、仅测试最终并发数时的一致性字段；128K／256K 测量。
- `afbe1a86`：E²-TTT 原生 job 调度；后续短输入与 recall 补测。各次计划另保存实际完整 Git commit、dirty／status 和运行环境；本轮 34 组测量计划的 Git 状态均为 clean。
- LaCT 分片汇总用六组真实数据的 3,000 个位置、8 个不等长分片及合成预测进行检查，与未分片的原评分一致；重复／缺失位置被拒绝。E²-TTT 汇总用八个合成原生 summary 检查 10,095 条完整规模、原指标直接保留及错误样本数拒绝。以上是调度／汇总检查，不是模型准确率测试。
- 所有本轮改动均在评估调度、测试与文档层；未改三个方法的 `upstream/` 或已有增量／优化模型实现。
