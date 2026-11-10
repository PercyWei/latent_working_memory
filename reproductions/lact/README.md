# 20261108_LaCT 第一阶段复现

创建时间：20261108 17:03:39 Asia/Shanghai

最后修订时间：20261110 11:34:26 Asia/Shanghai

本目录接入 [Test-Time Training Done Right](https://arxiv.org/abs/2505.23884v1) 的官方语言模型实现，保留与 sliding-window attention 并行的 SwiGLU fast weights、负内积写入及 Momentum／Muon 更新。源码、环境和 checkpoint 运行检查已完成，正式 RULER 评估尚未完成。

| 代码范围 | 位置 | 职责 |
|---|---|---|
| 迁入源码 | `upstream/`、`upstream_flame/` | 固定 revision 的 LaCT 与 flame；来源和必要注册改动见 [UPSTREAM.md](UPSTREAM.md) |
| 本地接入 | `scripts/`、`eval.sh`、`configs/` | checkpoint 加载、评估、单卡训练与来源记录 |
| 本地优化 | `incremental/` | attention KV、fast weights、Momentum 与待写 chunk 的增量缓存 |

验证、优化结果及适用范围统一记录在[复现状态](../20261110_ttt_reproduction_status.md)。增量实现不修改官方模型、配置或 TTT 更新函数。

## 评估目标与数据

首批评估作者公开重训的 **760M Momentum／RoPE／PreNorm**，两个 Muon 变体按需补充。这些 checkpoint 使用论文训练设置重训，不是论文表格对应的同一次训练产物。

完整 RULER 为六项任务、4K／8K／16K／32K、每项每长度 500 条，共 **12,000 条／checkpoint**。三个公开 checkpoint 的 tokenizer 和特殊 token 配置一致，共用 `data/benchmarks/ruler-v1/reproductions/lact-760m/`；数据下载和构造命令见[评估数据准备](../20261108_ttt_data_preparation.md)。

论文 `tab:ruler-single` 的 S-NIAH-1/2/3 平均准确率，Momentum 为 **89.9／64.3／40.9／33.1**，Muon 为 **77.9／60.5／40.0／35.1**；`tab:ruler-multi` 的 MK-NIAH、MQ-NIAH、MV-NIAH 平均，Momentum 为 **41.7／29.7／18.5／9.1**，Muon 为 **42.9／32.1／17.1／7.5**。以上依次对应 4K／8K／16K／32K，单位为 %，仅作原文参照。

## 模型下载

作者在[官方回复](https://github.com/a1600012888/LaCT/issues/2#issuecomment-3282879455)中发布了 [YunjinZhang/lact_llm](https://huggingface.co/YunjinZhang/lact_llm/tree/main)，使用 32K 序列、Long-Data-Collections 和约 40B tokens 重训。`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
# 首批 Momentum／RoPE／PreNorm
hf download YunjinZhang/lact_llm \
  --include 'lact-momentum-rope-prenorm-nheads4-chunk2048-760m/*' \
  --local-dir ~/rivermind-data/models/reproductions/lact

# 可选：Muon／NoPE 的 PreNorm 和 PostNorm
hf download YunjinZhang/lact_llm \
  --include 'lact-muon-*/*' \
  --local-dir ~/rivermind-data/models/reproductions/lact
```

目录名与实际 fast-weight head 数不一致：Momentum 配置为 **2 heads**，两个 Muon 为 **4 heads**。加载时以 checkpoint 自带配置为准，严格拒绝 missing、unexpected、mismatched keys，不根据目录名改配置；形状记录见 [UPSTREAM.md](UPSTREAM.md)。

## 环境与运行命令

使用 [requirements-eval.txt](requirements-eval.txt) 固定的独立评估环境，版本表见[评估环境](../20261109_environment_setup.md)。从服务器项目根目录安装并激活：

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh lact
source ~/rivermind-data/envs/ttt-lact/bin/activate
```

确认物理 GPU 0 空闲后运行。推荐首批 Momentum 使用已验证的增量路径：

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/lact/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --inference-mode incremental \
  --output-dir artifacts/reproductions/lact/eval/momentum-rope-prenorm-ruler-incremental
```

保留官方完整前缀路径作对照：

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/lact/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --inference-mode original \
  --output-dir artifacts/reproductions/lact/eval/momentum-rope-prenorm-ruler-original
```

| 参数 | 默认值与约定 |
|---|---|
| `--ruler-data` | `data/benchmarks/ruler-v1/reproductions/lact-760m` |
| `--lengths` | `4096 8192 16384 32768` |
| `--inference-mode` | `original`；可选 `incremental`，两者使用不同输出目录 |
| `--resume` | 同一设置与已有输出目录续评 |

`eval.sh` 使用当前环境的 `python`，仅接受单卡 `CUDA_VISIBLE_DEVICES=0` 或 `1`，默认 0。首次运行输出目录须不存在。中断后原命令追加 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/lact/eval.sh \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --inference-mode incremental \
  --output-dir artifacts/reproductions/lact/eval/momentum-rope-prenorm-ruler-incremental \
  --resume
```

续评核对 checkpoint、数据、长度、推理模式、生成与精度设置，跳过已有样本并重新汇总。仅自动移除末行未写完的 JSON，其他损坏、重复或不匹配记录报错。

每条输入为 `input + answer_prefix`，由 checkpoint tokenizer 添加默认 special tokens；完整输入加 128-token 输出预算不得超过长度档位，不截断或现场构造数据。采用单样本 greedy，参数 float32、CUDA BF16 autocast、关闭外层 autograd，仍执行显式 TTT 更新。评分复用 NVIDIA RULER 的 `postprocess_pred` 与 `string_match_all`。

`original` 调用迁入模型的 `generate`，强制 `use_cache=False`，逐 token 重算完整前缀。`incremental` 缓存历史状态，保持先读后更新及尾段延迟写入：前缀长为 \(L\) 时提交 \(\lfloor(L-1)/2048\rfloor\) 次更新，当前支持单样本、无 padding、非 fused TTT。

**增量路径支持范围：**Momentum 通过四长度、八条独立 128-token 生成的配对检查；Muon／PostNorm 仅通过 4K 检查；Muon／PreNorm 尚未通过，使用原始路径。代表性检查不保证任意输入逐位相同，也不替代正式评估。

## 来源记录与输出

`plan.json.source` 保存上游 revision、本地 commit、Git dirty/status、入口及实际实现路径。续评保留首次计划来源，每次执行的当前来源写入 `execution.attempts[]`，逐样本 `execution_attempt` 关联对应执行；代码来源变化会留痕。

| 文件 | 内容 |
|---|---|
| `plan.json` | checkpoint、数据、规模、生成参数、精度及来源 |
| `predictions.jsonl` | 逐样本生成、参考、分数、token 数、耗时和执行关联 |
| `summary.json` | 全部样本完成后的任务／长度分数、三项 S-NIAH／三项 multi／六项平均 |
| `execution.json`、`evaluation.log` | 历次状态、累计耗时、显存峰值与运行日志 |

汇总从全部已保存预测重新计分，原始分数不取整，汇总百分比保留两位小数。部分预测不代表完整结果，累计耗时包括加载、编译和失败执行。

需要重新验证增量实现时，从项目根目录执行：

```bash
source ~/rivermind-data/envs/ttt-lact/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/reproductions/lact:$PWD/reproductions/lact/upstream/lact_llm:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -u reproductions/lact/scripts/validate_incremental.py \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --output-dir artifacts/reproductions/lact/incremental-validation/momentum \
  --lengths 4096 8192 16384 32768
```

验证比较 chunk 边界、logits、完整生成及跨样本缓存隔离，输出目录须新建。配对入口使用 `IncrementalLaCTForCausalLM`，原始路径以 `use_cache=False` 委托父类；实际来源应结合模式与 forward 路径判断。`--reference-output` 可复用包含完整 `plan.source` 的既有参考。

## 备用训练入口

本阶段使用公开 checkpoint，不执行训练。原设置为 32K 序列、TTT chunk／SWA 2048、Long-Data-Collections 与 Mixstra tokenizer；公开命令训练 40960 steps、全局 batch 32，精确预算 42,949,672,960 tokens，论文称约 40B。Muon 实现与论文的差异见 [UPSTREAM.md](UPSTREAM.md)。

完整训练环境单独安装：

```bash
python -m pip install -r reproductions/lact/upstream/lact_llm/requirement.txt
python -m pip install ./reproductions/lact/upstream_flame
python -m pip install 'torchtitan @ git+https://github.com/pytorch/torchtitan.git@0b44d4c'
```

准备原始 Arrow 数据和 tokenizer 后，从项目根目录运行：

```bash
export DATA_FILES='/absolute/path/to/long-data-collections/*.arrow'
export TOKENIZER_PATH='/absolute/path/to/original-mixstra-tokenizer'
CUDA_VISIBLE_DEVICES=0 bash reproductions/lact/scripts/train_760m_muon.sh
```

入口调用官方 `flame.train`，单卡 microbatch 1、梯度累积 32、一个进程；默认输出 `artifacts/reproductions/lact/760m-muon-paper/train/`，可用 `OUTPUT_DIR` 指定目录。训练框架和原训练数据尚未完成运行验证，Book-3 来源与划分尚待落实。

可直接追加原生 flame 参数，例如执行一个实际 optimizer step：

```bash
CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR='/absolute/path/to/artifacts/reproductions/lact/760m-muon-one-step/train' \
  bash reproductions/lact/scripts/train_760m_muon.sh --training.steps 1
```

该命令仍执行 32K 序列和 32 次梯度累积。flame 保存 DCP checkpoint、配置与本地 TensorBoard，关闭 W&B；HF 转换入口为 `upstream_flame/flame/utils/convert_dcp_to_hf.py`。
