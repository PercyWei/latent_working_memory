# 20261108_LaCT 源码来源与接入边界

创建时间：20261108 17:03:39 Asia/Shanghai

最后修订时间：20261116 22:12:29 Asia/Shanghai

## 来源与迁入范围

| 组件 | 官方来源 | 固定 revision | 本地目录 |
|---|---|---|---|
| LaCT | [a1600012888/LaCT](https://github.com/a1600012888/LaCT) | `a648340f9798f173227a0626fde66a6e9b65879a` | `source/lact/` |
| flame | [fla-org/flame](https://github.com/fla-org/flame) | `24dea9133e2166da89a563afbdcefb7c55615646` | `source/flame/` |

LaCT 对应论文 [Test-Time Training Done Right v1](https://arxiv.org/abs/2505.23884v1)，官方来源由[作者项目页](https://tianyuanzhang.com/projects/ttt-done-right/)的 Code 链接确认。两份仓库的 MIT LICENSE 保存在各自根目录。

迁入 LaCT 的语言模型相关内容：顶层 `README.md`、`LICENSE`、完整 `lact_llm/` 和 `minimal_implementations/`。flame 保存固定 revision 的 tracked 文件；根 `.gitignore` 由本仓库统一管理，`.gitmodules` 保留上游元信息。

## 代码职责

| 位置 | 职责 |
|---|---|
| `source/lact/` | 官方模型、配置、TTT 计算函数和最小示例；`original` 调用此实现 |
| `source/flame/`、`configs/760m_muon_paper_single_gpu.toml` | 原训练框架、Transformers 注册接入和单卡训练配置 |
| `optimized/` | 本地增量缓存、混长输入调度及独立 TTT 状态的批量计算 |
| `eval.sh`、`../evaluate_multi_gpu.py`、`../evaluation_runtime.py`、`scripts/` | checkpoint 加载、组批、多卡调度、进度与进程管理、续评、计分和结果比较 |

训练侧接入集中于 `source/flame/custom_models/__init__.py` 导入并注册官方 `lact_model`，以及本地单卡配置和梯度累积。备用入口 [train_760m_muon.sh](scripts/train_760m_muon.sh) 调用原 `flame.train`；训练环境和使用方式见 [README](README.md#备用训练入口)。

## 推理实现改进

| 路径 | 行为 |
|---|---|
| `original`（默认） | 使用官方完整前缀推理，每个生成步骤重新计算历史 chunk；每卡 B1 |
| `optimized` | 缓存窗口 KV、fast weights 和待提交 chunk，复用历史计算；单样本、等长和混长输入共用状态路径 |

`optimized/` 面向公开 Momentum／Muon checkpoint 的非 fused TTT 推理，改进包括：

- **真实进度与位置。** 左 padding 从 attention 的有效序列和 TTT 写入中排除；每条样本按真实 token 数维护 RoPE 位置、窗口 KV 与更新进度。
- **批量 TTT。** 独立权重沿 batch × heads 合并读出，提交按各样本的 chunk 进度及状态 dtype 分组；沿用原始 Momentum／Muon、PreNorm／PostNorm 的计算公式与精度。
- **减少同步。** mask 在入口解析为长度和整数索引，各层使用切片、整数 gather／scatter。HF 标准生成追加的全有效 decode mask 通过张量对象和修改计数确认后复用进度；外部 mask 按实际内容解析。
- **固定待提交缓冲。** 缓冲复用替代逐 token 的小张量列表，保持原先读后写的边界。处理 $n>0$ 个真实 token 后，提交次数为 $\lfloor(n-1)/C\rfloor$，其中 $C$ 为 chunk 大小，恰好填满的末 chunk 在下一真实 token 到来时提交。

两种路径均通过显式矩阵公式在 `torch.no_grad()` 下执行 TTT 更新。性能、数值差异与验证范围统一见[复现状态](../20261110_ttt_reproduction_status.md)。

## 评估流程改进

- **统一组批。** 同一 RULER 长度内按真实 token 数稳定降序，相邻组批，左 padding 到批内最长输入；可按长度分别设置每卡 batch 上限。
- **保留评估协议。** 严格加载完整 checkpoint；输入为 `input + answer_prefix`，greedy 最多生成 128 tokens，按物理输入宽度与各自 EOS 截取，沿用 RULER 原生评分。数据路径固定在 `scripts/records.py`。
- **多卡与续评。** 每卡评估互斥样本，按全局样本 ID 收集历史原始预测、分配未完成输入并核对完整覆盖；固定模型和评估协议，GPU 数量、顺序及 batch 设置可按轮调整。运行目录锁与存活 worker 检查协调并发启动。
- **来源与汇总。** 分别记录上游 revision、本地 commit、Git 改动状态、实际实现路径及每轮执行设置，保留历史来源的组批信息；合并时读取原始预测重新计分。
- **结果展示。** [compare_results.py](scripts/compare_results.py) 为公开两头 Momentum／RoPE／PreNorm 760M 的完整六任务、四长度 RULER 结果生成离线论文对比，包括分组均值、分组对比和逐任务图表。

运行参数、输出命名与目录结构见 [README](README.md#环境与运行命令)。

## 公开语言模型权重

作者在[官方发布回复](https://github.com/a1600012888/LaCT/issues/2#issuecomment-3282879455)中提供 [YunjinZhang/lact_llm](https://huggingface.co/YunjinZhang/lact_llm/tree/main)。这些 760M 模型按论文设置重新训练，与论文表格来自不同训练运行。

20261109 读取三个 safetensors 头部元数据，确认实际权重与配置的 fast-weight heads 一致。目录保留发布原名，加载以实际配置和权重形状为准：

| 子目录 | `config.num_lact_heads` | 第 0 层 `attn.w1` | 第 0 层 `attn.w0.w_left` |
|---|---:|---|---|
| `lact-momentum-rope-prenorm-nheads4-chunk2048-760m` | 2 | `[2,768,768]` | `[2,768,32]` |
| `lact-muon-nope-postnorm-nheads2-chunk2048-760m` | 4 | `[4,384,384]` | `[4,384,32]` |
| `lact-muon-nope-prenorm-nheads2-chunk2048-760m` | 4 | `[4,384,384]` | `[4,384,32]` |

### 与论文版本的已知差异

论文 Appendix 的 Muon 描述采用五轮相同的 `(a,b,c)=(3.4445,-4.7750,2.0315)`；迁入官方 [ttt_operation.py](source/lact/lact_llm/lact_model/ttt_operation.py) 的 `zeropower_via_newtonschulz5` 采用五组逐轮变化的系数，并注明参考后续实现。当前 Muon 评估沿用该固定源码；纯 Momentum 分支使用自身更新规则。

这里记录的是本次迁入的固定快照；论文训练使用的精确 LaCT／flame revision 待确认。配套依赖版本见[评估环境](../20261109_environment_setup.md)。
