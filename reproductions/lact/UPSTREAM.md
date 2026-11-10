# 20261108_LaCT 源码来源与接入边界

创建时间：20261108 17:03:39 Asia/Shanghai

最后修订时间：20261110 11:34:26 Asia/Shanghai

## 代码职责

| 层次 | 本地位置 | 职责 |
|---|---|---|
| 迁入源码（source） | `upstream/`、`upstream_flame/` | 保存固定 revision 的 LaCT 与 flame；下文逐项说明迁入范围和必要接入差异 |
| 本地接入（adapters） | `scripts/`、`eval.sh`、`configs/` | 加载 checkpoint、准备运行环境、选择推理路径、评估、验证与记录结果 |
| 本地优化（optimization） | `incremental/` | 实现 sliding-window KV 与 TTT 状态缓存，复用历史计算 |

`original` 指迁入模型经本地评估接入运行的原始推理路径，不表示依赖、公开 checkpoint、数据构造和评估入口均与论文实验环境完全相同。`incremental` 使用本地缓存实现；本次缓存优化没有修改 `upstream/` 中的 LaCT 模型、配置或 TTT 更新函数，也没有改动 flame trainer。flame 注册所需的接入文件是下文单列的既有差异。

新运行将固定上游 revision 与本地 commit、Git 改动状态、实际入口和实现路径分别记录；续评保留首次计划来源，并在每次执行中记录当前来源。字段含义及输出位置见 [来源记录与输出](README.md#来源记录与输出)。

## 来源

| 组件 | 官方来源 | 固定 revision | 本地目录 |
|---|---|---|---|
| LaCT | <https://github.com/a1600012888/LaCT> | `a648340f9798f173227a0626fde66a6e9b65879a` | `upstream/` |
| flame | <https://github.com/fla-org/flame> | `24dea9133e2166da89a563afbdcefb7c55615646` | `upstream_flame/` |

LaCT 来源由作者项目页 <https://tianyuanzhang.com/projects/ttt-done-right/> 的 Code 链接确认。论文版本为 [arXiv:2505.23884v1](https://arxiv.org/abs/2505.23884v1)。两份仓库的 MIT LICENSE 分别保存在各自根目录。

LaCT 迁入范围为顶层 `README.md`、`LICENSE`、完整 `lact_llm/`、完整 `minimal_implementations/`。视频和视图合成不属于本次语言模型复现范围，未迁入；未迁入模型、样例图片或 `.git`。flame 保存该 revision 的完整 tracked 文件，除根 `.gitignore` 外；省略 `.gitignore` 是为了防止上游的 `*.sh` 等规则漏掉本仓库需要管理的源码。flame 的 `.gitmodules` 仅保留为上游元信息，当前快照没有引入额外 git submodule 或模型资源。

## 接入改动

- LaCT 模型、配置和更新函数原样保存。
- flame 新增 `custom_models/__init__.py`，内容为 `import lact_model`，通过 Transformers 的官方注册入口载入未修改的 LaCT 模型。原快照的 `custom_models/` 是 namespace package，仅复制文件并不能自动执行 LaCT 注册。
- 新增单卡 TOML 和 shell 入口。shell 直接调用官方 `flame.train`，接受物理 GPU 0 或 1（默认 0）、一个进程、按调用目录解析的绝对输出路径与 `PYTHONPATH`，不修改 trainer。原 flame `train.sh` 保留供对照，本地入口不调用其中未注册自定义模型的配置探测及对不存在的 `3rdparty/` 的复制操作。
- 新增 `eval.sh` 与 `scripts/evaluate.py`，读取本地六项 RULER 数据，严格拒绝缺失或额外权重。`--inference-mode original|incremental` 显式选择推理路径，默认 `original` 调用迁入模型的原始 `generate` 与完整前缀路径；两种模式共用 checkpoint、prompt、greedy、128-token 预算和评分。正式输入为每任务、每长度 500 条；`--resume` 核对推理模式等原计划设置，仅追加尚未完成的样本，并从全部预测重新计分。
- 新增 `incremental/`，在上游源码之外实现每层 sliding-window KV 和 TTT 状态缓存。TTT 按官方非 fused 公式实现可复用的范数、读出与更新编译函数，并复用 `silu_backprop`、Muon helper；保存 PreNorm 累计量、初始行范数、Muon 前 Momentum 和待写 chunk，保持满尾段在下一 token 到来时才提交。当前支持单样本、无 padding、非 fused TTT；原始 `upstream/` 保持不变。
- 新增 `scripts/validate_incremental.py`，比较 chunk 边界和真实样本 logits、独立 greedy 生成的 token IDs、缓存进度及跨样本隔离。支持范围与数值、耗时结论见[复现状态](../20261110_ttt_reproduction_status.md)。

## 公开语言模型权重

作者在[官方发布回复](https://github.com/a1600012888/LaCT/issues/2#issuecomment-3282879455)中提供 [YunjinZhang/lact_llm](https://huggingface.co/YunjinZhang/lact_llm/tree/main)，说明按论文设置重训了 760M 模型；不代表原论文指标对应的同一次训练产物。

20261109 仅读取三个 safetensors 文件的头部元数据，确认配置与实际权重的 fast-weight heads 一致，目录名则不一致：

| 子目录 | `config.num_lact_heads` | 第 0 层 `attn.w1` | 第 0 层 `attn.w0.w_left` |
|---|---:|---|---|
| `lact-momentum-rope-prenorm-nheads4-chunk2048-760m` | 2 | `[2,768,768]` | `[2,768,32]` |
| `lact-muon-nope-postnorm-nheads2-chunk2048-760m` | 4 | `[4,384,384]` | `[4,384,32]` |
| `lact-muon-nope-prenorm-nheads2-chunk2048-760m` | 4 | `[4,384,384]` | `[4,384,32]` |

评估按实际 checkpoint 配置构造并严格加载完整 state dict，不通过改 head 数、忽略 shape mismatch 或随机初始化使模型继续运行。读取元数据不是 GPU 模型加载或推理验证。

官方 `prepare_inputs_for_generation` 与 `forward` 均禁用 cache，TTT 层每次根据完整前缀重新计算 chunk 状态；该行为仅对应 `original` 模式。原更新函数使用负内积梯度的显式矩阵运算，两种推理模式均在 `torch.no_grad()` 下执行记忆更新。`original` 的耗时不代表增量解码性能；`incremental` 的数值一致性、速度和支持范围单独记录。

## 与论文版本的已知差异

论文 Appendix 的 Muon 描述使用五轮相同的 `(a,b,c)=(3.4445,-4.7750,2.0315)`。当前官方 `ttt_operation.py` 的 `zeropower_via_newtonschulz5` 使用五组不同系数，并说明参考了后续实现。当前源码 revision 没有在官方文档中标为论文训练的精确 revision。

因此，本次先保留官方现状；不能将未来使用该快照训练的结果无条件称为精确重放论文 v1 的 Muon 设置。需进一步定位作者实验代码 revision 或明确记录差异。Momentum 分支不受该 Muon 系数差异影响。

外部环境仍需解析 FLA、Transformers、torchtitan 的配套版本。当前保存的 flame revision 是本次接入的固定来源，不代表原论文使用的框架 revision 已经确认。
