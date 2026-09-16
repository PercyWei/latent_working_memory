# 20260915_GMSA 来源与迁移边界

创建时间：20260915 19:10:20 UTC+08:00

最后修订时间：20260916 22:45:47 UTC+08:00

上游：[Twilightaaa/GMSA](https://github.com/Twilightaaa/GMSA)，固定提交
`2da109e7da39805430e1efebe620f2c5cc6e94c9`。许可证为 MIT，原版权声明保存在
[LICENSE.GMSA](LICENSE.GMSA)。源代码通过 Git 获取，没有下载基座或实验数据。

## 保留的计算与训练关系

- `modeling_gmsa.py` → `gmsa.py`：同基座的截层 causal encoder（LoRA）、group mean pooling、从基座前层初始化的 LSA、完整 decoder。
- 首尾不足一个 group 时，只平均实际有效 tokens；不将 padding 纳入平均。
- AE 阶段仅训练 encoder LoRA 与 LSA Transformer blocks，LSA 最终 norm 冻结。
- QA 阶段冻结 encoder 与 LSA，全量训练 decoder。没有改成 reader LoRA。
- 原生 AE 指令 `Restate the aforementioned Text.`；目标正文后附 EOS。
- 上游的原始 `input`、`prompt`、`answer` JSONL 语义；v2 将 answer 列表确定为唯一格式。
- 原生 greedy generation 的 repetition penalty 1.3，支持显式覆盖。

## 有意进行的适配

- 模型、数据和训练入口按职责拆开，暴露 `encode`（pre-LSA）、`align`、`read`。
- 不保留未使用的线性对齐备选分支，以及 LSA 的 embedding／LM head；直接使用 decoder stack 与最终 norm。encoder 保留 PEFT 所需的原生模型封装。
- 逐样本先拼接 memory、prompt、target，再统一 padding；避免上游 batch 内部 padding 空洞改变位置和 target 对齐。生成使用左 padding。
- 不截断原文或答案。样本超过配置预算时明确失败，避免 AE 目标包含未被 encoder 看到的内容；数据筛选由调用方在共享数据选择流程中完成。
- PyTorch Dataset 读取共享 JSONL 并在内存分词，替代 datasets.map；无需生成派生副本。
- 使用根环境的 Transformers 4.x / PEFT，不引入上游 W&B、DeepSpeed 和额外环境。当前训练入口支持单设备／DDP；仅保存本地日志，不接入 SwanLab。
- 所有压缩率共用模型参数；训练每个 batch 采样一个已配置压缩率，DDP 广播同一个取值。dev 遍历全部配置压缩率，不随机选择。
- 冻结 encoder 在 QA／dynamic 阶段使用 eval 模式，防止 LoRA dropout 改变固定表示。启用 activation checkpointing 时 decoder 仍保持训练模式以执行重计算；冻结参数不等于关闭输入梯度。
- 模型 checkpoint 固定为 `model.json`、`stage.json`、`model.safetensors`；完整状态严格加载，不尝试上游多格式／部分加载。HF `checkpoint-N` 另保存 optimizer、scheduler、trainer state 与 RNG，阶段初始化和同 run 恢复分别处理。

## 验证与声明

`tests/v2/test_upstream_parity.py` 可在 `GMSA_UPSTREAM` 指定该提交源码目录时比较原实现。
在同一真实小型 Llama、同参数、FP32、eval 模式下，长度 3／4／5 × 压缩率 2／4 的单样本 logits 与 loss 通过 `rtol=1e-5, atol=1e-6` 比较。
变长 batch 的比较以逐样本计算为准，因为 padding 修正有意改变上游 batch 路径。

当前是源码迁移与随机小模型工程验证，不是 GMSA 的论文效果复现。
配置使用 4K 输入上限、SDPA 和当前项目依赖，不等于上游全部训练设置。
后续必须固定真实基座 revision、数据来源和配额，验证静态任务效果后再训练动态模块。
