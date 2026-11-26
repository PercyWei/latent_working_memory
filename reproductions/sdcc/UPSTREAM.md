# 20261125_SDCC 源码来源与适配

创建时间：20261125 21:06:58 Asia/Shanghai

最后修订时间：20261126 10:50:56 Asia/Shanghai

| 组件 | 来源 |
|---|---|
| 论文 | [Density-aware Soft Context Compression with Semi-Dynamic Compression Ratio](https://arxiv.org/abs/2603.25926) |
| 官方源码 | [yuyijiong/semi-dynamic-context-compress](https://github.com/yuyijiong/semi-dynamic-context-compress/tree/07fae01c5062e57dc8277f7e696ca928cb775ca5)，提交号保存在 [source/revision](source/revision) |
| 半动态权重 | [yuyijiong/qwen3-semi-dynamic-soft-context-compress](https://huggingface.co/yuyijiong/qwen3-semi-dynamic-soft-context-compress/tree/1f73dfe7c6f4acde9b81ecf678f55922179759cd/semi-dynamic)，使用编码器和解码器均为 Qwen3-4B-Instruct-2507 的子目录 |
| 基座与 tokenizer | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554)，Apache-2.0 |

[source/sdcc](source/sdcc/) 保留该提交的模型、训练与评估代码、模板、配置和 README。作者的四个 QA 评估数据文件位于[上游 Eval/eval_data](https://github.com/yuyijiong/semi-dynamic-context-compress/tree/07fae01c5062e57dc8277f7e696ca928cb775ca5/Eval/eval_data)。该提交未附仓库级 LICENSE，作者的权重模型卡将许可证标为 `cc`。

## 官方模型与权重

[CtxCompSemiDynamicModel](source/sdcc/modeling_ctxcomp.py) 从编码器输入末尾的 EOS 隐藏状态预测压缩倍率。对于选定的权重，选择器把预测值加上 `compress_ratio_scale` 后从 log₂ 空间还原，在 2、4、8、16、32 倍中取最近值。选中倍率为 `r` 时，对连续 `r` 个隐藏状态做均值池化，再投影到解码器的嵌入空间。长度为 `L` 的输入产生 `ceil(L / r)` 个软 token。

公开 checkpoint 的 `config.json` 记录训练时的编码器与解码器基座。对于两端基座相同的权重，`--model-path` 指向其本地副本。官方 4B 权重将 LoRA 放在 `encoder/resumed/`、`decoder/resumed/`，同基座的 0.6B 权重则直接放在 `encoder/`、`decoder/`。适配器按各自的 PEFT 配置定位权重，两套发布的投影层与选择器均保存在 `projector.pth`。

## LoCoMo 适配

[官方评估](source/sdcc/Eval/eval_ctxcomp_semi_dynamic.py) 面向短篇 QA，默认正文上限为 2,048 tokens。[本项目的 LoCoMo 评估](README.md)保留完整多轮对话，题目和系统提示只交给解码器。同一对话的不同问题共用压缩后的软 token，评分与续评采用[项目公共协议](../../benchmarks/locomo/README.md)。
