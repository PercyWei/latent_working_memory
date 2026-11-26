# 20261125_SDCC 复现

创建时间：20261125 21:08:04 Asia/Shanghai

最后修订时间：20261126 10:57:26 Asia/Shanghai

[环境准备](#一环境准备) · [原论文复现](#二原论文复现) · [通用功能（LoCoMo）](#三通用功能)

## 一、环境准备

推理默认以 BF16 加载模型权重，关闭模型级自动混合精度（autocast）。编码器使用双向注意力，解码器使用因果注意力，两者均使用 FlashAttention 2。

从服务器的项目根目录创建并激活独立评估环境。依赖版本见 [requirements-eval.txt](requirements-eval.txt)。

```bash
bash reproductions/setup_eval_env.sh sdcc
source ~/envs/sdcc/bin/activate
```

环境位置、uv 镜像配置与激活方式见[评估环境](../README.md#评估环境)。

## 二、原论文复现

本项目采用 SDCC 的官方半动态压缩实现。选用的公开 checkpoint 以 Qwen3-4B-Instruct-2507 作为编码器和解码器的基座，分别加载训练好的 LoRA。编码器根据正文选择压缩倍率，对连续 token 的隐藏状态做均值池化，再将结果投影为解码器的输入嵌入。[SDCC 论文](https://arxiv.org/abs/2603.25926) · [源码与权重说明](UPSTREAM.md)

原论文在 2,048 tokens 以内的阅读理解样本上评估。作者提供了半动态 SFT 脚本和训练数据，从源码启动训练还需补齐脚本引用的 `sft_ctxcomp_train_config` 模块。[官方训练脚本](source/sdcc/SFT/sft_ctxcomp_semi_dynamic.py) · [训练数据](https://huggingface.co/datasets/yuyijiong/context_qa_sum_qwen3_synthetic)

### 2.1 模型下载

选择编码器与解码器使用同一基座的半动态 checkpoint，并准备与权重配套的基座和 tokenizer。下例使用 Qwen3-4B-Instruct-2507。使用官方同基座的 Qwen3-0.6B 权重时，对应替换基座和 checkpoint 即可。

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir ~/models/Qwen3-4B-Instruct-2507

hf download yuyijiong/qwen3-semi-dynamic-soft-context-compress \
  --include 'semi-dynamic/*-enc=Qwen3-4B-Instruct-2507-lora-dec=Qwen3-4B-Instruct-2507-lora/*' \
  --local-dir ~/models/reproductions/sdcc
```

## 三、通用功能

<a id="locomo-评估"></a>

### 3.1 LoCoMo 评估

评估 LoCoMo 的 10 段完整对话，每道题使用所属对话的全部历史。按上述基座的 tokenizer 统计，对话正文长度为 13,479–25,580 tokens。[数据下载](../../benchmarks/README.md) · [LoCoMo 评分与续评规则](../../benchmarks/locomo/README.md)

| benchmark | 题型 | 题数 | 默认数据路径 |
|---|---|---:|---|
| LoCoMo | 多跳（1）、时间（2）、开放域（3）、单跳（4） | 1,540 | `data/benchmarks/locomo/locomo10.json` |

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/sdcc/scripts/eval_locomo.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --checkpoint ~/models/reproductions/sdcc/semi-dynamic/ctxcomp-semi-dynamic-mean_pooling-doclen=1200to64-ratios=0.5_0.03125-enc=Qwen3-4B-Instruct-2507-lora-dec=Qwen3-4B-Instruct-2507-lora \
  --output-id ctxcomp-semi-dynamic-mean_pooling-doclen=1200to64-ratios=0.5_0.03125-enc=Qwen3-4B-Instruct-2507-lora-dec=Qwen3-4B-Instruct-2507-lora_scale0_20261126 \
  --batch-size-per-gpu 2 \
  --compress-ratio-scale 0
```

| 参数 | 默认值与约定 |
|---|---|
| `--resume` | 继续未完成的评估，默认关闭 |
| `--model-path` | 必填，本地基座模型目录 |
| `--checkpoint` | 必填，SDCC 权重目录 |
| `--benchmark-data` | LoCoMo 数据目录，默认 `data/benchmarks/locomo` |
| `--output-root` | 评估结果根目录，默认 `artifacts/reproductions/sdcc` |
| `--output-id` | 运行标识，默认 `<checkpoint_name>_scale<compress-ratio-scale>_<YYYYMMDD-HHMM>` |
| `--batch-size-per-gpu` | 每卡评估批量大小，默认 1 |
| `--compress-ratio-scale` | 压缩倍率选择器的偏移量，默认 0 |

- Writer 只读取完整对话正文。按基座 tokenizer 的规则分词并以 EOS 结尾，选择器根据末位隐藏状态从 checkpoint 的候选倍率中选一档。示例权重提供 2×、4×、8×、16×、32× 五档。若编码器输入长度为 `L`、选中倍率为 `r`，则生成 `ceil(L / r)` 个记忆 token。
- Reader 通过公共 QA 接口，按基座 tokenizer 的聊天模板构造输入。以 Qwen3-4B-Instruct-2507 为例，`system` 消息放 QA 提示，`user` 消息依次放 `Context:`、对话记忆和简答问题，随后由模板生成 `assistant` 回答起始标记。SDCC 将 checkpoint 指定的占位符展开为记忆嵌入。
- 调度器按对话分配 GPU。同一对话只压缩一次，各题从共享的 Reader 前缀 KV 独立作答，贪心生成最多 50 tokens。

### 3.2 结果展示

结果保存在 `<output-root>/eval/<output-id>/locomo/`。`summary.json` 汇总总体与各题型的 F1 和题数，并记录各对话的压缩档位、正文与记忆 token 数。`details.jsonl` 保存逐题回答、参考答案和 F1。保留比例为记忆 token 数除以正文 token 数，分母只计原始正文，平均值按对话等权计算。

```bash
bash reproductions/sdcc/scripts/dashboard.sh \
  --output-dir artifacts/reproductions/sdcc/eval/ctxcomp-semi-dynamic-mean_pooling-doclen=1200to64-ratios=0.5_0.03125-enc=Qwen3-4B-Instruct-2507-lora-dec=Qwen3-4B-Instruct-2507-lora_scale0_20261126
```

`--output-dir` 指向包含 `locomo/` 的运行目录。看板可按题型和对话筛选逐题回答，并查看每段对话的压缩档位和实际保留比例。
