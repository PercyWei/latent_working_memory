# ICAE v1 复现（20260904 10:13:57 CST）

创建时间：20260904 10:13:57 CST（UTC+08:00）

最后修订时间：20260904 20:16:17 CST（UTC+08:00）

本目录保存后续复现 C-DIC 所需的 ICAE v1 压缩器代码，目标是论文使用的 Llama-2-7B-Chat 路径，而不是后续基于 Mistral 的 ICAE v2。

仓库初始化过程不会保存或下载 benchmark、数据集、模型权重与 checkpoint。

## 源码结构

- `src/icae/`：迁移后的 ICAE v1 模型与训练代码；
- `vendor/peft/`：ICAE 随附的定制 PEFT `0.4.0.dev0` 源码；
- `src/icae_repro/`：无需下载模型的本地环境检查，以及受测试的 checkpoint 加载和推理入口；
- `examples/ft_inference_upstream.py`：上游基于固定路径的原始推理示例，仅用于参考，不作为正式入口；
- `UPSTREAM.md`：上游版本、迁移路径与本地改动记录。

## 服务器环境

目标环境：

- NVIDIA A800-SXM4-80GB；
- Linux x86-64；
- 支持 CUDA 13.0 的服务器驱动与工具链；
- Python 3.10；
- PyTorch 2.0.1 CUDA 11.8 wheel；
- Transformers 4.31.0 与 ICAE 定制 PEFT。

使用 CUDA 11.8 wheel 是有意选择。NVIDIA 驱动能够向后兼容使用旧版 CUDA toolkit 构建的应用，而 PyTorch 2.0.1 比当前 CUDA 13.0 对应版本更接近 ICAE 最初使用的 2023 年依赖栈。迁移代码不编译本地 CUDA extension。

在服务器上创建环境：

```bash
export UV_CACHE_DIR=/data/bywei/cache/uv
uv sync --project reproductions/icae --frozen
uv run --project reproductions/icae icae-check-environment
```

该 uv 项目使用阿里云 PyPI 镜像安装常规依赖，并使用阿里云 CUDA 11.8 wheel 镜像安装 PyTorch。`UV_CACHE_DIR` 将下载缓存保存在 `/data/bywei` 下，便于后续重建环境时复用。

环境检查不会访问 Hugging Face 或下载权重，只报告 PyTorch、CUDA、GPU 计算能力和 bfloat16 支持情况。

环境检查通过后，可运行受测试的单样本推理入口：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project reproductions/icae --no-sync \
  icae-smoke-inference \
  --model-path /data/bywei/models/meta-llama/Llama-2-7b-chat-hf \
  --checkpoint /data/bywei/checkpoints/icae/v1/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt \
  --context "The access code for Project Quartz is ZETA-4827." \
  --prompt "What is the access code for Project Quartz?"
```

公开的 v1 checkpoint 使用标量 `0.0` 作为冻结 Llama 参数的占位符。推理入口会从本地基础模型恢复这些参数，再使用 `strict=True` 严格加载完整 state dict，不会通过 `strict=False` 跳过不匹配项。

ICAE 上游说明要求训练使用 bfloat16 而不是 fp16，且公开训练路径只支持 batch size 1。在受控兼容性测试证明可以修改之前，应保留这些限制。

## PwC 结果生成

`icae-reproduce-pwc` 分别生成 ICAE-128 与完整上下文基线的结果，支持按 sample ID 断点续跑。两个条件应写入不同文件；如需并行，可分别使用物理 GPU 0 和 1：

```bash
export HF_HOME=/data/bywei/cache/huggingface
export UV_CACHE_DIR=/data/bywei/cache/uv
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 uv run --project reproductions/icae --no-sync \
  icae-reproduce-pwc \
  --condition icae-128 \
  --model-path /data/bywei/models/meta-llama/Llama-2-7b-chat-hf \
  --checkpoint /data/bywei/checkpoints/icae/v1/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt \
  --input /data/bywei/datasets/sggetao/PwC/PwC_test.jsonl \
  --output artifacts/icae/20260904_pwc_reproduction/run/predictions_icae.jsonl

CUDA_VISIBLE_DEVICES=1 uv run --project reproductions/icae --no-sync \
  icae-reproduce-pwc \
  --condition full-context \
  --model-path /data/bywei/models/meta-llama/Llama-2-7b-chat-hf \
  --input /data/bywei/datasets/sggetao/PwC/PwC_test.jsonl \
  --output artifacts/icae/20260904_pwc_reproduction/run/predictions_full_context.jsonl
```

ICAE 条件遵循上游的 `[FT] prompt [FT]`、greedy decoding 和 token `1` 停止规则。上游未发布完整上下文 baseline 代码；当前 baseline 将最多 512 个原始上下文 tokens 与 prompt tokens 直接拼接，并使用 Llama tokenizer 的 EOS。该选择必须在结果中标记为实现假设。

## 当前迁移边界

迁移后的模型和 trainer 源码已完成打包与语法检查。上游 v1 推理示例包含占位路径、引用了公开 v1 训练文件中不存在的 tokenization helper，并保留了一个上游变量名拼写错误。为保留来源信息，该示例不作修改；本项目使用受测试的 `icae-smoke-inference` 作为推理入口。

首次批量推理时可准备 JSONL 文件，每行一个样本：

```json
{"input": "long context", "prompt": "question about the context", "answer": "reference answer"}
```

`answer` 仅用于结果比较，不会在生成时提供给模型。

## 暂不纳入仓库的资源

以下内容保留为服务器侧准备事项，不存放在本目录中：

- `meta-llama/Llama-2-7b-chat-hf` 的访问权限与本地模型目录；
- ICAE Llama-2 checkpoint；
- PwC、MSC、REALTALK、LongMemEval 或其他数据集；
- 生成结果、训练 checkpoint 与 benchmark 输出。

不得将这些文件提交到 Git。根目录 `.gitignore` 已排除常见模型与实验产物路径和扩展名。
