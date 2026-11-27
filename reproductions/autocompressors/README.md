# 20261124_AutoCompressors 复现

创建时间：20261124 11:54:28 Asia/Shanghai

最后修订时间：20261127 16:12:47 Asia/Shanghai

[环境准备](#一环境准备) · [原论文复现](#二原论文复现) · [通用功能](#三通用功能)

## 一、环境准备

推理默认以 BF16 加载模型权重，关闭模型级自动混合精度（autocast）。注意力由官方 Llama 实现调用 FlashAttention 2 计算。

从 Linux x86-64 服务器的项目根目录创建并激活独立评估环境。依赖版本见 [requirements-eval.txt](requirements-eval.txt)。固定版本的官方源码随项目保存在 [source/AutoCompressors](source/AutoCompressors/)，安装器仅负责创建环境和安装依赖。

```bash
bash reproductions/setup_eval_env.sh autocompressors
source ~/envs/autocompressors/bin/activate
```

环境位置、uv 镜像配置与激活方式见[评估环境](../README.md#评估环境)。

## 二、原论文复现

本项目采用 AutoCompressors 的官方实现。Princeton 发布的 AutoCompressor-Llama-2-7b-6k 基于 Llama-2-7B 进行语言建模训练，未做 QA 微调。作者提供了 Llama 的[训练脚本](source/AutoCompressors/run/train_llama.sh)和[困惑度评估脚本](source/AutoCompressors/run/eval_llama.sh)。源码与项目适配见 [UPSTREAM.md](UPSTREAM.md)。

### 2.1 模型下载

下载 [AutoCompressor-Llama-2-7b-6k](https://huggingface.co/princeton-nlp/AutoCompressor-Llama-2-7b-6k) 的完整权重与 tokenizer。`hf` CLI 安装方式见[复现总说明](../README.md#模型下载约定)。

```bash
hf download princeton-nlp/AutoCompressor-Llama-2-7b-6k \
  --local-dir ~/models/reproductions/autocompressors/AutoCompressor-Llama-2-7b-6k
```

## 三、通用功能

### 3.1 benchmark 评估

| 完整评估脚本共同参数 | 用途与默认值 |
|---|---|
| `--resume` | 在原评估命令后添加即可从同一结果目录续评 |
| `--checkpoint` | 必填，本地完整的 checkpoint 目录 |
| `--benchmark-data` | 输入数据目录，默认路径见各任务小节 |
| `--output-root` | 输出根目录，默认 `artifacts/reproductions/autocompressors` |
| `--output-id` | 默认 `seg<segment_length>_<task>` |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |
| `--segment-length` | 每段正文的 token 数上限，默认值见各任务小节 |

- **运行方式**：在项目根目录使用 AutoCompressors 环境运行。每张可见 GPU 加载一个模型副本。[环境准备](#一环境准备) · [数据下载](../../benchmarks/README.md)
- **历史压缩**：Writer 逐段压缩正文，每段（含末段）生成 50 个摘要向量。累积摘要与当前正文合计最多占用 4,096 个位置。
- **输入与解码**：Reader 依次读入 `BOS`、系统提示、全部摘要和问题，在 `Answer:` 后贪心生成回答。Reader 输入与回答预算之和也须在 4,096 个位置内。
- **产物保存**：评估结果写入 `<output_root>/eval/<checkpoint_name>/<output_id>/`。`<checkpoint_name>` 取 `--checkpoint` 指向的目录名。`<output_id>` 默认值中的 `<task>` 标识评估任务，具体路径见各任务小节。

<a id="locomo-评估"></a>
<a id="31-locomo-评估"></a>

#### 1. LoCoMo

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |
| `--segment-length` | 默认 `2048` |

- **评估范围**：覆盖 10 段完整对话中类别 1–4 的 1,540 道题。[LoCoMo 评估协议](../../benchmarks/locomo/README.md)
- **推理方式**：按对话分配 GPU，每段对话历史只压缩一次。默认分段下，完整段的压缩率约为 41 倍。同一对话的各题复用摘要与 Reader 前缀缓存，剩余题目可与下一段对话的问题一起解码。
- **回答长度与评分**：每题最多生成 50 个 token。按官方规则逐题计算 F1，再对所有题目取平均。
- **产物保存**：评估结果默认写入 `<output_root>/eval/<checkpoint_name>/seg<segment_length>_locomo/`。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/autocompressors/scripts/eval_locomo.sh \
  --checkpoint ~/models/reproductions/autocompressors/AutoCompressor-Llama-2-7b-6k \
  --batch-size-per-gpu 2
```

<a id="32-longmemeval-评估"></a>

#### 2. LongMemEval

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/longmemeval` |
| `--split` | 支持 `longmemeval_s_cleaned`（默认）和 `longmemeval_oracle` |
| `--segment-length` | 默认 `2304` |
| `--max-new-tokens` | 每题最多生成的 token 数，默认 `128` |

- **评估范围**：S 和 Oracle 使用相同的 500 道题。S 提供证据会话和干扰会话，Oracle 只提供证据会话。
- **推理方式**：相同的完整历史归到同一卡。Writer 并行写入本批各份历史的摘要，Reader 根据各自的摘要、提问日期和问题并行作答。随着摘要累积，Writer 按剩余位置缩短后续正文段，为本段新增的 50 个摘要向量留出空间。
- **输入长度**：按模型 tokenizer 统计，S 的历史长 120,431–132,553 个 token。使用默认分段长度与回答预算时，500 道 S 题均能放入 Writer 和 Reader 窗口。
- **自动评分**：生成全部回答后，脚本调用 `gpt-6-luna`，按官方分题型提示词逐题判定正误，汇总准确率。裁判密钥 `OPENAI_API_KEY` 可在项目根目录的 `.env` 中配置。[裁判配置与评分口径](../../benchmarks/longmemeval/README.md#问答评分)
- **产物保存**：评估结果默认写入 `<output_root>/eval/<checkpoint_name>/seg<segment_length>_<split>/`。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/autocompressors/scripts/eval_longmemeval.sh \
  --split longmemeval_s_cleaned \
  --checkpoint ~/models/reproductions/autocompressors/AutoCompressor-Llama-2-7b-6k \
  --batch-size-per-gpu 2
```

| 独立评分代码参数 | 用途与默认值 |
|---|---|
| `--run-dir` | 必填，所选设置的结果目录 |
| `--benchmark-data` | 默认沿用生成时记录的数据目录 |
| `--resume` | 用于续评分，默认关闭 |

- **独立评分**：已有完整回答时，可在项目根环境中直接评分，无需重新运行 AutoCompressors 推理。

```bash
uv run -m benchmarks.longmemeval.score \
  --run-dir artifacts/reproductions/autocompressors/eval/AutoCompressor-Llama-2-7b-6k/seg2304_longmemeval_s_cleaned
```

#### 3. QASPER

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/qasper` |
| `--split` | 支持 `test`（默认） |
| `--segment-length` | 默认 `2048` |
| `--max-new-tokens` | 每题最多生成的 token 数，默认 `128` |

- **评估范围**：test 包含 416 篇论文、1,451 道题。[QASPER 评估协议](../../benchmarks/qasper/README.md)
- **推理方式**：按论文分配 GPU。Writer 对每篇论文只压缩一次，各题复用摘要和 Reader 前缀缓存，独立作答。后续正文段的长度受剩余窗口限制。
- **评分口径**：在本地逐题计算官方 Answer F1。每题取各参考答案中的最高分，再对所有题目取平均。
- **产物保存**：评估结果默认写入 `<output_root>/eval/<checkpoint_name>/seg<segment_length>_qasper_<split>/`。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproductions/autocompressors/scripts/eval_qasper.sh \
  --split test \
  --checkpoint ~/models/reproductions/autocompressors/AutoCompressor-Llama-2-7b-6k \
  --batch-size-per-gpu 2
```

<a id="33-结果展示"></a>

### 3.2 结果展示

将一个或多个 `<output_root>/eval/<checkpoint_name>/` 传给 `--output-dir`，多个目录可用空格分隔或重复指定该参数。看板按 `summary.json` 中的 benchmark 及其设置或划分展示已评分的运行。每个任务直接列出两级复选框，默认全部选中。每个 checkpoint 下始终列出各运行的 `output_id` 和已选数量。子项可单独勾选。全选时点击 checkpoint 可清空该组，只选部分或未选时点击可全选。

各任务先按 checkpoint 名自然排序，同一 checkpoint 内按分段长度（`segment_length`）升序排列，同参数的运行按 `output_id` 自然排序。每个 checkpoint 使用一种色系。看板按全部输入目录、全部任务中的不同压缩设置预先分配深浅，参数从小到大对应从深到浅的等距色阶，只有一种设置时使用中等明度。同一 checkpoint 和设置跨任务同色，筛选时颜色保持固定。

柱图按核心指标分组，顶部图例列出所有所选运行，柱顶显示百分比分数。评分详细表、LoCoMo 记忆统计和逐题回答随勾选同步更新。清空某个任务的选择后，该任务显示选择提示。

```bash
# 查看一个 checkpoint
bash reproductions/autocompressors/scripts/dashboard.sh --output-dir artifacts/reproductions/autocompressors/eval/AutoCompressor-Llama-2-7b-6k

# 比较两个 checkpoint
bash reproductions/autocompressors/scripts/dashboard.sh --output-dir \
  "artifacts/reproductions/autocompressors/eval/<checkpoint_a>" \
  "artifacts/reproductions/autocompressors/eval/<checkpoint_b>"
```

LoCoMo 摘要用量表列出模型和分段长度，并按对话等权汇总摘要 token 数和保留比例。展开明细可查看每段对话的正文与摘要 token 数及保留比例。

汇总指标保存在各运行的 `summary.json` 中，逐题回答、参考答案及评分保存在 `details.jsonl` 中。页面底部可对照各运行对同一道题的回答和评分。先选择任务，再选择问题。LoCoMo 可按题型和对话筛选，LongMemEval 可按题型筛选，QASPER 可按论文筛选。
