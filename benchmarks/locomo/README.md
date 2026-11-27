# 20261124_公共 LoCoMo 评估

创建时间：20261124 15:43:38 Asia/Shanghai

最后修订时间：20261127 16:36:28 Asia/Shanghai

本目录提供 LoCoMo 的公共评估与看板接口。公共层维护数据、调度、评分和结果格式；各方法维护模型加载、正文处理、答案生成及自己的启动脚本。数据准备见[公共 benchmark 说明](../README.md)。

## 评估约定

- **数据范围**：使用全部 10 段对话的完整历史，评估类别 1–4 的 1,540 道题：multi-hop、temporal、open-domain、single-hop。
- **答案生成**：采用贪心解码，每题最多生成 50 tokens；参考答案仅用于评分。
- **评分口径**：使用官方 F1 规则，总体及各题型均按题取算术平均。分数保存为 0–1，看板以百分制展示。
- **正文复用**：同一对话的待答问题分配给同一 worker，正文状态准备一次，各题从该状态独立作答；完成整段对话后释放状态。
- **续评方式**：按已保存的题目 ID 补齐剩余问题，部分完成的对话重新准备正文状态。

## 代码接入

### 数据与输入构造

[数据层](../data.py) 通过 `load_samples("locomo", data_dir)` 提供 `BenchmarkSample`。其中 `prompt` 为完整正文与问题提示，`references` 为参考答案，`max_new_tokens` 为生成预算，`metadata` 保存题型、对话 ID 和原始问题。

| 维护位置 | 提供的内容与职责 |
|---|---|
| LoCoMo 数据层 | `locomo_history(sample)` 提取正文；`locomo_question_prompt(question, category)` 构造简答要求与问题 |
| [公共提示模块](../../src/latent_working_memory/prompts.py) | 维护 `QA_SYSTEM_PROMPT`，通过 `QAInputBuilder` 构造 reader 输入及其模板片段 |
| 方法适配层 | 提供 tokenizer、原文或记忆、方法专属标记；维护分词拼接、位置编码、模型窗口和实际计算 |

方法通过 `qa_input_format` 声明是否套用聊天模板，并将返回值保存为 `settings.input_format`：

- `use_chat_template=True`：使用 tokenizer 的聊天模板，记录实际 `chat_template`、`add_generation_prompt` 和 `enable_thinking`；回答起始由模板提供。
- `use_chat_template=False`：按 `<QA 系统提示><system_separator><内容><answer_prefix>` 拼接，分隔符默认 `\n\n`，答案前缀默认 `Answer:`；二者均由方法设置。

完整文本可直接编码：

```python
from latent_working_memory.prompts import QAInputBuilder, qa_input_format

input_format = qa_input_format(tokenizer, use_chat_template=True)
builder = QAInputBuilder(tokenizer, input_format)
input_ids = builder.encode(sample.prompt)
```

使用软记忆时，`builder.parts()` 提供内容前后的 `prefix` 和 `suffix`，方法在中间拼接记忆与问题；`template_prefix` 和 `system_text` 可用于分别编码模板前缀与系统提示。正文参与方法的压缩或状态更新，QA 系统提示和问题用于 reader 作答。方法专属标记及分词选项随方法配置保存。

### 会话适配器

[公共批处理模块](../documents.py) 定义 `DocumentAdapter`，方法实现以下两个接口：

| 接口 | 输入与返回约定 |
|---|---|
| `prepare_contexts(documents)` | 批量准备本批首次遇到的正文。每项为 `(正文, 待答问题列表)`，按顺序返回各自的 `PreparedConversation` |
| `answer_batch(states, questions, complete)` | 接收与每题一一对应的正文状态和一批 `PreparedQuestion`；每完成一题，调用一次 `complete(question, answer)`，其中 `answer["prediction"]` 为回答字符串 |

返回对象的字段如下：

| 对象 | 字段 |
|---|---|
| `PreparedConversation` | `state`：共享正文状态；`questions`：全部待答题的准备结果；`statistics`：可选对话统计字典，默认空 |
| `PreparedQuestion` | `sample`：原始样本；`inputs`：方法编码的问题输入 |

公共层按完整正文分组，优先从当前组取题，不足一批时从下一组补齐。同一正文跨批只准备一次，多篇新正文可在同一批中准备。各题使用自己的位置、生成预算和停止条件，解码完成即保存。

TTT 方法通过 [CausalLoCoMo](causal.py) 的 `fork_cache` 从各题对应的正文状态选择行，构造独立的生成分支。

### 评估入口

方法在可导入模块中提供顶层函数 `load_adapter(settings)`，加载模型并返回会话适配器。`settings` 为可写入 JSON 的字典，记录权重身份、`input_format` 及影响回答的参数。

Python 入口添加方法参数和公共参数，再调用 `run_evaluation`。以下示例中的 `my_method.locomo_adapter` 由接入方实现，`configuration(args)` 返回方法配置：

```python
import argparse
from pathlib import Path

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from my_method.locomo_adapter import configuration, load_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    add_evaluation_arguments(parser, output_root="artifacts/my-method")
    args = parser.parse_args()
    run_evaluation(args, "my-method", configuration(args), load_adapter,
                   output_name=args.checkpoint.name)


if __name__ == "__main__":
    main()
```

`run_evaluation` 负责数据读取、GPU 分片、worker 启动、评分、保存和续评。按模型或 checkpoint 组织结果的方法调用 `add_evaluation_arguments(..., checkpoint_layout=True)`，并向 `run_evaluation` 传入 `checkpoint_name` 和评估设置前缀 `output_name`。Full-context 使用模型目录名作为 `checkpoint_name`，`output_name=""` 时默认结果目录名仅含任务名。方法自己的 `eval_locomo.sh` 选择运行环境，将项目根目录和 `src/` 加入 `PYTHONPATH`，并调用 Python 入口。

已有调度流程可直接复用 `shard_conversations(samples, workers)` 与 `evaluate_conversations(samples, adapter, batch_size, commit)`。前者按整段对话分片，后者完成一题即调用 `commit(sample, answer, statistics)`；调用方按下文的产物规范评分和保存。

## 运行参数

以下约定适用于公共调度器 `run_evaluation`。调度器按对话分配任务，每张参与评估的 GPU 使用独立进程。

| 参数 | 含义与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |
| `--output-root` | 评估产物根目录，默认值由方法入口提供 |
| `--output-id` | 结果目录名；Full-context 默认 `locomo`，ICAE、AutoCompressors 和 SDCC 默认 `<压缩设置>_locomo`，其他方法默认由方法名称与时间戳组成 |
| `--batch-size-per-gpu` | 每卡批处理数，默认 1 |
| `--resume` | 沿用原结果目录和评估配置续评 |

ICAE、AutoCompressors 和 SDCC 写入 `<output-root>/eval/<checkpoint_name>/<output-id>/`，默认 ID 为 `<压缩设置>_locomo`。Full-context 写入 `<output-root>/eval/<model_name>/<output-id>/`，`model_name` 取 `--model-path` 指向的目录名，默认 ID 为 `locomo`。显式指定 `--output-id` 时以该值作为完整目录名；在相同命令后加 `--resume` 即可续评，自定义 ID 须沿用原值。其他方法写入 `<output-root>/eval/<output-id>/locomo/`，续评时指定原 ID。

## 产物规范

### 公共结果字段

所有接入方式共用 `summary.json` 和 `details.jsonl` 的以下字段。公共调度器自动生成；自有调度使用 [data_identity](../protocol.py) 记录数据身份，使用 [score_sample 与 summarize_scores](../scoring.py) 计算逐题和汇总 F1，三个函数的 benchmark 参数均为 `"locomo"`。

`summary.json` 保存一个 JSON 对象：

| 字段 | 类型与含义 |
|---|---|
| `data` | `data_identity(samples, data_dir, "locomo")` 的完整返回值：数据目录 `directory`、源仓库提交 `source_commit`、题数 `samples`、样本内容指纹 `fingerprint` |
| `metrics.metric` | 字符串，固定为 `"f1"` |
| `metrics.score` | 数值，全部题目的平均 F1 |
| `metrics.samples` | 整数，已评题数；完整评估与 `data.samples` 一致 |
| `metrics.groups` | 按类别排序的数组；每项包含整数 `category`、字符串 `category_name`、整数 `samples` 和数值 `score`，分别表示题型、名称、题数和该类平均 F1 |

`details.jsonl` 按数据原顺序保存，每行一个 JSON 对象，每题一条：

| 字段 | 类型与来源 |
|---|---|
| `id` | 字符串，原样保留 `sample.id`，作为跨运行对齐题目的唯一标识 |
| `metadata` | 原样保留 `sample.metadata`；展示使用 `question`、`category` 和 `conversation_id` |
| `references` | 字符串数组，取自 `list(sample.references)` |
| `prediction` | 字符串，方法生成的回答 |
| `score` | 数值，`score_sample("locomo", sample, prediction)` 返回的逐题 F1 |

### 公共调度产物

公共调度器在结果目录中保存以下文件：

| 文件 | 内容 |
|---|---|
| `evaluation_config.json` | `method`、`benchmark`、`settings`、`adapter`、`data`、`protocol`；分别记录方法标识、任务、方法配置、加载函数、数据身份和评估协议 |
| `summary.json` | 评估配置及标准 `metrics`，可附方法统计和汇总指标 |
| `details.jsonl` | 完整逐题结果 |
| `progress/worker-*.jsonl` | 已完成的逐题记录，作为续评依据 |
| `progress/worker-*.log` | worker 日志 |
| `work.json`、`execution.json` | 本次任务分配，以及 Git commit、GPU、批量、耗时和执行状态 |

输入格式记录于 `settings.input_format`，QA 系统提示记录于 `protocol.system_prompt`。正文状态在评估进程内复用，续评依据为已完成的逐题记录。

### 方法指标扩展

方法按需记录与自身评估有关的指标，值均可写入 JSON：

- **逐题统计**：随 `answer` 返回，写入对应的逐题记录。
- **对话统计**：通过 `PreparedConversation.statistics` 返回，写入逐题记录的 `context`，并按对话 ID 去重后保存为 `summary.conversations`。统计只依赖正文和方法配置。
- **汇总指标**：向 `run_evaluation` 提供 `summarize_method(records, conversations)` 回调，接收全部逐题记录和按对话 ID 组织的统计，返回字典写入 `summary.method_metrics`。方法定义指标的单位与汇总方式。

可选统计字段在提供内容时保存；续评结束后使用全部记录重新汇总。

## 结果展示

### 跨运行比较

[公共看板脚本](scripts/dashboard.sh) 使用当前环境的 Python 启动 Streamlit。启动前激活已安装看板依赖的环境。`--run` 后按“展示标签、LoCoMo 结果目录”成对传参，可连续指定多组：

```bash
source .venv/bin/activate
bash benchmarks/locomo/scripts/dashboard.sh \
  --run 方法A-运行1 artifacts/method-a/eval/run-1/locomo \
        方法A-运行2 artifacts/method-a/eval/run-2/locomo \
        方法B-运行1 artifacts/method-b/eval/run-1/locomo
```

每个目录直接包含 `summary.json` 和 `details.jsonl`，展示标签各不相同。侧栏选择运行，页面展示总体与各题型 F1，并按题型、对话筛选同题回答、参考答案和逐题 F1。

参与比较的运行使用相同数据指纹、题目和参考答案。看板读取已保存的分数，支持不同方法、模型及输入设置的运行并列展示。

### 方法看板接入

[dashboard.py](dashboard.py) 提供可复用的展示组件，方法维护自己的页面和启动脚本。按运行 ID 保存结果的方法使用以下接口：

| 接口 | 用法 |
|---|---|
| `run_dashboard(method, title, render_results, argv=None, qasper=False)` | `--output-dir` 指向运行目录，选择一个已评分任务。支持 LoCoMo、LongMemEval-S 和 Oracle，传 `qasper=True` 可启用 QASPER。选中 LoCoMo 后调用 `render_results(结果目录, summary)` |
| `show_scores(metrics, extra_metrics=())` | 展示总体及分题型 F1 和题数；可通过 `(标签, 已格式化数值)` 序列追加方法指标卡 |
| `show_answers(directory, metrics)` | `directory` 指向含 `details.jsonl` 的 LoCoMo 结果目录，展示筛选后的逐题回答 |

ICAE、AutoCompressors 和 SDCC 使用 [checkpoint 看板入口](../checkpoint_dashboard.py) `benchmarks.checkpoint_dashboard.run_dashboard(method, title, render_memory, argv=None)`。`--output-dir` 接收一个或多个 `<output-root>/eval/<checkpoint_name>/`，可在一次参数后连续列出目录，也可重复该参数。例如，可在同一看板比较以下两个 checkpoint：

```bash
bash reproductions/icae/scripts/dashboard.sh --output-dir \
  artifacts/reproductions/icae/eval/mistral_7b_pretrained_icae \
  artifacts/reproductions/icae/eval/mistral_7b_ft_icae
```

入口按 `summary.json` 中的 benchmark 及其设置或划分组织任务。每个任务直接列出两级复选框，默认全部选中。每个 checkpoint 下始终列出各运行的 `output_id` 和已选数量。子项可单独勾选。全选时点击 checkpoint 可清空该组，只选部分或未选时点击可全选。LoCoMo 柱图按总体及各题型 F1 分组，顶部列出所选运行的完整图例，柱顶显示百分比分数。

各任务先按 checkpoint 名自然排序，同一 checkpoint 内按压缩参数数值升序排列，同参数的运行按 `output_id` 自然排序。每个 checkpoint 使用一种色系。看板按全部输入目录、全部任务中的不同压缩设置预先分配深浅，参数从小到大对应从深到浅的等距色阶，只有一种设置时使用中等明度。同一 checkpoint 和设置跨任务同色，筛选时颜色保持固定。评分详细表、`render_memory(runs)` 展示的记忆统计和底部逐题对照同步更新。清空某任务的选择后，该任务显示选择提示。

按运行 ID 保存结果的方法，其 `render_results` 回调依次调用 `show_scores(summary["metrics"])`、方法专属展示和 `show_answers(directory, summary["metrics"])`。

方法环境包含 Streamlit 等看板依赖，启动脚本使用该环境的 Python。按运行 ID 保存结果的方法在项目根目录运行，例如：

```bash
PYTHONPATH=.:src python -m streamlit run my_method/dashboard.py -- \
  --output-dir artifacts/my-method/eval/run-1
```
