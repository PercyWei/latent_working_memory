# 20261124_公共 LoCoMo 评估

创建时间：20261124 15:43:38 Asia/Shanghai

最后修订时间：20261125 10:58:32 Asia/Shanghai

本目录提供方法无关的 LoCoMo 评估接口。固定评估全部对话的完整历史及类别 1–4，共 10 段对话、1,540 道题；数据准备见[公共 benchmark 说明](../README.md)。

## 公共计算接口

[engine.py](engine.py) 定义 `ConversationAdapter`、`PreparedConversation` 和 `PreparedQuestion`。

| 接口 | 职责 |
|---|---|
| `shard_conversations(samples, workers)` | 将完整对话分配给各 worker，按待答题数均衡负载 |
| `evaluate_conversations(samples, adapter, batch_size, commit)` | 逐段准备正文，同对话按 `batch_key` 组批，完成一题即调用 `commit(sample, answer, statistics)` |
| `adapter.prepare_context(history, questions)` | 返回只读正文状态、编码后的问题及可选对话统计 |
| `adapter.answer_batch(state, questions, complete)` | 为各题创建独立状态分支，完成一题即调用 `complete(question, answer)` |

公共层规定以下行为：

- **会话调度**：同一对话的待答问题分配给同一 worker。正文只准备一次，全部答完后释放状态。
- **批量解码**：同一对话内按 `batch_key` 组批，每批最多包含 `batch_size` 道问题，各题复用正文状态。
- **问题隔离**：每道题从相同正文状态独立作答，不继承其他问题或答案。适配器须保证共享状态只读。
- **模型适配**：适配器负责正文状态、记忆与文本的拼接、位置编码及模型窗口；声明是否套用聊天模板，包装逻辑复用公共输入构造。
- **评估协议**：采用贪心解码，最多生成 50 tokens。按官方规则计算逐题 F1，汇总总体与各题型的按题平均 F1 和题数。
- **指标扩展**：额外指标由方法定义，公共层只传递和保存；未提供时不记录。
- **断点续评**：仅调度未完成的问题，部分完成的对话重新准备正文。缓存仅在当前 worker 生命周期内复用。

## 使用方法

各方法在自己的目录维护评估脚本和模型适配，按以下步骤接入：

1. **实现会话适配器。** `prepare_context` 返回 `PreparedConversation`，每道题对应一个 `PreparedQuestion`；`batch_key` 相同的题目可一起解码。`answer_batch` 独立回答各题，通过 `complete(question, answer)` 提交结果，`answer` 至少包含字符串 `prediction`。
2. **提供模型加载函数。** 在可导入模块中定义顶层函数 `load_adapter(settings)`，加载模型并返回适配器。`settings` 使用可写入 JSON 的配置，记录权重身份、`input_format` 及影响回答的参数。
3. **编写评估入口。** 添加方法参数和公共参数，再调用 `run_evaluation`。该接口负责数据读取、GPU 分片、worker 启动、评分、结果保存和续评。

下面以占位模块 `my_method.locomo_adapter` 示意；其中 `add_arguments`、`configuration` 和 `load_adapter` 均由接入方实现，示例中的 `add_arguments` 包含 `Path` 类型的 `--checkpoint` 参数。

```python
import argparse

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from my_method.locomo_adapter import add_arguments, configuration, load_adapter


def main():
    parser = argparse.ArgumentParser()
    add_evaluation_arguments(parser, output_root="artifacts/my-method")
    add_arguments(parser)
    args = parser.parse_args()
    run_evaluation(args, "my-method", configuration(args), load_adapter,
                   output_name=args.checkpoint.name)


if __name__ == "__main__":
    main()
```

方法自己的 `eval_locomo.sh` 负责选择环境并调用上述 Python 入口；`CUDA_VISIBLE_DEVICES` 指定 GPU，调度器为每张卡启动独立进程，无需 `torchrun`。

方法通过 `output_name` 指定默认运行名称；未传 `--output-id` 时，公共层追加上海时间生成 `<output_name>_<YYYYMMDD-HHMM>`。显式指定的 ID 原样使用，续评必须指定原 ID。

已有评估流程也可以直接调用 `shard_conversations` 和 `evaluate_conversations`，自行管理模型加载、结果保存和续评。此时 `commit` 接收每题回答与会话统计，评分可调用 `benchmarks.scoring.score_sample`。

对于通过完整前缀缓存续写的模型，[causal.py](causal.py) 提供 `CausalLoCoMo`：接入方传入模型、tokenizer、`fork_cache(prefix, batch_size)` 和输入配置。它调用公共输入构造，预填共享前缀后逐 token 输入问题及生成答案；缓存分支必须包含所有可变状态。该路径沿用模型的缓存更新规则，不冻结问答阶段的递归状态。

### 输入格式

输入由以下模块共同构造：

| 维护位置 | 职责与数据 |
|---|---|
| LoCoMo 数据层 | 提供正文、简答要求和问题；参考答案仅用于评分 |
| [公共提示模块](../../src/latent_working_memory/prompts.py) | 维护 QA 系统提示，构造完整输入或内容前后的模板片段 |
| 方法适配层 | 加载 tokenizer，声明包装设置，提供原文或记忆及方法专属标记；维护分词拼接边界和模型计算 |

`settings.input_format` 使用 `use_chat_template` 声明是否套用聊天模板，输入构造和配置记录共用同一份定义：

- 启用时保存 tokenizer 的实际 `chat_template`、`add_generation_prompt` 和 `enable_thinking`；回答起始由模板提供。
- 未启用时保存 `system_separator` 和 `answer_prefix`；答案前缀默认 `Answer:`，也可显式设为空或包含换行的文本。
- 方法附加标记及实际使用的特殊 token、分词选项由方法记录，不单列输入类型。

`qa_input_format` 创建配置，`QAInputBuilder` 负责渲染。例如：

```python
from latent_working_memory.prompts import QAInputBuilder, qa_input_format

input_format = qa_input_format(tokenizer, use_chat_template=True)
builder = QAInputBuilder(tokenizer, input_format)
input_ids = builder.encode(history + question_prompt)
```

插入软记忆时使用 `builder.parts()` 获取 `prefix` 和 `suffix`，由方法在中间拼接记忆与问题；需要分别编码角色前缀和系统提示时，可读取 `template_prefix` 和 `system_text`。模板包装不合并原有分词边界，也不修改方法提供的记忆。

系统提示由 `protocol.system_prompt` 统一记录，模型与阶段沿用方法配置。输入格式随评估计划保存，续评时严格核对；逐题结果不重复保存模板。公共调度不根据方法名或 tokenizer 是否具有聊天模板自动选择格式。

### 额外指标

方法可在 `PreparedConversation.statistics` 中提供对话统计，在 `answer` 中提供逐题统计；值须可写入 JSON。对话统计应只依赖正文和方法设置，不随待答题数、批量或续评改变。

需要汇总时，向 `run_evaluation` 传入 `summarize_method(records, conversations)` 回调：`records` 包含全部逐题结果，`conversations` 按对话 ID 保存去重后的统计。回调返回的非空字典写入 `summary.json` 的 `method_metrics`，与标准 F1 的 `metrics` 分开。方法自行规定单位与平均方式；续评完成后从全部记录重新汇总，无需重新加载模型。未提供对话统计或汇总指标时，省略对应字段。

## 独立评估调度

[evaluate.py](evaluate.py) 保存评估计划和加载函数信息，[worker.py](worker.py) 在当前运行环境中导入 `load_adapter`。以下参数和产物适用于 `run_evaluation`。

| 公共参数 | 默认值与约定 |
|---|---|
| `--resume` | 指定原 ID、数据和模型参数，补齐缺失问题；可调整 GPU 数量和批量 |
| `--benchmark-data` | 含 `locomo10.json` 的目录，默认 `data/benchmarks/locomo` |
| `--output-root` | 由方法入口指定默认目录 |
| `--output-id` | 默认名称由方法入口提供，再追加上海时间；具体规则见方法说明 |
| `--batch-size-per-gpu` | 每卡评估批量大小，默认 1 |

结果位于 `<output-root>/eval/<output-id>/locomo/`：

| 文件 | 内容 |
|---|---|
| `evaluation_config.json` | 方法配置（含 `settings.input_format`）、加载函数、模型与数据身份、评分协议 |
| `progress/worker-*.jsonl` | 逐题结果及续评依据，中断写入的末行在续评时截去 |
| `progress/worker-*.log` | 各 worker 的加载和进度日志 |
| `work.json` | 本次待评题目分配和 batch 设置 |
| `execution.json` | 本次执行的 Git commit、GPU、耗时和状态 |
| `details.jsonl` | 按数据原顺序汇总的逐题结果 |
| `summary.json` | `metrics` 中的总体与分类 F1、题数；可选 `conversations` 和 `method_metrics` 保存方法统计及其汇总 |

## 看板接口

[dashboard.py](dashboard.py) 提供方法无关的 Streamlit 组件。各方法维护自己的 `dashboard.py` 和启动脚本，读取已完成的本地评估结果。

| 接口 | 职责 |
|---|---|
| `run_dashboard(method, title, render_results, argv=None)` | 解析 `--output-dir`，读取其下 `locomo/summary.json`，校验方法与任务，再调用方法的渲染函数 |
| `show_scores(metrics, extra_metrics=())` | 展示总体及各题型 F1、题数；可追加方法提供的指标卡 |
| `show_answers(directory, metrics)` | 读取 `details.jsonl`，按题型与对话筛选，展示问题、回答、参考答案和逐题 F1 |

`metrics` 包含标准评分汇总的 `score`、`samples` 和 `groups`；`directory` 指向 `locomo/`。对话选项直接取自逐题记录，不依赖额外的对话统计。`extra_metrics` 为已格式化的 `(标签, 数值)` 序列，计算与单位由方法维护；专属表格和图表也由方法自行渲染。公共层不重算分数或写入结果。

方法入口示例：

```python
from benchmarks.locomo.dashboard import run_dashboard, show_answers, show_scores


def render_results(directory, summary):
    show_scores(summary["metrics"])
    # 在此添加本方法需要的额外展示。
    show_answers(directory, summary["metrics"])


if __name__ == "__main__":
    run_dashboard("my-method", "方法名称", render_results)
```

各方法的启动脚本使用项目根环境执行 `uv run --frozen streamlit run <方法看板路径> -- --output-dir <运行目录>`，并将项目根目录加入 `PYTHONPATH`。

## 关键验证

[输入构造检查](../../tests/locomo/test_input_construction.py) 对照原始模板与文本拼接，核对 token、记忆插入位置和答案起始；[会话计算检查](../../tests/locomo/test_shared_context.py) 覆盖独立计算与缓存结果的一致性、问题顺序、批量、缓存只读和断点记录；[调度检查](../../tests/locomo/test_evaluation.py) 覆盖加载函数、多进程分片及调整批量后的续评。会话计算检查可通过 `LOCOMO_TEST_DEVICE=cuda` 在 GPU 执行。
