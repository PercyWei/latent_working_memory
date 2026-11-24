# 20261124_公共 LoCoMo 评估

创建时间：20261124 15:43:38 Asia/Shanghai

最后修订时间：20261124 16:42:46 Asia/Shanghai

本目录提供方法无关的 LoCoMo 评估接口。固定评估全部对话的完整历史及类别 1–4，共 10 段对话、1,540 道题；数据准备见[公共 benchmark 说明](../README.md)。

## 公共计算接口

[engine.py](engine.py) 定义 `ConversationAdapter`、`PreparedConversation` 和 `PreparedQuestion`。

| 接口 | 职责 |
|---|---|
| `shard_conversations(samples, workers)` | 将完整对话分配给各 worker，按待答题数均衡负载 |
| `evaluate_conversations(samples, adapter, batch_size, commit)` | 逐段准备正文，同对话按 `batch_key` 组批，完成一题即调用 `commit(sample, answer, statistics)` |
| `adapter.prepare_context(history, questions)` | 返回只读正文状态、编码后的问题及对话统计 |
| `adapter.answer_batch(state, questions, complete)` | 为各题创建独立状态分支，完成一题即调用 `complete(question, answer)` |

公共层规定以下行为：

- **会话调度**：同一对话的待答问题分配给同一 worker。正文只准备一次，全部答完后释放状态。
- **批量解码**：同一对话内按 `batch_key` 组批，每批最多包含 `batch_size` 道问题，各题复用正文状态。
- **问题隔离**：每道题从相同正文状态独立作答，不继承其他问题或答案。适配器须保证共享状态只读。
- **模型适配**：适配器负责前缀 KV、压缩记忆等正文状态的实现，以及 chat template、位置编码和模型窗口处理。
- **评估协议**：采用贪心解码，最多生成 50 tokens。F1 按题平均，正文 token 数与压缩后记忆 token 数以对话为单位统计。
- **断点续评**：仅调度未完成的问题，部分完成的对话重新准备正文。缓存仅在当前 worker 生命周期内复用。

## 使用方法

各方法在自己的目录维护评估脚本和模型适配，按以下步骤接入：

1. **实现会话适配器。** `prepare_context` 返回 `PreparedConversation`，每道题对应一个 `PreparedQuestion`；`batch_key` 相同的题目可一起解码。`answer_batch` 独立回答各题，通过 `complete(question, answer)` 提交结果，`answer` 至少包含字符串 `prediction`。
2. **提供模型加载函数。** 在可导入模块中定义顶层函数 `load_adapter(settings)`，加载模型并返回适配器。`settings` 使用可写入 JSON 的配置，记录权重身份及影响回答的参数。
3. **编写评估入口。** 添加方法参数和公共参数，再调用 `run_evaluation`。该接口负责数据读取、GPU 分片、worker 启动、评分、结果保存和续评。

下面以占位模块 `my_method.locomo_adapter` 示意；其中 `add_arguments`、`configuration` 和 `load_adapter` 均由接入方实现。

```python
import argparse

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from my_method.locomo_adapter import add_arguments, configuration, load_adapter


def main():
    parser = argparse.ArgumentParser()
    add_evaluation_arguments(parser, output_root="artifacts/my-method")
    add_arguments(parser)
    args = parser.parse_args()
    run_evaluation(args, "my-method", configuration(args), load_adapter)


if __name__ == "__main__":
    main()
```

方法自己的 `eval_locomo.sh` 负责选择环境并调用上述 Python 入口；`CUDA_VISIBLE_DEVICES` 指定 GPU，调度器为每张卡启动独立进程，无需 `torchrun`。

已有评估流程也可以直接调用 `shard_conversations` 和 `evaluate_conversations`，自行管理模型加载、结果保存和续评。此时 `commit` 接收每题回答与会话统计，评分可调用 `benchmarks.scoring.score_sample`。

## 独立评估调度

[evaluate.py](evaluate.py) 保存评估计划和加载函数信息，[worker.py](worker.py) 在当前运行环境中导入 `load_adapter`。以下参数和产物适用于 `run_evaluation`。

| 公共参数 | 默认值与约定 |
|---|---|
| `--resume` | 指定原 ID、数据和模型参数，补齐缺失问题；可调整 GPU 数量和批量 |
| `--benchmark-data` | 含 `locomo10.json` 的目录，默认 `data/benchmarks/locomo` |
| `--output-root` | 由方法入口指定默认目录 |
| `--output-id` | 默认 `<method>_<YYYYMMDD-HHMM>` |
| `--batch-size-per-gpu` | 每卡评估批量大小，默认 1 |

结果位于 `<output-root>/eval/<output-id>/locomo/`：

| 文件 | 内容 |
|---|---|
| `evaluation_config.json` | 方法配置、加载函数、模型与数据身份、评分协议 |
| `progress/worker-*.jsonl` | 逐题结果及续评依据，中断写入的末行在续评时截去 |
| `progress/worker-*.log` | 各 worker 的加载和进度日志 |
| `work.json` | 本次待评题目分配和 batch 设置 |
| `execution.json` | 本次执行的 Git commit、GPU、耗时和状态 |
| `details.jsonl` | 按数据原顺序汇总的逐题结果 |
| `summary.json` | `metrics` 中的总体与分类 F1，`conversations` 中的会话统计 |

## 关键验证

[会话计算检查](../../tests/locomo/test_shared_context.py) 覆盖独立计算与缓存结果的一致性、问题顺序、批量、缓存只读和断点记录；[调度检查](../../tests/locomo/test_evaluation.py) 覆盖加载函数、多进程分片及调整批量后的续评。会话计算检查可通过 `LOCOMO_TEST_DEVICE=cuda` 在 GPU 执行。
