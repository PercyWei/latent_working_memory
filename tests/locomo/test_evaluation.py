"""验证公共调度通过方法提供的工厂运行独立 worker，并按题续评。"""

import argparse
from datetime import datetime
from importlib import import_module
import json
from pathlib import Path

import pytest

from benchmarks.locomo.evaluate import add_evaluation_arguments, run_evaluation
from benchmarks.locomo.records import summarize


def test_method_factory_workers_and_resume(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}:{root}:{root / 'src'}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    (tmp_path / "locomo_test_method.py").write_text('''
import json
import os
from pathlib import Path
from benchmarks.documents import PreparedConversation, PreparedQuestion

class Adapter:
    def __init__(self, settings):
        self.trace = Path(settings["trace"])

    def prepare_contexts(self, documents):
        prepared = []
        for history, questions in documents:
            cid = questions[0].metadata["conversation_id"]
            with self.trace.open("a") as stream:
                stream.write(json.dumps([cid, os.environ["CUDA_VISIBLE_DEVICES"]]) + "\\n")
            prepared.append(PreparedConversation(cid, [PreparedQuestion(q, None) for q in questions],
                                                  {"context_tokens": len(history)}))
        return prepared

    def answer_batch(self, states, questions, complete):
        for state, question in zip(states, questions, strict=True):
            assert question.sample.metadata["conversation_id"] == state
            complete(question, {"prediction": "red", "decode_batch_size": len(questions)})

def load_adapter(settings):
    return Adapter(settings)

def summarize_metrics(records, conversations):
    return {"total_context_tokens": sum(stats["context_tokens"] for stats in conversations.values())}
''')
    method = import_module("locomo_test_method")
    factory = method.load_adapter
    data = tmp_path / "data"
    data.mkdir()
    (data / "locomo10.json").write_text(json.dumps([{
        "sample_id": cid,
        "conversation": {
            "speaker_a": "Alice", "speaker_b": "Bob",
            "session_1_date_time": "1 January 2026",
            "session_1": [{"speaker": "Alice", "text": f"My {cid} bike and hat are red."}],
        },
        "qa": [{"category": 4, "question": f"What color is Alice's {item}?", "answer": "red"}
               for item in ("bike", "hat")],
    } for cid in ("first", "second")]))
    parser = argparse.ArgumentParser()
    add_evaluation_arguments(parser, tmp_path / "results")
    args = parser.parse_args([
        "--benchmark-data", str(data), "--batch-size-per-gpu", "2", "--no-llm-as-judge",
    ])
    trace = tmp_path / "prepared.jsonl"
    settings = {"trace": str(trace), "input_format": {"use_chat_template": False, "answer_prefix": "\nAnswer:"}}
    summary = run_evaluation(args, "local-test", settings, factory,
                             output_name="tiny-checkpoint", summarize_method=method.summarize_metrics)
    assert args.output_id.startswith("tiny-checkpoint_")
    datetime.strptime(args.output_id.removeprefix("tiny-checkpoint_"), "%Y%m%d-%H%M")
    original_id = args.output_id
    directory = args.output_root / "eval" / original_id / "locomo"
    rows = [json.loads(line) for line in (directory / "details.jsonl").read_text().splitlines()]
    assert summary["metrics"]["score"] == 1
    assert len(rows) == 4
    assert all(row["decode_batch_size"] == 2 for row in rows)
    assert sorted(json.loads(line) for line in trace.read_text().splitlines()) == [
        ["first", "0"], ["second", "1"],
    ]
    assert summary["adapter"] == {"module": "locomo_test_method", "factory": "load_adapter"}
    assert summary["method_metrics"] == {"total_context_tokens": sum(
        stats["context_tokens"] for stats in summary["conversations"].values()
    )}
    assert summary["method_metrics"]["total_context_tokens"] * 2 == sum(
        row["context"]["context_tokens"] for row in rows
    )
    # 方法可以只提供回答；公共汇总无需 token 数或其他方法字段。
    plain = summarize({}, [{key: value for key, value in row.items() if key != "context"} for row in rows])
    assert plain == {"metrics": summary["metrics"]}

    # 模拟最后一题写入中断；保留另一题，改变 GPU 数量和 batch 后补齐。
    shard = directory / "progress/worker-1.jsonl"
    first_line = shard.read_text().splitlines()[0]
    shard.write_text(first_line + '\n{"id":')
    args.resume = True
    args.batch_size_per_gpu = 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    resumed = run_evaluation(args, "local-test", settings, factory,
                             output_name="unused-default", summarize_method=method.summarize_metrics)
    assert args.output_id == original_id
    assert resumed["metrics"] == summary["metrics"]
    assert resumed["method_metrics"] == summary["method_metrics"]
    assert resumed["conversations"] == summary["conversations"]
    after = [json.loads(line) for line in (directory / "details.jsonl").read_text().splitlines()]
    assert [(r["id"], r["prediction"]) for r in after] == [(r["id"], r["prediction"]) for r in rows]
    assert sum(row["decode_batch_size"] == 1 for row in after) == 1
    assert len(trace.read_text().splitlines()) == 3
    assert json.loads(trace.read_text().splitlines()[-1]) == ["second", "0"]
    assert shard.read_text() == first_line + "\n"

    # 全部完成后不再加载适配器；协议不一致仍须拒绝。
    finished = run_evaluation(args, "local-test", settings, factory,
                              output_name="tiny-checkpoint", summarize_method=method.summarize_metrics)
    assert finished["method_metrics"] == summary["method_metrics"]
    assert len(trace.read_text().splitlines()) == 3
    with pytest.raises(ValueError, match="settings differ"):
        run_evaluation(args, "local-test", {
            **settings, "input_format": {**settings["input_format"], "answer_prefix": "\nShort answer:"},
        }, factory, output_name="tiny-checkpoint")
    args.output_id = None
    with pytest.raises(ValueError, match="--resume requires the original --output-id"):
        run_evaluation(args, "local-test", settings, factory, output_name="tiny-checkpoint")
