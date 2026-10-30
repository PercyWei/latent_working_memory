"""无需 GPU，验证样本分片、统一评分和跨重试的逐样本恢复。"""

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from latent_working_memory.benchmarks.data import RULER_TASKS
from latent_working_memory.v4_1.config import ExperimentConfig, TrainingConfig
from latent_working_memory.v4_1.data import Episode
from latent_working_memory.v4_1.evaluation import evaluate as evaluation


@pytest.fixture
def evaluation_setup(tmp_path, monkeypatch):
    config = ExperimentConfig(training=TrainingConfig(
        output_dir=str(tmp_path / "train"), tracking="swanlab",
    ))
    checkpoint = tmp_path / "train/checkpoints/global_step_7/state.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({
        "config": asdict(config), "model": {}, "optimizer": {}, "scheduler": {},
        "cursor": {"step": 7}, "rng": [{}], "query_rng": [torch.Generator().get_state()],
        "binding": {"train_documents": [], "dev_documents": []},
    }, checkpoint)
    data = tmp_path / "ruler"
    for length in (32, 64, 128):
        for task in RULER_TASKS:
            path = data / str(length) / task / "test.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "index": 0, "input": f"{length} question", "outputs": ["answer"],
            }) + "\n", encoding="utf-8")
    calls = SimpleNamespace(lengths=[], fail_at=None, loads=0)

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return [int(messages[0]["content"].split()[0]), 4]

        def decode(self, tokens, **kwargs):
            return "answer" if tokens[0] == 3 else "wrong"

    class Model:
        def __init__(self, config):
            calls.loads += 1

        def to(self, device):
            return self

        def load_trainable_state_dict(self, state):
            pass

        def eval(self):
            return self

        def __call__(self, sample, **kwargs):
            count = len(sample.target_ids)
            return {"qa_loss": {1: 1.0, 2: 2.0, 3: 4.0}[count], "target_tokens": count}

        def generate(self, context, question, budget, eos):
            calls.lengths.append(context[0])
            if len(calls.lengths) == calls.fail_at:
                raise RuntimeError("interrupted generation")
            return {
                "token_ids": [3 if context[0] == 64 else 5], "stop_reason": "max_new_tokens",
                "write_events": 1, "inner_before": 1.0, "inner_after": 0.5,
            }

    monkeypatch.setattr(evaluation.AutoTokenizer, "from_pretrained", lambda path: Tokenizer())
    monkeypatch.setattr(evaluation.AutoConfig, "from_pretrained",
                        lambda path: SimpleNamespace(max_position_embeddings=256))
    monkeypatch.setattr(evaluation, "OnlineMemoryLM", Model)
    return SimpleNamespace(checkpoint=checkpoint, data=data, output=tmp_path / "eval", calls=calls, model=Model)


def test_completed_result_reuses_without_model_and_rejects_changed_protocol(evaluation_setup, monkeypatch):
    case = evaluation_setup
    arguments = dict(task="ruler", max_new_tokens=1, lengths=[64], eval_id="comparison")
    summary = evaluation.evaluate(case.checkpoint, case.data, case.output, device="cpu", **arguments)
    assert not (case.checkpoint.parents[2] / "swanlab.json").exists()
    assert summary["training_run"] == {"directory": str(case.checkpoint.parents[2])}
    request = json.loads((case.output / "eval.json").read_text())
    assert "config" not in request and "swanlab" not in request["training_run"]

    def unexpected(*args, **kwargs):
        raise AssertionError("cached evaluation must not load a model or initialize CUDA")

    monkeypatch.setattr(evaluation, "OnlineMemoryLM", unexpected)
    monkeypatch.setattr(evaluation, "initialize_device", unexpected)
    assert evaluation.evaluate(case.checkpoint, case.data, case.output, **arguments) == summary
    assert evaluation.load_cached_evaluation(case.checkpoint, case.data, case.output, **arguments) == summary
    with pytest.raises(ValueError, match="protocol differs"):
        evaluation.evaluate(case.checkpoint, case.data, case.output,
                            **{**arguments, "max_new_tokens": 2})
    path = case.data / "64" / RULER_TASKS[0] / "test.jsonl"
    row = json.loads(path.read_text())
    path.write_text(json.dumps({**row, "outputs": ["changed"]}) + "\n")
    with pytest.raises(ValueError, match="data differs"):
        evaluation.load_cached_evaluation(case.checkpoint, case.data, case.output, **arguments)
    with pytest.raises(ValueError, match="only to RULER"):
        evaluation.evaluate(case.checkpoint, case.data, case.output, task="test", lengths=[64])

    # Test 的 token 加权 NLL 不能通过平均 worker 均值取得。
    episodes = (
        Episode("qa:0", "qa", (64, 4), (5,), (3,)),
        Episode("qa:1", "qa", (128, 4), (5,), (5, 5, 5)),
        Episode("qa:2", "qa", (64, 4), (5,), (3, 3)),
    )
    references = {episode.id: {"question": episode.id, "answer": "answer"} for episode in episodes}
    monkeypatch.setattr(evaluation, "load_episodes", lambda *args, **kwargs: (
        episodes, references, {"raw_document_ids": ["qa"], "episodes": len(episodes)},
    ))
    monkeypatch.setattr(evaluation, "OnlineMemoryLM", case.model)
    monkeypatch.setattr(evaluation, "initialize_device", lambda device: torch.device(device))
    options = dict(task="test", max_new_tokens=1, eval_id="qa")
    serial = evaluation.evaluate(case.checkpoint, case.data, case.output / "serial", device="cpu", **options)
    output = case.output / "sharded"
    work = evaluation.prepare_sample_work(case.checkpoint, case.data, output, num_workers=2, **options)
    assert work["workers"] == [["qa:0", "qa:2"], ["qa:1"]]
    for rank in (1, 0):
        evaluation.evaluate_shard(case.checkpoint, case.data, output, rank, device="cpu", **options)
    merged = evaluation.finalize_sample_work(case.checkpoint, case.data, output, **options)
    assert merged["metrics"] == serial["metrics"]
    assert merged["metrics"]["nll"] == pytest.approx(17 / 6)
    assert merged["metrics"]["em"] == merged["metrics"]["f1"] == pytest.approx(2 / 3)
    records = [json.loads(line) for line in (output / "details.jsonl").read_text().splitlines()]
    assert [record["id"] for record in records] == [episode.id for episode in episodes]
    assert merged["resources"]["seconds"] == pytest.approx(sum(record["seconds"] for record in records))
    assert merged["resources"]["wall_seconds"] >= 0


def test_ruler_extension_only_computes_missing_length_and_merges_scores(evaluation_setup):
    case = evaluation_setup
    options = dict(task="ruler", max_new_tokens=1)
    first = evaluation.evaluate(case.checkpoint, case.data, case.output, device="cpu", lengths=[64], **options)
    original = (case.output / "lengths/64/details.jsonl").read_bytes()
    work = evaluation.prepare_sample_work(case.checkpoint, case.data, case.output,
                                         num_workers=2, lengths=[128, 64], **options)
    assert work["completed_samples"] == work["pending_samples"] == len(RULER_TASKS)
    for rank in (1, 0):
        evaluation.evaluate_shard(case.checkpoint, case.data, case.output, rank,
                                  device="cpu", lengths=[128, 64], **options)
    second = evaluation.finalize_sample_work(case.checkpoint, case.data, case.output, lengths=[128, 64], **options)
    assert first["metrics"]["score"] == 1.0
    assert second["metrics"]["score"] == 0.5
    assert second["metrics"]["samples"] == 2 * len(RULER_TASKS)
    assert second["protocol"]["lengths"] == [64, 128]
    assert case.calls.lengths == [64] * len(RULER_TASKS) + [128] * len(RULER_TASKS)
    assert (case.output / "lengths/64/details.jsonl").read_bytes() == original
    assert json.loads((case.output / "summary.json").read_text()) == second
    assert second["outputs"]["budget_exhausted"] == 2 * len(RULER_TASKS)
    assert not (case.output / "report").exists()
    # 新增更短长度会改变全局位置，已有样本仍应按稳定 ID 复用。
    work = evaluation.prepare_sample_work(case.checkpoint, case.data, case.output,
                                         num_workers=3, lengths=[32, 128, 64], **options)
    assert work["completed_samples"] == 2 * len(RULER_TASKS)
    assert all(sample_id.startswith("ruler/32/") for shard in work["workers"] for sample_id in shard)
    case.calls.lengths.clear()
    for rank in range(len(work["workers"])):
        evaluation.evaluate_shard(case.checkpoint, case.data, case.output, rank,
                                  device="cpu", lengths=[32, 128, 64], **options)
    third = evaluation.finalize_sample_work(case.checkpoint, case.data, case.output,
                                            lengths=[32, 128, 64], **options)
    assert case.calls.lengths == [32] * len(RULER_TASKS)
    assert third["metrics"]["score"] == pytest.approx(1 / 3)
    assert (case.output / "lengths/64/details.jsonl").read_bytes() == original
    with pytest.raises(ValueError, match="only stay the same or expand"):
        evaluation.evaluate(case.checkpoint, case.data, case.output, lengths=[64], **options)
    with pytest.raises(ValueError, match="unique"):
        evaluation.evaluate(case.checkpoint, case.data, case.output, lengths=[64, 64], **options)


def test_interrupted_ruler_length_restarts_without_recomputing_complete_length(evaluation_setup, monkeypatch):
    case = evaluation_setup
    options = dict(task="ruler", max_new_tokens=1, lengths=[64, 128])
    case.calls.fail_at = len(RULER_TASKS) + 3
    with pytest.raises(RuntimeError, match="interrupted generation"):
        evaluation.evaluate(case.checkpoint, case.data, case.output, device="cpu", **options)
    assert (case.output / "lengths/64/summary.json").is_file()
    assert not (case.output / "lengths/128/summary.json").exists()
    assert not (case.output / "summary.json").exists()
    progress = case.output / "progress/worker_0.jsonl"
    with progress.open("ab") as stream:
        stream.write(b'{"record":')
    work = evaluation.prepare_sample_work(case.checkpoint, case.data, case.output, num_workers=3, **options)
    assert work["completed_samples"] == len(RULER_TASKS) + 2
    assert work["pending_samples"] == len(RULER_TASKS) - 2
    assert progress.read_bytes().endswith(b"\n")
    with pytest.raises(evaluation.IncompleteEvaluationError, match="15/26") as incomplete:
        evaluation.finalize_sample_work(case.checkpoint, case.data, case.output, **options)
    assert incomplete.value.completed_samples == 15
    assert incomplete.value.total_samples == 26
    case.calls.lengths.clear()
    case.calls.fail_at = None
    for rank in (2, 0, 1):
        evaluation.evaluate_shard(case.checkpoint, case.data, case.output, rank, device="cpu", **options)
    assert case.calls.lengths == [128] * (len(RULER_TASKS) - 2)
    # 完整坏行不可作为“未提交末行”静默丢弃。
    committed = progress.read_bytes()
    with progress.open("ab") as stream:
        stream.write(b'{bad json}\n')
    with pytest.raises(json.JSONDecodeError):
        evaluation.prepare_sample_work(case.checkpoint, case.data, case.output, num_workers=2, **options)
    assert progress.read_bytes().endswith(b'{bad json}\n')
    progress.write_bytes(committed)
    assert not (case.output / "summary.json").exists()

    def unexpected(*args, **kwargs):
        raise AssertionError("pure merge must not initialize a GPU or load a model")

    monkeypatch.setattr(evaluation, "initialize_device", unexpected)
    monkeypatch.setattr(evaluation, "OnlineMemoryLM", unexpected)
    work = evaluation.prepare_sample_work(case.checkpoint, case.data, case.output, num_workers=2, **options)
    assert work["pending_samples"] == 0 and work["workers"] == []
    summary = evaluation.finalize_sample_work(case.checkpoint, case.data, case.output, **options)
    assert summary["metrics"]["score"] == 0.5
    assert summary["metrics"]["samples"] == 26
    assert not list(case.output.glob("*.tmp"))
