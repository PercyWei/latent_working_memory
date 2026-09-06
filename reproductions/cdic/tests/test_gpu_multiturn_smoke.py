from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from cdic_repro.config import RetrievalConfig
from cdic_repro.engine import CdicInferenceEngine
from cdic_repro.icae_adapter import (
    IcaeV1AdapterConfig,
    IcaeV1InferenceAdapter,
    torch_cosine_similarity,
)
from cdic_repro.training_checkpoint import load_model_from_training_checkpoint
from cdic_repro.writeback import WriteAction


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _required_resource(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        pytest.fail(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{field} does not exist: {path}")
    return path


def _load_gpu_config(pytestconfig: pytest.Config) -> tuple[Path, dict[str, object]]:
    config_value = pytestconfig.getoption("--cdic-gpu-config")
    if config_value is None:
        pytest.skip("pass --cdic-gpu-config to run the real multi-turn GPU smoke test")
    config_path = Path(config_value)
    if not config_path.is_file():
        pytest.fail(f"GPU smoke configuration does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        pytest.fail("GPU smoke configuration must be a JSON object")
    return config_path, config


def _assert_transition(output: object, *, memory_size_before: int) -> None:
    trace = output.trace  # type: ignore[attr-defined]
    action = trace.write_back.action

    if memory_size_before == 0:
        assert action is WriteAction.INITIALIZE
        assert trace.retrieval.peak_score is None
        assert trace.retrieval.selected_state_ids == ()
        assert len(trace.write_back.memory_after) == 1
        return

    assert trace.retrieval.peak_score is not None
    assert math.isfinite(trace.retrieval.peak_score)
    assert trace.retrieval.selected_state_ids
    assert trace.retrieval.used_fallback is not trace.retrieval.on_topic
    for score in trace.retrieval.scores:
        assert math.isfinite(score.raw_similarity)
        assert math.isfinite(score.decay_weight)
        assert math.isfinite(score.score)
        assert score.recency_turns >= 0

    if trace.retrieval.on_topic:
        assert action is WriteAction.REPLACE
        assert trace.write_back.replaced_state_id == trace.retrieval.best_state_id
        assert len(trace.write_back.memory_after) == memory_size_before
    else:
        assert action is WriteAction.INSERT
        assert trace.write_back.replaced_state_id is None
        assert len(trace.write_back.memory_after) == memory_size_before + 1


def test_real_icae_multiturn_state_machine(pytestconfig: pytest.Config) -> None:
    config_path, config = _load_gpu_config(pytestconfig)
    model_path = _required_resource(config.get("model_path"), "model_path")
    checkpoint_path = _required_resource(config.get("checkpoint_path"), "checkpoint_path")
    training_checkpoint_value = config.get("training_checkpoint_path")
    training_checkpoint_path = (
        _required_resource(training_checkpoint_value, "training_checkpoint_path")
        if training_checkpoint_value is not None
        else None
    )
    artifact_dir = Path(str(config["artifact_dir"]))
    device = str(config.get("device", "cuda:0"))
    max_new_tokens = int(config.get("max_new_tokens", 32))
    threshold = float(config.get("threshold", 0.8))
    decay = float(config.get("decay", 0.05))
    query_records = config.get("queries")
    if not isinstance(query_records, list) or not query_records:
        pytest.fail("queries must be a non-empty JSON list")

    assert torch.cuda.is_available()
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    adapter = IcaeV1InferenceAdapter.load(
        IcaeV1AdapterConfig(
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            device=device,
            max_new_tokens=max_new_tokens,
        )
    )
    training_progress = None
    if training_checkpoint_path is not None:
        training_progress = load_model_from_training_checkpoint(
            training_checkpoint_path,
            model=adapter,
        )
        adapter.model.eval()
    engine = CdicInferenceEngine(
        model=adapter,
        similarity=torch_cosine_similarity,
        retrieval_config=RetrievalConfig(threshold=threshold, decay=decay),
    )

    records: list[dict[str, object]] = []
    for turn, query_record in enumerate(query_records, start=1):
        if not isinstance(query_record, dict) or not isinstance(query_record.get("query"), str):
            pytest.fail(f"invalid query record at index {turn - 1}")
        query = query_record["query"]
        query_id = str(query_record.get("id", f"gpu-smoke-{turn:02d}"))
        expected_substring = query_record.get("expected_response_substring")
        if expected_substring is not None and not isinstance(expected_substring, str):
            pytest.fail(f"expected_response_substring must be a string at index {turn - 1}")
        memory_before = {state.state_id: state for state in engine.memory.states}
        output = engine.step(query, query_id=query_id)

        # The public ICAE checkpoint was not trained for C-DIC's empty-memory first turn.
        # A whitespace-only response is therefore recorded, not treated as an engine failure.
        assert isinstance(output.response, str)
        assert output.response
        assert tuple(output.state.latent.shape) == (128, 4096)
        assert tuple(output.state.retrieval_key.shape) == (4096,)
        assert output.state.latent.dtype is torch.bfloat16
        assert torch.isfinite(output.state.latent).all().item()
        assert torch.isfinite(output.state.retrieval_key).all().item()
        assert not output.state.latent.requires_grad
        assert not output.state.retrieval_key.requires_grad
        _assert_transition(output, memory_size_before=len(memory_before))

        replaced_state_id = output.trace.write_back.replaced_state_id
        if replaced_state_id is not None:
            replaced_state = memory_before[replaced_state_id]
            assert output.state.thread_id == replaced_state.thread_id
            assert output.state.parent_state_id == replaced_state.state_id
            assert output.state.revision == replaced_state.revision + 1

        records.append(
            {
                "turn": turn,
                "query": query,
                "response": output.response,
                "expected_response_substring": expected_substring,
                "expected_response_match": (
                    expected_substring.casefold() in output.response.casefold()
                    if isinstance(expected_substring, str)
                    else None
                ),
                "state_shape": list(output.state.latent.shape),
                "state_dtype": str(output.state.latent.dtype),
                "trace": output.trace.to_dict(),
            }
        )

    torch.cuda.synchronize()
    report = {
        "config_path": str(config_path),
        "initial_checkpoint_path": str(checkpoint_path),
        "training_checkpoint_path": (
            str(training_checkpoint_path) if training_checkpoint_path is not None else None
        ),
        "training_progress": (
            {
                "epoch": training_progress.epoch,
                "next_episode_position": training_progress.next_episode_position,
                "global_step": training_progress.global_step,
            }
            if training_progress is not None
            else None
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": device,
        "device_name": torch.cuda.get_device_name(device),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "final_memory_states": len(engine.memory),
        "turns": records,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))

    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "gpu_smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
