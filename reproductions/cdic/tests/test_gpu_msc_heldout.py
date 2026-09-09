from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest
import torch

from cdic_repro.config import RetrievalConfig, RetrievedStateOrder
from cdic_repro.experiments.checkpoint import load_cdic_model_checkpoint
from cdic_repro.experiments.msc.data import load_msc_episodes, summarize_msc_episodes
from cdic_repro.experiments.msc.evaluate import evaluate_msc_diagnostics
from cdic_repro.icae.adapter import (
    IcaeV1AdapterConfig,
    IcaeV1TrainingAdapter,
    torch_cosine_similarity,
)


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _required_path(config: dict[str, object], field: str) -> Path:
    value = config.get(field)
    if not isinstance(value, str) or not value:
        pytest.fail(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{field} does not exist: {path}")
    return path


def test_msc_heldout_initialization_vs_trained(pytestconfig: pytest.Config) -> None:
    config_value = pytestconfig.getoption("--cdic-msc-eval-config")
    if config_value is None:
        pytest.skip("pass --cdic-msc-eval-config to run held-out MSC evaluation")
    config_path = Path(config_value)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        pytest.fail("MSC evaluation configuration must be a JSON object")

    model_path = _required_path(config, "model_path")
    checkpoint_path = _required_path(config, "checkpoint_path")
    training_checkpoint_path = _required_path(config, "training_checkpoint_path")
    data_root = _required_path(config, "data_root")
    artifact_dir = Path(str(config["artifact_dir"]))
    device = str(config.get("device", "cuda:0"))
    retrieval_config = RetrievalConfig(
        threshold=float(config.get("threshold", 0.8)),
        decay=float(config.get("decay", 0.05)),
        retrieved_state_order=RetrievedStateOrder(
            str(config.get("retrieved_state_order", "score_desc"))
        ),
        max_retrieved=(
            int(config["max_retrieved"]) if config.get("max_retrieved") is not None else None
        ),
    )
    episodes = load_msc_episodes(
        data_root,
        session_id=int(config.get("session_id", 4)),
        split=str(config.get("split", "valid")),
        max_episodes=int(config.get("max_episodes", 8)),
        max_turns_per_episode=int(config.get("max_turns_per_episode", 8)),
    )

    assert torch.cuda.is_available()
    torch.cuda.set_device(device)
    adapter = IcaeV1TrainingAdapter.load(
        IcaeV1AdapterConfig(
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            device=device,
            max_turn_tokens=int(config.get("max_turn_tokens", 512)),
            max_new_tokens=1,
            gradient_checkpointing=False,
        )
    )
    adapter.model.eval()

    def run_condition() -> tuple[dict[str, object], float, int]:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        started_at = time.perf_counter()
        with torch.inference_mode():
            result = evaluate_msc_diagnostics(
                adapter,
                episodes,
                similarity=torch_cosine_similarity,
                retrieval_config=retrieval_config,
            )
        torch.cuda.synchronize()
        return result, time.perf_counter() - started_at, torch.cuda.max_memory_allocated(device)

    initial, initial_seconds, initial_peak_memory = run_condition()
    progress = load_cdic_model_checkpoint(
        training_checkpoint_path,
        model=adapter,
    )
    adapter.model.eval()
    trained, trained_seconds, trained_peak_memory = run_condition()

    initial_records = initial["records"]
    trained_records = trained["records"]
    assert isinstance(initial_records, list) and isinstance(trained_records, list)
    assert len(initial_records) == len(trained_records)
    for initial_record, trained_record in zip(initial_records, trained_records, strict=True):
        assert isinstance(initial_record, dict) and isinstance(trained_record, dict)
        assert initial_record["turn_id"] == trained_record["turn_id"]
        assert math.isfinite(float(initial_record["loss"]))
        assert math.isfinite(float(trained_record["loss"]))

    report = {
        "config_path": str(config_path),
        "data_summary": summarize_msc_episodes(episodes),
        "device": device,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "initial_checkpoint_path": str(checkpoint_path),
        "training_checkpoint_path": str(training_checkpoint_path),
        "training_progress": {
            "epoch": progress.epoch,
            "next_episode_position": progress.next_episode_position,
            "global_step": progress.global_step,
        },
        "initialization": {
            **initial,
            "duration_seconds": initial_seconds,
            "peak_memory_bytes": initial_peak_memory,
        },
        "trained": {
            **trained,
            "duration_seconds": trained_seconds,
            "peak_memory_bytes": trained_peak_memory,
        },
    }
    print(
        json.dumps(
            {
                "initialization": report["initialization"]["summary"],
                "trained": report["trained"]["summary"],
                "training_progress": report["training_progress"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "msc_heldout_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
