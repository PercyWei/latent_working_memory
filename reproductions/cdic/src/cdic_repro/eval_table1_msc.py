from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cdic_repro.config import RetrievalConfig, SupportOrder
from cdic_repro.credit import build_credit_plan
from cdic_repro.generation_metrics import score_generation_records
from cdic_repro.icae_adapter import (
    IcaeV1AdapterConfig,
    IcaeV1TrainingAdapter,
    torch_cosine_similarity,
)
from cdic_repro.memory_state import MemoryBank
from cdic_repro.msc import MscEpisode, load_msc_episodes, summarize_msc_episodes
from cdic_repro.retrieval import retrieve
from cdic_repro.training_checkpoint import load_model_from_training_checkpoint
from cdic_repro.writeback import NewStatePayload, apply_write_back


@dataclass(frozen=True, slots=True)
class Table1MscConfig:
    model_path: Path
    checkpoint_path: Path
    training_checkpoint_path: Path
    data_root: Path
    artifact_dir: Path
    device: str = "cuda:0"
    session_id: int = 5
    split: str = "test"
    target_min_session: int = 2
    max_episodes: int | None = None
    max_turns_per_episode: int | None = None
    max_turn_tokens: int = 512
    max_new_tokens: int = 128
    threshold: float = 0.8
    decay: float = 0.05
    support_order: SupportOrder = SupportOrder.SCORE_DESC
    max_retrieved: int | None = None
    shard_index: int = 0
    num_shards: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.shard_index < self.num_shards:
            raise ValueError("shard_index must be within [0, num_shards)")
        if self.target_min_session < 1 or self.target_min_session > self.session_id:
            raise ValueError("target_min_session must be within the loaded session range")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the C-DIC row of Table 1 on MSC")
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    run(config, config_path=arguments.config)


def load_config(path: Path) -> Table1MscConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Table 1 MSC config must be a JSON object")
    return Table1MscConfig(
        model_path=Path(_required_string(payload, "model_path")),
        checkpoint_path=Path(_required_string(payload, "checkpoint_path")),
        training_checkpoint_path=Path(_required_string(payload, "training_checkpoint_path")),
        data_root=Path(_required_string(payload, "data_root")),
        artifact_dir=Path(_required_string(payload, "artifact_dir")),
        device=str(payload.get("device", "cuda:0")),
        session_id=int(payload.get("session_id", 5)),
        split=str(payload.get("split", "test")),
        target_min_session=int(payload.get("target_min_session", 2)),
        max_episodes=_optional_int(payload.get("max_episodes")),
        max_turns_per_episode=_optional_int(payload.get("max_turns_per_episode")),
        max_turn_tokens=int(payload.get("max_turn_tokens", 512)),
        max_new_tokens=int(payload.get("max_new_tokens", 128)),
        threshold=float(payload.get("threshold", 0.8)),
        decay=float(payload.get("decay", 0.05)),
        support_order=SupportOrder(str(payload.get("support_order", "score_desc"))),
        max_retrieved=_optional_int(payload.get("max_retrieved")),
        shard_index=int(payload.get("shard_index", 0)),
        num_shards=int(payload.get("num_shards", 1)),
    )


def run(config: Table1MscConfig, *, config_path: Path) -> None:
    _validate_resources(config)
    episodes = load_msc_episodes(
        config.data_root,
        session_id=config.session_id,
        split=config.split,
        max_episodes=config.max_episodes,
        max_turns_per_episode=config.max_turns_per_episode,
    )
    episodes = tuple(
        episode for index, episode in enumerate(episodes) if index % config.num_shards == config.shard_index
    )
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.artifact_dir / f"predictions.shard{config.shard_index:02d}.jsonl"
    error_path = config.artifact_dir / f"errors.shard{config.shard_index:02d}.jsonl"
    completed_ids = _load_completed_ids(output_path)

    adapter = IcaeV1TrainingAdapter.load(
        IcaeV1AdapterConfig(
            model_path=config.model_path,
            checkpoint_path=config.checkpoint_path,
            device=config.device,
            max_turn_tokens=config.max_turn_tokens,
            max_new_tokens=config.max_new_tokens,
            gradient_checkpointing=False,
        )
    )
    progress = load_model_from_training_checkpoint(
        config.training_checkpoint_path,
        model=adapter,
    )
    adapter.model.eval()
    retrieval_config = RetrievalConfig(
        threshold=config.threshold,
        decay=config.decay,
        support_order=config.support_order,
        max_retrieved=config.max_retrieved,
    )

    started_at = time.perf_counter()
    processed = 0
    with (
        output_path.open("a", encoding="utf-8") as output,
        error_path.open("a", encoding="utf-8") as errors,
        torch.inference_mode(),
    ):
        for episode in episodes:
            try:
                records = evaluate_episode(
                    adapter,
                    episode,
                    retrieval_config=retrieval_config,
                    target_min_session=config.target_min_session,
                    completed_ids=completed_ids,
                )
                for record in records:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    completed_ids.add(str(record["id"]))
                    processed += 1
            except Exception as error:
                errors.write(
                    json.dumps(
                        {
                            "episode_id": episode.episode_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                errors.flush()
                raise

    records = _read_records(output_path)
    metrics = score_generation_records(records)
    metrics.update(
        {
            "duration_seconds": time.perf_counter() - started_at,
            "processed_this_run": processed,
            "episodes_in_shard": len(episodes),
        }
    )
    _write_json(config.artifact_dir / f"metrics.shard{config.shard_index:02d}.json", metrics)
    _write_json(
        config.artifact_dir / f"manifest.shard{config.shard_index:02d}.json",
        {
            "config_path": str(config_path),
            "config": _serialize_config(config),
            "data_summary_before_sharding": summarize_msc_episodes(
                load_msc_episodes(
                    config.data_root,
                    session_id=config.session_id,
                    split=config.split,
                    max_episodes=config.max_episodes,
                    max_turns_per_episode=config.max_turns_per_episode,
                )
            ),
            "training_progress": {
                "epoch": progress.epoch,
                "next_episode_position": progress.next_episode_position,
                "global_step": progress.global_step,
            },
            "protocol": {
                "history": "gold responses are compressed after every turn",
                "scored_sessions": f"{config.target_min_session}-{config.session_id}",
                "metric_implementation": "local corpus BLEU-4 and macro ROUGE F1/recall",
                "paper_ambiguities": [
                    "generation length",
                    "BLEU smoothing and tokenizer",
                    "ROUGE tokenizer, stemming, and aggregation",
                    "exact MSC split aggregation",
                    "instruction-memory initialization",
                ],
            },
        },
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def evaluate_episode(
    adapter: Any,
    episode: MscEpisode,
    *,
    retrieval_config: RetrievalConfig,
    target_min_session: int,
    completed_ids: set[str],
) -> list[dict[str, object]]:
    memory = MemoryBank()
    results: list[dict[str, object]] = []
    for turn_number, turn in enumerate(episode.turns, start=1):
        query_key = adapter.encode_query(turn.query)
        retrieval = retrieve(
            memory,
            query_key=query_key,
            turn=turn_number,
            similarity=torch_cosine_similarity,
            config=retrieval_config,
        )
        supports = memory.select(retrieval.selected_state_ids)
        record_id = turn.turn_id
        if turn.session_index >= target_min_session and record_id not in completed_ids:
            credit = build_credit_plan(retrieval)
            response_loss = adapter.response_loss(supports, turn.query, turn.response, credit)
            prediction = adapter.generate(supports, turn.query)
            results.append(
                {
                    "id": record_id,
                    "episode_id": episode.episode_id,
                    "turn": turn_number,
                    "session_index": turn.session_index,
                    "pair_index": turn.pair_index,
                    "query": turn.query,
                    "reference": turn.response,
                    "prediction": prediction,
                    "loss": adapter.loss_to_float(response_loss.value),
                    "loss_tokens": response_loss.token_count,
                    "retrieval": {
                        "peak_score": retrieval.peak_score,
                        "on_topic": retrieval.on_topic,
                        "used_fallback": retrieval.used_fallback,
                        "selected_state_ids": list(retrieval.selected_state_ids),
                    },
                }
            )
        compressed = adapter.compress_gold(supports, turn.query, turn.response)
        apply_write_back(
            memory,
            retrieval=retrieval,
            payload=NewStatePayload(
                latent=compressed.latent,
                retrieval_key=compressed.retrieval_key,
                provenance=compressed.provenance,
                graph_connected=False,
            ),
            turn=turn_number,
        )
    return results


def _load_completed_ids(path: Path) -> set[str]:
    return {str(record["id"]) for record in _read_records(path)}


def _read_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise ValueError(f"invalid prediction record at {path}:{line_number}")
            records.append(record)
    return records


def _validate_resources(config: Table1MscConfig) -> None:
    for field in ("model_path", "checkpoint_path", "training_checkpoint_path", "data_root"):
        path = getattr(config, field)
        if not path.exists():
            raise FileNotFoundError(f"{field} does not exist: {path}")


def _required_string(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _serialize_config(config: Table1MscConfig) -> dict[str, object]:
    return {
        field: (str(value) if isinstance(value, Path) else value.value if isinstance(value, SupportOrder) else value)
        for field, value in ((name, getattr(config, name)) for name in config.__dataclass_fields__)
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
