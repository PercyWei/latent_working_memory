"""统一的 MSC 评估与 initialization/final 配对比较入口。"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

from cdic_repro.config import RetrievalConfig, RetrievedStateOrder
from cdic_repro.credit import build_credit_plan
from cdic_repro.experiments.checkpoint import load_cdic_model_checkpoint
from cdic_repro.experiments.generation_metrics import score_generation_records
from cdic_repro.experiments.msc.data import (
    MscEpisode,
    load_msc_episodes,
    summarize_msc_episodes,
)
from cdic_repro.icae.adapter import (
    IcaeV1AdapterConfig,
    IcaeV1TrainingAdapter,
    torch_cosine_similarity,
)
from cdic_repro.memory_state import MemoryBank
from cdic_repro.model_protocol import CdicEvaluationAdapter
from cdic_repro.retrieval import SimilarityFunction, retrieve
from cdic_repro.writeback import NewStatePayload, apply_write_back


SCOPES = ("all_turns_s2_s5", "session_final_s2_s5", "episode_final_s5")


@dataclass(frozen=True, slots=True)
class MscEvaluationConfig:
    model_path: Path
    checkpoint_path: Path
    data_root: Path
    artifact_dir: Path
    condition: str
    training_checkpoint_path: Path | None = None
    split: str = "valid"
    episode_count: int | None = None
    sample_seed: int | None = None
    device: str = "cuda:0"
    max_turn_tokens: int = 512
    max_new_tokens: int = 128
    threshold: float = 0.8
    decay: float = 0.05
    retrieved_state_order: RetrievedStateOrder = RetrievedStateOrder.SCORE_DESC
    max_retrieved: int | None = None
    use_ft_markers: bool = True
    turn_template: str = "<s>[INST] {query} [/INST] {response} </s>"
    shard_index: int = 0
    num_shards: int = 1

    def __post_init__(self) -> None:
        if self.condition not in {"initialization", "final"}:
            raise ValueError("condition must be initialization or final")
        if (self.training_checkpoint_path is not None) != (self.condition == "final"):
            raise ValueError("only the final condition must load a training checkpoint")
        if self.split not in {"valid", "test"}:
            raise ValueError("split must be valid or test")
        if self.episode_count is not None and self.episode_count < 1:
            raise ValueError("episode_count must be positive when provided")
        if self.sample_seed is not None and self.episode_count is None:
            raise ValueError("sample_seed requires episode_count")
        if not 0 <= self.shard_index < self.num_shards:
            raise ValueError("shard_index must be within [0, num_shards)")


@dataclass(frozen=True, slots=True)
class _AdjacentProbe:
    query_key: object
    preceding_state_key: object


def load_config(path: Path) -> MscEvaluationConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("MSC evaluation config must be a JSON object")
    field_names = {field.name for field in fields(MscEvaluationConfig)}
    unknown_fields = set(payload) - field_names
    if unknown_fields:
        raise ValueError(f"unknown MSC evaluation config fields: {sorted(unknown_fields)}")
    required_fields = {"model_path", "checkpoint_path", "data_root", "artifact_dir", "condition"}
    missing_fields = required_fields - set(payload)
    if missing_fields:
        raise ValueError(f"missing MSC evaluation config fields: {sorted(missing_fields)}")
    for field_name in ("model_path", "checkpoint_path", "data_root", "artifact_dir"):
        value = payload[field_name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} must be a non-empty string")
        payload[field_name] = Path(value)
    training_checkpoint_path = payload.get("training_checkpoint_path")
    if training_checkpoint_path is not None:
        if not isinstance(training_checkpoint_path, str) or not training_checkpoint_path:
            raise ValueError("training_checkpoint_path must be a non-empty string or null")
        payload["training_checkpoint_path"] = Path(training_checkpoint_path)
    if "retrieved_state_order" in payload:
        payload["retrieved_state_order"] = RetrievedStateOrder(payload["retrieved_state_order"])
    return MscEvaluationConfig(**payload)


def select_episodes(
    episodes: tuple[MscEpisode, ...],
    count: int | None,
    seed: int | None,
) -> tuple[MscEpisode, ...]:
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("episode IDs must be unique")
    if seed is not None and count is None:
        raise ValueError("seed requires episode_count")
    if count is None:
        selected = episodes
    elif count < 1:
        raise ValueError("episode_count must be positive")
    elif count > len(episodes):
        raise ValueError("episode_count exceeds the available episodes")
    elif seed is None:
        selected = episodes[:count]
    else:
        indices = sorted(random.Random(seed).sample(range(len(episodes)), count))
        selected = tuple(episodes[index] for index in indices)
    if any(
        {turn.session_index for turn in episode.turns} != set(range(1, 6))
        for episode in selected
    ):
        raise ValueError("MSC evaluation requires complete sessions 1-5 in every episode")
    return selected


def evaluate_msc_diagnostics(
    model: CdicEvaluationAdapter,
    episodes: tuple[MscEpisode, ...],
    similarity: SimilarityFunction,
    retrieval_config: RetrievalConfig,
) -> dict[str, object]:
    if not episodes:
        raise ValueError("evaluation requires at least one episode")

    records: list[dict[str, object]] = []
    probes: list[_AdjacentProbe] = []
    action_counts: Counter[str] = Counter()
    peak_scores: list[float] = []
    selected_state_counts: list[int] = []
    final_memory_sizes: list[int] = []
    loss_sum = 0.0
    weighted_loss_sum = 0.0
    loss_tokens = 0

    for episode in episodes:
        memory = MemoryBank()
        for turn_number, turn in enumerate(episode.turns, start=1):
            query_key = model.encode_query(turn.query)
            if turn_number == 2 and len(memory) == 1:
                probes.append(
                    _AdjacentProbe(
                        query_key=query_key,
                        preceding_state_key=memory.states[0].retrieval_key,
                    )
                )
            retrieval = retrieve(
                memory,
                query_key=query_key,
                turn=turn_number,
                similarity=similarity,
                config=retrieval_config,
            )
            retrieved_states = memory.select(retrieval.selected_state_ids)
            credit = build_credit_plan(retrieval)
            response_loss = model.response_loss(
                retrieved_states,
                turn.query,
                turn.response,
                credit,
            )
            loss = float(response_loss.value)
            if not math.isfinite(loss):
                raise ValueError(f"non-finite loss for {turn.turn_id}")
            compressed = model.compress_gold(
                retrieved_states,
                turn.query,
                turn.response,
                credit,
            )
            write_back = apply_write_back(
                memory,
                retrieval=retrieval,
                payload=NewStatePayload(
                    latent=compressed.latent,
                    retrieval_key=compressed.retrieval_key,
                    provenance=compressed.provenance,
                    graph_connected=False,
                    gradient_depth=compressed.gradient_depth,
                ),
                turn=turn_number,
            )
            loss_sum += loss
            weighted_loss_sum += loss * response_loss.token_count
            loss_tokens += response_loss.token_count
            action_counts[write_back.action.value] += 1
            selected_state_counts.append(len(retrieval.selected_state_ids))
            if retrieval.peak_score is not None:
                peak_scores.append(retrieval.peak_score)
            records.append(
                {
                    "episode_id": episode.episode_id,
                    "turn": turn_number,
                    "turn_id": turn.turn_id,
                    "session_index": turn.session_index,
                    "pair_index": turn.pair_index,
                    "loss": loss,
                    "loss_tokens": response_loss.token_count,
                    "peak_score": retrieval.peak_score,
                    "on_topic": retrieval.on_topic,
                    "used_fallback": retrieval.used_fallback,
                    "selected_states": len(retrieval.selected_state_ids),
                    "write_action": write_back.action.value,
                    "memory_states": len(memory),
                }
            )
        final_memory_sizes.append(len(memory))

    weighted_loss = weighted_loss_sum / loss_tokens
    summary = {
        "episodes": len(episodes),
        "turns": len(records),
        "loss_tokens": loss_tokens,
        "mean_turn_loss": loss_sum / len(records),
        "token_weighted_loss": weighted_loss,
        "token_weighted_perplexity": math.exp(weighted_loss),
        "action_counts": dict(sorted(action_counts.items())),
        "on_topic_rate_after_first_turn": action_counts["replace"]
        / max(1, len(records) - len(episodes)),
        "mean_selected_states": statistics.fmean(selected_state_counts),
        "mean_final_memory_states": statistics.fmean(final_memory_sizes),
        "peak_score": _score_summary(peak_scores),
        "adjacent_vs_cross_episode": _summarize_adjacent_probes(
            probes,
            similarity=similarity,
            retrieval_config=retrieval_config,
        ),
    }
    return {"summary": summary, "records": records}


def evaluate_episode(
    adapter: Any,
    episode: MscEpisode,
    retrieval_config: RetrievalConfig,
    completed_ids: set[str],
    similarity: SimilarityFunction = torch_cosine_similarity,
):
    memory = MemoryBank()
    final_ids = {turn.session_index: turn.turn_id for turn in episode.turns}
    for turn_number, turn in enumerate(episode.turns, start=1):
        retrieval = retrieve(
            memory,
            query_key=adapter.encode_query(turn.query),
            turn=turn_number,
            similarity=similarity,
            config=retrieval_config,
        )
        retrieved_states = memory.select(retrieval.selected_state_ids)
        credit = build_credit_plan(retrieval)
        if turn.session_index >= 2 and turn.turn_id not in completed_ids:
            response_loss = adapter.response_loss(
                retrieved_states,
                turn.query,
                turn.response,
                credit,
                collect_token_nll=True,
            )
            token_nll = response_loss.token_nll
            if token_nll is None or len(token_nll) < 2:
                raise ValueError("each response requires content tokens and an EOS target")
            loss = float(response_loss.value)
            if not all(math.isfinite(value) for value in token_nll):
                raise ValueError(f"non-finite token NLL at {turn.turn_id}")
            if not math.isclose(sum(token_nll) / len(token_nll), loss, abs_tol=1e-5):
                raise ValueError(f"token NLL does not reconstruct response loss: {turn.turn_id}")
            yield {
                "id": turn.turn_id,
                "episode_id": episode.episode_id,
                "turn": turn_number,
                "session_index": turn.session_index,
                "pair_index": turn.pair_index,
                "is_session_final": turn.turn_id == final_ids[turn.session_index],
                "is_episode_final": (
                    turn.session_index == 5 and turn.turn_id == final_ids.get(5)
                ),
                "query": turn.query,
                "reference": turn.response,
                "prediction": adapter.generate(retrieved_states, turn.query),
                "loss": loss,
                "loss_tokens": response_loss.token_count,
                "token_nll": list(token_nll),
                "retrieval": {
                    "peak_score": retrieval.peak_score,
                    "on_topic": retrieval.on_topic,
                    "used_fallback": retrieval.used_fallback,
                    "selected_state_ids": list(retrieval.selected_state_ids),
                    "memory_states_before": len(memory),
                },
            }
        # 已完成和不计分的轮次也必须重放 gold history；当前答案在计分后才写回。
        compressed = adapter.compress_gold(
            retrieved_states,
            turn.query,
            turn.response,
            credit,
        )
        apply_write_back(
            memory,
            retrieval=retrieval,
            payload=NewStatePayload(
                latent=compressed.latent,
                retrieval_key=compressed.retrieval_key,
                provenance=compressed.provenance,
                graph_connected=False,
                gradient_depth=compressed.gradient_depth,
            ),
            turn=turn_number,
        )


def scoped_records(records: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "all_turns_s2_s5":
        return records
    if scope == "session_final_s2_s5":
        return [row for row in records if row["is_session_final"]]
    if scope == "episode_final_s5":
        return [row for row in records if row["is_episode_final"]]
    raise ValueError(f"unknown scope: {scope}")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = score_generation_records(records)
    all_nll = [value for row in records for value in row["token_nll"]]
    content_nll = [value for row in records for value in row["token_nll"][:-1]]
    eos_nll = [row["token_nll"][-1] for row in records]
    if not content_nll:
        raise ValueError("cannot compute content PPL without response content tokens")
    metrics["ppl_including_eos"] = math.exp(sum(all_nll) / len(all_nll))
    metrics["ppl_excluding_eos"] = math.exp(sum(content_nll) / len(content_nll))
    metrics["content_tokens"] = len(content_nll)
    metrics["eos_tokens"] = len(eos_nll)
    metrics["mean_eos_nll"] = sum(eos_nll) / len(eos_nll)
    metrics["mean_turn_nll"] = sum(row["loss"] for row in records) / len(records)
    metrics["episodes"] = len({row["episode_id"] for row in records})
    metrics["turns_by_session"] = dict(
        sorted(Counter(row["session_index"] for row in records).items())
    )
    for name in ("rouge_1", "rouge_2", "rouge_l"):
        metrics[f"{name}_f1"] = metrics.pop(name)
    metrics.pop("ppl")
    metrics["on_topic_rate"] = sum(
        row["retrieval"]["on_topic"] for row in records
    ) / len(records)
    metrics["mean_retrieved_states"] = sum(
        len(row["retrieval"]["selected_state_ids"]) for row in records
    ) / len(records)
    metrics["mean_memory_states_before"] = sum(
        row["retrieval"]["memory_states_before"] for row in records
    ) / len(records)
    return metrics


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate prediction IDs in {path}")
    return rows


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def paired_episode_bootstrap(
    initialization: list[dict[str, Any]],
    final: dict[str, dict[str, Any]],
    seed: int = 20260906,
    resamples: int = 2000,
) -> dict[str, Any]:
    """以完整 episode 为单位重采样，保留同一对话内各轮的相关性。"""
    by_episode: dict[str, tuple[float, int]] = {}
    for row in initialization:
        delta, tokens = by_episode.get(row["episode_id"], (0.0, 0))
        by_episode[row["episode_id"]] = (
            delta + sum(final[row["id"]]["token_nll"]) - sum(row["token_nll"]),
            tokens + row["loss_tokens"],
        )
    values = list(by_episode.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = rng.choices(values, k=len(values))
        estimates.append(sum(delta for delta, _ in sample) / sum(n for _, n in sample))
    estimates.sort()
    return {
        "unit": "episode",
        "episodes": len(values),
        "seed": seed,
        "resamples": resamples,
        "statistic": "final minus initialization token-weighted NLL, including EOS",
        "percentile_95_interval": [
            estimates[int(resamples * 0.025)],
            estimates[int(resamples * 0.975)],
        ],
    }


def run(config: MscEvaluationConfig) -> None:
    _validate_resources(config)
    loaded_episodes = load_msc_episodes(
        config.data_root,
        session_id=5,
        split=config.split,
    )
    selected_episodes = select_episodes(
        loaded_episodes,
        count=config.episode_count,
        seed=config.sample_seed,
    )
    episodes = tuple(
        episode
        for index, episode in enumerate(selected_episodes)
        if index % config.num_shards == config.shard_index
    )
    if not episodes:
        raise ValueError("the selected shard contains no episodes")

    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_ids = {
        turn.turn_id
        for episode in episodes
        for turn in episode.turns
        if turn.session_index >= 2
    }
    protocol = {
        "config": _serialize_config(config),
        "episode_ids": [episode.episode_id for episode in episodes],
        "expected_scored_turns": len(expected_ids),
        "data_summary_before_sharding": summarize_msc_episodes(selected_episodes),
        "history": "sessions 1-5 gold write-back after scoring",
        "scored_sessions": "2-5",
        "initial_memory": "empty; Algorithm 1 interpretation, instruction seed unspecified",
        "scopes": SCOPES,
        "generation": "greedy; tokenizer EOS stop; max_new_tokens fixed across conditions",
        "metrics": (
            "corpus BLEU-4; casefold regex tokenization; no smoothing; "
            "macro ROUGE recall/F1; no stemming"
        ),
        "paper_alignment": (
            "Appendix K explicitly uses session-5 final turn; "
            "Table 1 scope and metric implementation remain unconfirmed"
        ),
    }
    protocol = json.loads(json.dumps(protocol))
    protocol_path = output_dir / "protocol.json"
    output_path = output_dir / "predictions.jsonl"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("refusing to mix runs with different evaluation protocols")
    elif output_path.exists():
        raise ValueError("predictions exist without a recorded protocol")
    else:
        write_json(protocol_path, protocol)

    completed_ids = {row["id"] for row in read_records(output_path)}
    if not completed_ids <= expected_ids:
        raise ValueError("saved predictions do not belong to this evaluation sample")
    adapter = IcaeV1TrainingAdapter.load(
        IcaeV1AdapterConfig(
            model_path=config.model_path,
            checkpoint_path=config.checkpoint_path,
            device=config.device,
            max_turn_tokens=config.max_turn_tokens,
            max_new_tokens=config.max_new_tokens,
            use_ft_markers=config.use_ft_markers,
            turn_template=config.turn_template,
            gradient_checkpointing=False,
        )
    )
    progress = None
    if config.training_checkpoint_path is not None:
        progress = asdict(
            load_cdic_model_checkpoint(config.training_checkpoint_path, model=adapter)
        )
    adapter.model.eval()
    retrieval_config = RetrievalConfig(
        threshold=config.threshold,
        decay=config.decay,
        retrieved_state_order=config.retrieved_state_order,
        max_retrieved=config.max_retrieved,
    )
    started = time.perf_counter()
    write_json(
        output_dir / "runtime.json",
        {
            "torch_version": torch.__version__,
            "device": config.device,
            "gpu_name": torch.cuda.get_device_name(config.device),
            "stop_token_id": int(adapter.model.tokenizer.eos_token_id),
            "training_progress": progress,
        },
    )
    error_path = output_dir / "errors.jsonl"
    with (
        output_path.open("a", encoding="utf-8") as output,
        error_path.open("a", encoding="utf-8") as errors,
        torch.inference_mode(),
    ):
        for index, episode in enumerate(episodes, start=1):
            episode_ids = {
                turn.turn_id for turn in episode.turns if turn.session_index >= 2
            }
            if not episode_ids <= completed_ids:
                try:
                    for row in evaluate_episode(
                        adapter,
                        episode,
                        retrieval_config=retrieval_config,
                        completed_ids=completed_ids,
                    ):
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        completed_ids.add(row["id"])
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
            print(
                json.dumps(
                    {
                        "condition": config.condition,
                        "episode": index,
                        "episodes": len(episodes),
                        "completed_turns": len(completed_ids),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    if completed_ids != expected_ids:
        raise ValueError("evaluation ended with missing target turns")

    records = read_records(output_path)
    scopes = {
        scope: summarize_records(selected)
        for scope in SCOPES
        if (selected := scoped_records(records, scope))
    }
    summary = {
        "condition": config.condition,
        "duration_seconds": time.perf_counter() - started,
        "scopes": scopes,
        "by_session": {
            str(session): summarize_records(selected)
            for session in range(2, 6)
            if (selected := [row for row in records if row["session_index"] == session])
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "condition": config.condition,
                "status": "complete",
                "episode_final_s5": scopes.get("episode_final_s5"),
            }
        ),
        flush=True,
    )


def compare_runs(initialization: Path, final: Path, output: Path) -> dict[str, Any]:
    protocols = [
        json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
        for folder in (initialization, final)
    ]
    ignored = {"condition", "training_checkpoint_path", "artifact_dir", "device"}
    configs = [
        {key: value for key, value in protocol["config"].items() if key not in ignored}
        for protocol in protocols
    ]
    if configs[0] != configs[1] or protocols[0]["episode_ids"] != protocols[1]["episode_ids"]:
        raise ValueError("comparison requires matched samples and inference settings")
    for protocol, condition in zip(protocols, ("initialization", "final"), strict=True):
        if protocol["config"]["condition"] != condition:
            raise ValueError("comparison conditions are reversed or mislabeled")
    rows = [read_records(folder / "predictions.jsonl") for folder in (initialization, final)]
    maps = [{row["id"]: row for row in records} for records in rows]
    if maps[0].keys() != maps[1].keys() or any(
        len(records) != protocol["expected_scored_turns"]
        for records, protocol in zip(rows, protocols, strict=True)
    ):
        raise ValueError("comparison requires both complete, matching target sets")
    for row_id, left in maps[0].items():
        right = maps[1][row_id]
        for field in (
            "query",
            "reference",
            "loss_tokens",
            "episode_id",
            "session_index",
            "is_session_final",
            "is_episode_final",
        ):
            if left[field] != right[field]:
                raise ValueError(f"paired target mismatch: {row_id}, {field}")

    report: dict[str, Any] = {
        "episode_ids": protocols[0]["episode_ids"],
        "common_config": configs[0],
        "scopes": {},
    }
    for scope in SCOPES:
        selected = [scoped_records(records, scope) for records in rows]
        if not selected[0]:
            continue
        metrics = [summarize_records(records) for records in selected]
        deltas = [maps[1][row["id"]]["loss"] - row["loss"] for row in selected[0]]
        report["scopes"][scope] = {
            "initialization": metrics[0],
            "final": metrics[1],
            "paired_turns_improved": sum(delta < 0 for delta in deltas),
            "paired_turns": len(deltas),
            "mean_paired_turn_nll_delta": sum(deltas) / len(deltas),
            "token_weighted_nll_delta": (
                metrics[1]["token_weighted_loss"] - metrics[0]["token_weighted_loss"]
            ),
            "paired_episode_bootstrap": paired_episode_bootstrap(selected[0], maps[1]),
        }
    write_json(output, report)
    return report


def _summarize_adjacent_probes(
    probes: list[_AdjacentProbe],
    similarity: SimilarityFunction,
    retrieval_config: RetrievalConfig,
) -> dict[str, object]:
    if len(probes) < 2:
        return {
            "pairs": 0,
            "same_episode_scores": _score_summary([]),
            "cross_episode_scores": _score_summary([]),
            "mean_margin": None,
            "pairwise_accuracy": None,
            "same_episode_accept_rate": None,
            "cross_episode_false_accept_rate": None,
        }
    decay_weight = math.exp(-retrieval_config.decay)
    same_scores = [
        similarity(probe.query_key, probe.preceding_state_key) * decay_weight
        for probe in probes
    ]
    cross_scores = [
        similarity(probe.query_key, probes[(index + 1) % len(probes)].preceding_state_key)
        * decay_weight
        for index, probe in enumerate(probes)
    ]
    margins = [same - cross for same, cross in zip(same_scores, cross_scores, strict=True)]
    return {
        "pairs": len(probes),
        "same_episode_scores": _score_summary(same_scores),
        "cross_episode_scores": _score_summary(cross_scores),
        "mean_margin": statistics.fmean(margins),
        "pairwise_accuracy": sum(margin > 0.0 for margin in margins) / len(margins),
        "same_episode_accept_rate": sum(
            score >= retrieval_config.threshold for score in same_scores
        )
        / len(same_scores),
        "cross_episode_false_accept_rate": sum(
            score >= retrieval_config.threshold for score in cross_scores
        )
        / len(cross_scores),
    }


def _score_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _validate_resources(config: MscEvaluationConfig) -> None:
    resources = {
        "model_path": config.model_path,
        "checkpoint_path": config.checkpoint_path,
        "data_root": config.data_root,
    }
    if config.training_checkpoint_path is not None:
        resources["training_checkpoint_path"] = config.training_checkpoint_path
    for field_name, path in resources.items():
        if not path.exists():
            raise FileNotFoundError(f"{field_name} does not exist: {path}")


def _serialize_config(config: MscEvaluationConfig) -> dict[str, object]:
    return {
        field.name: (
            str(value)
            if isinstance(value, Path)
            else value.value
            if isinstance(value, RetrievedStateOrder)
            else value
        )
        for field in fields(config)
        for value in (getattr(config, field.name),)
    }


def _output_dir(config: MscEvaluationConfig) -> Path:
    if config.num_shards == 1:
        return config.artifact_dir
    return config.artifact_dir / f"shard-{config.shard_index:02d}-of-{config.num_shards:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("INITIALIZATION", "FINAL"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.config:
        run(load_config(arguments.config))
    else:
        if arguments.output is None:
            parser.error("--compare requires --output")
        compare_runs(*arguments.compare, arguments.output)


if __name__ == "__main__":
    main()
