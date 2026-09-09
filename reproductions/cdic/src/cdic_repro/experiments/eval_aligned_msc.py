"""Paired MSC evaluation with explicit turn scopes and likelihood denominators."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from cdic_repro.checkpoint import load_cdic_model_checkpoint
from cdic_repro.config import RetrievalConfig
from cdic_repro.credit import build_credit_plan
from cdic_repro.generation_metrics import score_generation_records
from cdic_repro.icae_adapter import (
    IcaeV1AdapterConfig,
    IcaeV1TrainingAdapter,
    torch_cosine_similarity,
)
from cdic_repro.memory_state import MemoryBank
from cdic_repro.msc import MscEpisode, load_msc_episodes
from cdic_repro.retrieval import SimilarityFunction, retrieve
from cdic_repro.writeback import NewStatePayload, apply_write_back


SCOPES = ("all_turns_s2_s5", "session_final_s2_s5", "episode_final_s5")


@dataclass(frozen=True)
class AlignmentConfig:
    model_path: str
    checkpoint_path: str
    training_checkpoint_path: str | None
    data_root: str
    artifact_dir: str
    condition: str
    split: str = "valid"
    sample_size: int = 32
    sample_seed: int = 20260906
    device: str = "cuda:0"
    max_turn_tokens: int = 512
    max_new_tokens: int = 128
    threshold: float = 0.8
    decay: float = 0.05
    use_ft_markers: bool = True
    turn_template: str = "<s>[INST] {query} [/INST] {response} </s>"

    def __post_init__(self) -> None:
        if self.condition not in {"initialization", "final"}:
            raise ValueError("condition must be initialization or final")
        if (self.training_checkpoint_path is not None) != (self.condition == "final"):
            raise ValueError("only the final condition must load a training checkpoint")
        if self.split not in {"valid", "test"} or self.sample_size < 1:
            raise ValueError("use a positive sample_size and a held-out split")


def select_episodes(
    episodes: tuple[MscEpisode, ...], size: int, seed: int
) -> tuple[MscEpisode, ...]:
    if size > len(episodes):
        raise ValueError("sample_size exceeds the available episodes")
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("episode IDs must be unique")
    indices = sorted(random.Random(seed).sample(range(len(episodes)), size))
    selected = tuple(episodes[index] for index in indices)
    if any({turn.session_index for turn in episode.turns} != set(range(1, 6))
           for episode in selected):
        raise ValueError("alignment requires complete sessions 1-5 in every episode")
    return selected


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
            memory, query_key=adapter.encode_query(turn.query), turn=turn_number,
            similarity=similarity, config=retrieval_config,
        )
        retrieved_states = memory.select(retrieval.selected_state_ids)
        credit = build_credit_plan(retrieval)
        if turn.session_index >= 2 and turn.turn_id not in completed_ids:
            response_loss = adapter.response_loss(
                retrieved_states, turn.query, turn.response, credit,
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
                "is_episode_final": turn.turn_id == episode.turns[-1].turn_id,
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
        # Replay gold history even for previously completed or unscored turns.
        # The current reference is written only after that turn has been scored.
        compressed = adapter.compress_gold(
            retrieved_states,
            turn.query,
            turn.response,
            credit,
        )
        apply_write_back(
            memory, retrieval=retrieval,
            payload=NewStatePayload(
                latent=compressed.latent, retrieval_key=compressed.retrieval_key,
                provenance=compressed.provenance, graph_connected=False,
                gradient_depth=compressed.gradient_depth,
            ), turn=turn_number,
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
    metrics["turns_by_session"] = dict(sorted(Counter(
        row["session_index"] for row in records
    ).items()))
    for name in ("rouge_1", "rouge_2", "rouge_l"):
        metrics[f"{name}_f1"] = metrics.pop(name)
    metrics.pop("ppl")
    metrics["on_topic_rate"] = sum(row["retrieval"]["on_topic"] for row in records) / len(records)
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
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate prediction IDs in {path}")
    return rows


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def paired_episode_bootstrap(
    left: list[dict[str, Any]], right: dict[str, dict[str, Any]],
    seed: int = 20260906, resamples: int = 2000,
) -> dict[str, Any]:
    """Resample whole dialogues, preserving within-dialogue dependence."""
    by_episode: dict[str, tuple[float, int]] = {}
    for row in left:
        delta, tokens = by_episode.get(row["episode_id"], (0.0, 0))
        by_episode[row["episode_id"]] = (
            delta + sum(right[row["id"]]["token_nll"]) - sum(row["token_nll"]),
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
        "unit": "episode", "episodes": len(values), "seed": seed, "resamples": resamples,
        "statistic": "final minus initialization token-weighted NLL, including EOS",
        "percentile_95_interval": [estimates[int(resamples * 0.025)],
                                   estimates[int(resamples * 0.975)]],
    }


def run(config: AlignmentConfig) -> None:
    episodes = select_episodes(load_msc_episodes(
        Path(config.data_root), session_id=5, split=config.split,
    ), size=config.sample_size, seed=config.sample_seed)
    output_dir = Path(config.artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_ids = {turn.turn_id for episode in episodes for turn in episode.turns
                    if turn.session_index >= 2}
    protocol = {
        "config": asdict(config),
        "episode_ids": [episode.episode_id for episode in episodes],
        "expected_scored_turns": len(expected_ids),
        "history": "all sessions 1-5; gold write-back after scoring; no turn-count truncation",
        "initial_memory": "empty; Algorithm 1 interpretation, instruction seed unspecified",
        "scopes": SCOPES,
        "generation": "greedy; tokenizer EOS stop; max_new_tokens fixed across conditions",
        "metrics": "corpus BLEU-4; casefold regex tokenization; no smoothing; macro ROUGE recall/F1; no stemming",
        "paper_alignment": "Appendix K explicitly uses session-5 final turn; Table 1 scope and metric implementation remain unconfirmed",
    }
    # JSON canonicalization makes tuple/list serialization stable on resume.
    protocol = json.loads(json.dumps(protocol))
    protocol_path = output_dir / "protocol.json"
    output_path = output_dir / "predictions.jsonl"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("refusing to mix runs with different evaluation protocols")
    elif output_path.exists():
        raise ValueError("predictions exist without a recorded protocol")
    else:
        write_json(protocol_path, protocol)
    completed_ids = {row["id"] for row in read_records(output_path)}
    if not completed_ids <= expected_ids:
        raise ValueError("saved predictions do not belong to this evaluation sample")
    adapter = IcaeV1TrainingAdapter.load(IcaeV1AdapterConfig(
        model_path=Path(config.model_path), checkpoint_path=Path(config.checkpoint_path),
        device=config.device, max_turn_tokens=config.max_turn_tokens,
        max_new_tokens=config.max_new_tokens, use_ft_markers=config.use_ft_markers,
        turn_template=config.turn_template, gradient_checkpointing=False,
    ))
    progress = None
    if config.training_checkpoint_path is not None:
        progress = asdict(load_cdic_model_checkpoint(
            Path(config.training_checkpoint_path), model=adapter,
        ))
    adapter.model.eval()
    started = time.perf_counter()
    write_json(output_dir / "runtime.json", {
        "torch_version": torch.__version__, "device": config.device,
        "gpu_name": torch.cuda.get_device_name(config.device),
        "stop_token_id": int(adapter.model.tokenizer.eos_token_id),
        "training_progress": progress,
    })
    with output_path.open("a", encoding="utf-8") as output, torch.inference_mode():
        for index, episode in enumerate(episodes, start=1):
            episode_ids = {turn.turn_id for turn in episode.turns if turn.session_index >= 2}
            if not episode_ids <= completed_ids:
                for row in evaluate_episode(
                    adapter, episode, completed_ids=completed_ids,
                    retrieval_config=RetrievalConfig(threshold=config.threshold, decay=config.decay),
                ):
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    completed_ids.add(row["id"])
            print(json.dumps({"condition": config.condition, "episode": index,
                              "episodes": len(episodes), "completed_turns": len(completed_ids),
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
    if completed_ids != expected_ids:
        raise ValueError("evaluation ended with missing target turns")
    records = read_records(output_path)
    summary = {scope: summarize_records(scoped_records(records, scope)) for scope in SCOPES}
    summary["by_session"] = {
        str(session): summarize_records([row for row in records if row["session_index"] == session])
        for session in range(2, 6)
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"condition": config.condition, "status": "complete",
                      "episode_final_s5": summary["episode_final_s5"]}), flush=True)


def compare_runs(initialization: Path, final: Path, output: Path) -> dict[str, Any]:
    protocols = [json.loads((folder / "protocol.json").read_text())
                 for folder in (initialization, final)]
    ignored = {"condition", "training_checkpoint_path", "artifact_dir", "device"}
    configs = [{key: value for key, value in protocol["config"].items() if key not in ignored}
               for protocol in protocols]
    if configs[0] != configs[1] or protocols[0]["episode_ids"] != protocols[1]["episode_ids"]:
        raise ValueError("comparison requires matched samples and inference settings")
    for protocol, condition in zip(protocols, ("initialization", "final")):
        if protocol["config"]["condition"] != condition:
            raise ValueError("comparison conditions are reversed or mislabeled")
    rows = [read_records(folder / "predictions.jsonl") for folder in (initialization, final)]
    maps = [{row["id"]: row for row in records} for records in rows]
    if maps[0].keys() != maps[1].keys() or any(
        len(records) != protocol["expected_scored_turns"]
        for records, protocol in zip(rows, protocols)
    ):
        raise ValueError("comparison requires both complete, matching target sets")
    for row_id, left in maps[0].items():
        right = maps[1][row_id]
        for field in ("query", "reference", "loss_tokens", "episode_id", "session_index",
                      "is_session_final", "is_episode_final"):
            if left[field] != right[field]:
                raise ValueError(f"paired target mismatch: {row_id}, {field}")
    report: dict[str, Any] = {
        "episode_ids": protocols[0]["episode_ids"], "common_config": configs[0], "scopes": {},
    }
    for scope in SCOPES:
        selected = [scoped_records(records, scope) for records in rows]
        metrics = [summarize_records(records) for records in selected]
        deltas = [maps[1][row["id"]]["loss"] - row["loss"] for row in selected[0]]
        report["scopes"][scope] = {
            "initialization": metrics[0], "final": metrics[1],
            "paired_turns_improved": sum(delta < 0 for delta in deltas),
            "paired_turns": len(deltas),
            "mean_paired_turn_nll_delta": sum(deltas) / len(deltas),
            "token_weighted_nll_delta": metrics[1]["token_weighted_loss"] - metrics[0]["token_weighted_loss"],
            "paired_episode_bootstrap": paired_episode_bootstrap(selected[0], maps[1]),
        }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--compare", nargs=2, type=Path, metavar=("INITIALIZATION", "FINAL"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.config:
        run(AlignmentConfig(**json.loads(arguments.config.read_text())))
    else:
        if arguments.output is None:
            parser.error("--compare requires --output")
        compare_runs(*arguments.compare, arguments.output)


if __name__ == "__main__":
    main()
