from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass

from cdic_repro.config import RetrievalConfig
from cdic_repro.credit import build_credit_plan
from cdic_repro.memory_state import MemoryBank
from cdic_repro.model_protocol import CdicEvaluationAdapter
from cdic_repro.msc import MscEpisode
from cdic_repro.retrieval import SimilarityFunction, retrieve
from cdic_repro.writeback import NewStatePayload, apply_write_back


@dataclass(frozen=True, slots=True)
class _AdjacentProbe:
    query_key: object
    preceding_state_key: object


def evaluate_msc_episodes(
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
    selected_supports: list[int] = []
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
            supports = memory.select(retrieval.selected_state_ids)
            credit = build_credit_plan(retrieval)
            response_loss = model.response_loss(
                supports,
                turn.query,
                turn.response,
                credit,
            )
            loss = float(response_loss.value)  # type: ignore[arg-type]
            if not math.isfinite(loss):
                raise ValueError(f"non-finite loss for {turn.turn_id}")
            compressed = model.compress_gold(supports, turn.query, turn.response, credit)
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
            selected_supports.append(len(retrieval.selected_state_ids))
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
                    "selected_supports": len(retrieval.selected_state_ids),
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
        "mean_selected_supports": statistics.fmean(selected_supports),
        "mean_final_memory_states": statistics.fmean(final_memory_sizes),
        "peak_score": _score_summary(peak_scores),
        "adjacent_vs_cross_episode": _summarize_adjacent_probes(
            probes,
            similarity=similarity,
            retrieval_config=retrieval_config,
        ),
    }
    return {"summary": summary, "records": records}


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
        similarity(probe.query_key, probe.preceding_state_key) * decay_weight for probe in probes
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
