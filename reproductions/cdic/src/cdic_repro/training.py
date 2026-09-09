from __future__ import annotations

from dataclasses import dataclass

from cdic_repro.config import RetrievalConfig
from cdic_repro.credit import build_compression_gradient_plan, build_credit_plan
from cdic_repro.memory_state import MemoryBank
from cdic_repro.model_protocol import CdicTrainingAdapter
from cdic_repro.msc import MscEpisode
from cdic_repro.retrieval import SimilarityFunction, retrieve
from cdic_repro.trace import TurnTrace
from cdic_repro.writeback import NewStatePayload, apply_write_back


@dataclass(frozen=True, slots=True)
class TrainingTurnRecord:
    turn_id: str
    query: str
    response: str
    loss: float
    loss_tokens: int
    backward_applied: bool
    trace: TurnTrace

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "response": self.response,
            "loss": self.loss,
            "loss_tokens": self.loss_tokens,
            "backward_applied": self.backward_applied,
            "trace": self.trace.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EpisodeTrainingResult:
    episode_id: str
    mean_loss: float
    turns: int
    loss_tokens: int
    backward_turns: int
    final_memory_states: int
    records: tuple[TrainingTurnRecord, ...]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "mean_loss": self.mean_loss,
            "turns": self.turns,
            "loss_tokens": self.loss_tokens,
            "backward_turns": self.backward_turns,
            "final_memory_states": self.final_memory_states,
        }


class CdicTrainingEngine:
    """Teacher-forced C-DIC episode loop with bounded retrieval-aware credit."""

    def __init__(
        self,
        model: CdicTrainingAdapter,
        similarity: SimilarityFunction,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.model = model
        self.similarity = similarity
        self.retrieval_config = retrieval_config or RetrievalConfig()

    def train_episode(self, episode: MscEpisode) -> EpisodeTrainingResult:
        memory = MemoryBank()
        loss_scale = 1.0 / len(episode.turns)
        records: list[TrainingTurnRecord] = []
        backward_turns = 0
        total_loss = 0.0
        total_loss_tokens = 0

        for turn_number, turn in enumerate(episode.turns, start=1):
            query_key = self.model.encode_query(turn.query)
            retrieval = retrieve(
                memory,
                query_key=query_key,
                turn=turn_number,
                similarity=self.similarity,
                config=self.retrieval_config,
            )
            retrieved_states = memory.select(retrieval.selected_state_ids)
            credit = build_credit_plan(retrieval)
            training_loss = self.model.response_loss(
                retrieved_states,
                turn.query,
                turn.response,
                credit,
            )
            loss_value = float(training_loss.value)
            backward_applied = training_loss.value.requires_grad
            compression_gradient_plan = build_compression_gradient_plan(
                retrieved_states,
                credit,
                gradient_window_size=self.model.gradient_window_size,
            )
            if backward_applied:
                scaled_loss = training_loss.value * loss_scale
                scaled_loss.backward(
                    retain_graph=compression_gradient_plan.retained_state_id is not None,
                )
                backward_turns += 1

            compressed = self.model.compress_gold(
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
                    graph_connected=True,
                    gradient_depth=compressed.gradient_depth,
                ),
                turn=turn_number,
            )
            trace = TurnTrace(
                turn=turn_number,
                query_id=turn.turn_id,
                retrieval=retrieval,
                credit=credit,
                write_back=write_back,
            )
            records.append(
                TrainingTurnRecord(
                    turn_id=turn.turn_id,
                    query=turn.query,
                    response=turn.response,
                    loss=loss_value,
                    loss_tokens=training_loss.token_count,
                    backward_applied=backward_applied,
                    trace=trace,
                )
            )
            total_loss += loss_value
            total_loss_tokens += training_loss.token_count

        return EpisodeTrainingResult(
            episode_id=episode.episode_id,
            mean_loss=total_loss / len(episode.turns),
            turns=len(episode.turns),
            loss_tokens=total_loss_tokens,
            backward_turns=backward_turns,
            final_memory_states=len(memory),
            records=tuple(records),
        )
