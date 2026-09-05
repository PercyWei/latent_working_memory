from __future__ import annotations

from cdic_repro.config import RetrievalConfig
from cdic_repro.evaluation import evaluate_msc_episodes
from cdic_repro.model_protocol import CompressedTurn, TrainingLoss
from cdic_repro.msc import MscEpisode, MscTurn


class FakeEvaluationAdapter:
    def encode_query(self, query: str) -> object:
        return float(query)

    def response_loss(
        self,
        supports: tuple[object, ...],
        query: str,
        response: str,
        credit: object,
    ) -> TrainingLoss:
        del supports, query, credit
        return TrainingLoss(value=float(response), token_count=1)

    def compress_gold(
        self,
        supports: tuple[object, ...],
        query: str,
        response: str,
    ) -> CompressedTurn:
        del supports, query
        value = float(response)
        return CompressedTurn(latent=value, retrieval_key=value)

    def loss_to_float(self, loss: object) -> float:
        return float(loss)

    def load_trainable_state_dict(self, state_dict: object, *, strict: bool = True) -> None:
        del state_dict, strict


def _episode(episode_id: str, values: tuple[tuple[str, str], ...]) -> MscEpisode:
    return MscEpisode(
        episode_id=episode_id,
        source_session_id=4,
        turns=tuple(
            MscTurn(
                turn_id=f"{episode_id}:s1:p{index}",
                query=query,
                response=response,
                session_index=1,
                pair_index=index,
            )
            for index, (query, response) in enumerate(values, start=1)
        ),
    )


def test_heldout_evaluation_reports_loss_routing_and_separation() -> None:
    episodes = (
        _episode("a", (("0.0", "0.2"), ("0.2", "0.3"))),
        _episode("b", (("1.0", "0.8"), ("0.8", "0.9"))),
    )

    result = evaluate_msc_episodes(
        FakeEvaluationAdapter(),  # type: ignore[arg-type]
        episodes,
        similarity=lambda left, right: 1.0 - abs(float(left) - float(right)),
        retrieval_config=RetrievalConfig(threshold=0.8, decay=0.0),
    )

    summary = result["summary"]
    assert isinstance(summary, dict)
    assert summary["episodes"] == 2
    assert summary["turns"] == 4
    assert summary["loss_tokens"] == 4
    assert summary["mean_turn_loss"] == 0.55
    assert summary["action_counts"] == {"initialize": 2, "replace": 2}
    assert summary["mean_final_memory_states"] == 1.0
    separation = summary["adjacent_vs_cross_episode"]
    assert isinstance(separation, dict)
    assert separation["pairwise_accuracy"] == 1.0
    assert separation["same_episode_accept_rate"] == 1.0
    assert separation["cross_episode_false_accept_rate"] == 0.0
