from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from cdic_repro.config import RetrievalConfig
from cdic_repro.credit import CreditPlan
from cdic_repro.memory_state import ThreadState
from cdic_repro.model_protocol import CompressedTurn, TrainingLoss
from cdic_repro.msc import MscEpisode, MscTurn
from cdic_repro.training import CdicTrainingEngine
from cdic_repro.writeback import WriteAction


def cosine(left: object, right: object) -> float:
    left_vector = tuple(float(value) for value in left)  # type: ignore[union-attr]
    right_vector = tuple(float(value) for value in right)  # type: ignore[union-attr]
    numerator = sum(a * b for a, b in zip(left_vector, right_vector, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    return numerator / (left_norm * right_norm)


class FakeLoss:
    def __init__(self, value: float, requires_grad: bool) -> None:
        self.value = value
        self.requires_grad = requires_grad


class FakeTrainingAdapter:
    def __init__(self) -> None:
        self.keys = {
            "a1": (1.0, 0.0),
            "a2": (0.99, 0.01),
            "b1": (0.0, 1.0),
        }
        self.loss_calls: list[tuple[tuple[str, ...], str, CreditPlan]] = []
        self.compression_responses: list[str] = []
        self.backward_scales: list[float] = []

    def encode_query(self, query: str) -> object:
        return self.keys[query]

    def response_loss(
        self,
        supports: tuple[ThreadState, ...],
        query: str,
        response: str,
        credit: CreditPlan,
    ) -> TrainingLoss:
        self.loss_calls.append((tuple(state.state_id for state in supports), response, credit))
        return TrainingLoss(
            value=FakeLoss(
                float(len(response)), requires_grad=credit.connected_state_id is not None
            ),
            token_count=len(response),
        )

    def compress_gold(
        self,
        supports: tuple[ThreadState, ...],
        query: str,
        response: str,
    ) -> CompressedTurn:
        self.compression_responses.append(response)
        return CompressedTurn(
            latent=f"latent:{query}:{response}",
            retrieval_key=self.keys[query],
            provenance=("gold-response",),
        )

    def backward(self, loss: object, scale: float) -> None:
        assert isinstance(loss, FakeLoss)
        self.backward_scales.append(scale)

    def loss_requires_grad(self, loss: object) -> bool:
        assert isinstance(loss, FakeLoss)
        return loss.requires_grad

    def loss_to_float(self, loss: object) -> float:
        assert isinstance(loss, FakeLoss)
        return loss.value

    def trainable_parameters(self) -> Iterable[object]:
        return ()

    def trainable_state_dict(self) -> Mapping[str, object]:
        return {}

    def load_trainable_state_dict(
        self,
        state_dict: Mapping[str, object],
        strict: bool = True,
    ) -> None:
        assert not state_dict
        assert strict


def make_episode() -> MscEpisode:
    return MscEpisode(
        episode_id="episode-1",
        source_session_id=4,
        turns=(
            MscTurn("turn-1", "a1", "gold-1", 1, 1),
            MscTurn("turn-2", "a2", "gold-2", 1, 2),
            MscTurn("turn-3", "b1", "gold-3", 2, 1),
        ),
    )


def test_training_engine_uses_gold_responses_and_one_hop_credit() -> None:
    adapter = FakeTrainingAdapter()
    engine = CdicTrainingEngine(
        model=adapter,
        similarity=cosine,
        retrieval_config=RetrievalConfig(threshold=0.8, decay=0.0),
    )

    result = engine.train_episode(make_episode())

    assert adapter.compression_responses == ["gold-1", "gold-2", "gold-3"]
    assert adapter.loss_calls[0][0] == ()
    assert adapter.loss_calls[1][2].connected_state_id == "state-000001"
    assert adapter.loss_calls[2][2].connected_state_id is None
    assert adapter.backward_scales == [1.0 / 3.0]
    assert result.backward_turns == 1
    assert [record.trace.write_back.action for record in result.records] == [
        WriteAction.INITIALIZE,
        WriteAction.REPLACE,
        WriteAction.INSERT,
    ]
    assert all(record.trace.write_back.new_state.graph_connected for record in result.records)
