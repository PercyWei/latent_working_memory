from __future__ import annotations

import math

from cdic_repro.config import RetrievalConfig
from cdic_repro.credit import CreditPlan, build_compression_gradient_plan
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
    def __init__(
        self,
        value: float,
        requires_grad: bool,
        backward_scales: list[float],
        backward_retain_graph: list[bool],
    ) -> None:
        self.value = value
        self.requires_grad = requires_grad
        self.backward_scales = backward_scales
        self.backward_retain_graph = backward_retain_graph
        self.scale = 1.0

    def __float__(self) -> float:
        return self.value

    def __mul__(self, scale: float) -> FakeLoss:
        self.scale = scale
        return self

    def backward(self, retain_graph: bool = False) -> None:
        self.backward_scales.append(self.scale)
        self.backward_retain_graph.append(retain_graph)


class FakeTrainingAdapter:
    def __init__(self, gradient_window_size: int = 1) -> None:
        self.keys = {
            "a1": (1.0, 0.0),
            "a2": (0.99, 0.01),
            "a3": (0.98, 0.02),
            "b1": (0.0, 1.0),
        }
        self._gradient_window_size = gradient_window_size
        self.loss_calls: list[tuple[tuple[str, ...], str, CreditPlan]] = []
        self.compression_responses: list[str] = []
        self.backward_scales: list[float] = []
        self.backward_retain_graph: list[bool] = []

    @property
    def gradient_window_size(self) -> int:
        return self._gradient_window_size

    def encode_query(self, query: str) -> object:
        return self.keys[query]

    def response_loss(
        self,
        retrieved_states: tuple[ThreadState, ...],
        query: str,
        response: str,
        credit: CreditPlan,
    ) -> TrainingLoss:
        self.loss_calls.append(
            (tuple(state.state_id for state in retrieved_states), response, credit)
        )
        return TrainingLoss(
            value=FakeLoss(
                float(len(response)),
                requires_grad=credit.connected_state_id is not None,
                backward_scales=self.backward_scales,
                backward_retain_graph=self.backward_retain_graph,
            ),
            token_count=len(response),
        )

    def compress_gold(
        self,
        retrieved_states: tuple[ThreadState, ...],
        query: str,
        response: str,
        credit: CreditPlan,
    ) -> CompressedTurn:
        self.compression_responses.append(response)
        gradient_plan = build_compression_gradient_plan(
            retrieved_states,
            credit,
            gradient_window_size=self.gradient_window_size,
        )
        return CompressedTurn(
            latent=f"latent:{query}:{response}",
            retrieval_key=self.keys[query],
            provenance=("gold-response",),
            gradient_depth=gradient_plan.new_state_gradient_depth,
        )

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
    assert adapter.backward_retain_graph == [False]
    assert result.backward_turns == 1
    assert [record.trace.write_back.action for record in result.records] == [
        WriteAction.INITIALIZE,
        WriteAction.REPLACE,
        WriteAction.INSERT,
    ]
    assert all(record.trace.write_back.new_state.graph_connected for record in result.records)


def test_training_engine_retains_graph_only_inside_gradient_window() -> None:
    adapter = FakeTrainingAdapter(gradient_window_size=2)
    episode = MscEpisode(
        episode_id="episode-window",
        source_session_id=4,
        turns=(
            MscTurn("turn-1", "a1", "gold-1", 1, 1),
            MscTurn("turn-2", "a2", "gold-2", 1, 2),
            MscTurn("turn-3", "a3", "gold-3", 1, 3),
        ),
    )
    engine = CdicTrainingEngine(
        model=adapter,
        similarity=cosine,
        retrieval_config=RetrievalConfig(threshold=0.8, decay=0.0),
    )

    result = engine.train_episode(episode)

    assert adapter.backward_retain_graph == [True, False]
    assert [record.trace.write_back.new_state.gradient_depth for record in result.records] == [
        1,
        2,
        1,
    ]
