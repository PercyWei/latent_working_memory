from __future__ import annotations

import json
import math

from cdic_repro.config import RetrievalConfig
from cdic_repro.engine import CdicInferenceEngine
from cdic_repro.memory_state import ThreadState
from cdic_repro.model_protocol import CompressedTurn
from cdic_repro.writeback import WriteAction


def cosine(left: object, right: object) -> float:
    left_vector = tuple(float(value) for value in left)  # type: ignore[union-attr]
    right_vector = tuple(float(value) for value in right)  # type: ignore[union-attr]
    numerator = sum(a * b for a, b in zip(left_vector, right_vector, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    return numerator / (left_norm * right_norm)


class FakeAdapter:
    def __init__(self) -> None:
        self.query_keys = {
            "topic-a-1": (1.0, 0.0),
            "topic-a-2": (0.99, 0.01),
            "topic-b": (0.0, 1.0),
        }
        self.retrieved_state_history: list[tuple[str, ...]] = []

    def encode_query(self, query: str) -> object:
        return self.query_keys[query]

    def generate(self, retrieved_states: tuple[ThreadState, ...], query: str) -> str:
        self.retrieved_state_history.append(
            tuple(state.state_id for state in retrieved_states)
        )
        return f"response:{query}"

    def compress(
        self,
        retrieved_states: tuple[ThreadState, ...],
        query: str,
        response: str,
    ) -> CompressedTurn:
        assert response == f"response:{query}"
        return CompressedTurn(
            latent=f"latent:{query}",
            retrieval_key=self.query_keys[query],
            provenance=(query,),
        )


def test_inference_engine_runs_retrieve_generate_compress_writeback() -> None:
    adapter = FakeAdapter()
    engine = CdicInferenceEngine(
        model=adapter,
        similarity=cosine,
        retrieval_config=RetrievalConfig(threshold=0.8, decay=0.0),
    )

    first = engine.step("topic-a-1")
    second = engine.step("topic-a-2")
    third = engine.step("topic-b")

    assert first.trace.write_back.action is WriteAction.INITIALIZE
    assert second.trace.write_back.action is WriteAction.REPLACE
    assert second.state.thread_id == first.state.thread_id
    assert second.state.revision == 1
    assert third.trace.write_back.action is WriteAction.INSERT
    assert len(engine.memory) == 2
    assert adapter.retrieved_state_history[0] == ()
    assert adapter.retrieved_state_history[1] == (first.state.state_id,)
    assert adapter.retrieved_state_history[2] == (second.state.state_id,)

    serialized = json.loads(third.trace.to_json())
    assert serialized["write_back"]["action"] == "insert"
    assert serialized["retrieval"]["used_fallback"] is True
