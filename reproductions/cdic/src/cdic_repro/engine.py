from __future__ import annotations

from dataclasses import dataclass

from cdic_repro.config import RetrievalConfig
from cdic_repro.credit import build_credit_plan
from cdic_repro.memory_state import MemoryBank, ThreadState
from cdic_repro.model_protocol import CdicModelAdapter
from cdic_repro.retrieval import SimilarityFunction, retrieve
from cdic_repro.trace import TurnTrace
from cdic_repro.writeback import NewStatePayload, apply_write_back


@dataclass(frozen=True, slots=True)
class TurnOutput:
    response: str
    state: ThreadState
    trace: TurnTrace


class CdicInferenceEngine:

    def __init__(
        self,
        model: CdicModelAdapter,
        similarity: SimilarityFunction,
        retrieval_config: RetrievalConfig | None = None,
        memory: MemoryBank | None = None,
    ) -> None:
        self.model = model
        self.similarity = similarity
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.memory = memory or MemoryBank()
        self._next_turn = 1

    def step(self, query: str, query_id: str | None = None) -> TurnOutput:
        turn = self._next_turn
        query_key = self.model.encode_query(query)
        retrieval = retrieve(
            self.memory,
            query_key=query_key,
            turn=turn,
            similarity=self.similarity,
            config=self.retrieval_config,
        )
        retrieved_states = self.memory.select(retrieval.selected_state_ids)
        response = self.model.generate(retrieved_states, query)
        compressed = self.model.compress(retrieved_states, query, response)
        credit = build_credit_plan(retrieval)
        write_back = apply_write_back(
            self.memory,
            retrieval=retrieval,
            payload=NewStatePayload(
                latent=compressed.latent,
                retrieval_key=compressed.retrieval_key,
                provenance=compressed.provenance,
                graph_connected=False,
                gradient_depth=compressed.gradient_depth,
            ),
            turn=turn,
        )
        trace = TurnTrace(
            turn=turn,
            query_id=query_id or f"turn-{turn:06d}",
            retrieval=retrieval,
            credit=credit,
            write_back=write_back,
        )
        self._next_turn += 1
        return TurnOutput(response=response, state=write_back.new_state, trace=trace)
