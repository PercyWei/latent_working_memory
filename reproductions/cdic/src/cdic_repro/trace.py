from __future__ import annotations

import json
from dataclasses import dataclass

from cdic_repro.credit import CreditPlan
from cdic_repro.retrieval import RetrievalResult
from cdic_repro.writeback import WriteBackResult


@dataclass(frozen=True, slots=True)
class TurnTrace:
    turn: int
    query_id: str
    retrieval: RetrievalResult
    credit: CreditPlan
    write_back: WriteBackResult

    def to_dict(self) -> dict[str, object]:
        return {
            "turn": self.turn,
            "query_id": self.query_id,
            "retrieval": {
                "scores": [
                    {
                        "state_id": score.state_id,
                        "thread_id": score.thread_id,
                        "memory_index": score.memory_index,
                        "raw_similarity": score.raw_similarity,
                        "recency_turns": score.recency_turns,
                        "decay_weight": score.decay_weight,
                        "score": score.score,
                    }
                    for score in self.retrieval.scores
                ],
                "selected_state_ids": list(self.retrieval.selected_state_ids),
                "best_state_id": self.retrieval.best_state_id,
                "peak_score": self.retrieval.peak_score,
                "on_topic": self.retrieval.on_topic,
                "used_fallback": self.retrieval.used_fallback,
            },
            "credit": {
                "connected_state_id": self.credit.connected_state_id,
                "detached_state_ids": list(self.credit.detached_state_ids),
            },
            "write_back": {
                "action": self.write_back.action.value,
                "new_state_id": self.write_back.new_state.state_id,
                "new_thread_id": self.write_back.new_state.thread_id,
                "new_revision": self.write_back.new_state.revision,
                "new_gradient_depth": self.write_back.new_state.gradient_depth,
                "replaced_state_id": self.write_back.replaced_state_id,
                "memory_before": list(self.write_back.memory_before),
                "memory_after": list(self.write_back.memory_after),
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
