from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceCosts:
    storage: float
    write: float
    read: float

    def weighted_total(
        self,
        state_weight: float,
        write_weight: float,
        read_weight: float,
    ) -> float:
        if min(state_weight, write_weight, read_weight) < 0:
            raise ValueError("resource weights must be non-negative")
        return state_weight * self.storage + write_weight * self.write + read_weight * self.read


def storage_cost(num_slots: int, reference_slots: int) -> float:
    _require_positive("num_slots", num_slots)
    _require_positive("reference_slots", reference_slots)
    return num_slots / reference_slots


def write_cost(
    next_slots: int,
    previous_slots: int,
    chunk_tokens: int,
    reference_slots: int,
) -> float:
    _require_positive("next_slots", next_slots)
    _require_positive("previous_slots", previous_slots)
    _require_positive("chunk_tokens", chunk_tokens)
    _require_positive("reference_slots", reference_slots)
    numerator = next_slots**2 + next_slots * (previous_slots + chunk_tokens)
    denominator = 3 * reference_slots**2
    return numerator / denominator


def read_cost(
    num_slots: int,
    question_tokens: int,
    answer_tokens: int,
    reference_slots: int,
) -> float:
    _require_positive("num_slots", num_slots)
    _require_positive("reference_slots", reference_slots)
    if type(question_tokens) is not int or question_tokens < 0:
        raise ValueError("question_tokens must be a non-negative integer")
    if type(answer_tokens) is not int or answer_tokens < 0:
        raise ValueError("answer_tokens must be a non-negative integer")
    sequence_tail = question_tokens + answer_tokens + 2
    return ((num_slots + sequence_tail) / (reference_slots + sequence_tail)) ** 2


def action_objective(
    gold_nll: float,
    costs: ResourceCosts,
    state_weight: float,
    write_weight: float,
    read_weight: float,
) -> float:
    if gold_nll < 0:
        raise ValueError("gold_nll must be non-negative")
    return gold_nll + costs.weighted_total(state_weight, write_weight, read_weight)


def _require_positive(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
