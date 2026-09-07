from __future__ import annotations

import pytest

from latent_working_memory.v1.capacity import (
    ResourceCosts,
    action_objective,
    read_cost,
    storage_cost,
    write_cost,
)


def test_reference_resource_costs_are_one() -> None:
    assert storage_cost(64, 64) == 1.0
    assert write_cost(64, 64, 64, 64) == 1.0
    assert read_cost(64, 12, 4, 64) == 1.0


def test_action_objective_keeps_quality_and_resources_separate() -> None:
    costs = ResourceCosts(storage=2.0, write=3.0, read=4.0)
    assert costs.weighted_total(0.5, 0.25, 0.125) == pytest.approx(2.25)
    assert action_objective(1.5, costs, 0.5, 0.25, 0.125) == pytest.approx(3.75)
