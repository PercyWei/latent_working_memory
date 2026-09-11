from __future__ import annotations

import json
import random
import socket

import pytest
import torch

from cdic_repro.experiments.tracking import swanlab_run


def test_offline_swanlab_run_records_identity_without_changing_rng(tmp_path, monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("offline tracking must not access the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    random.seed(42)
    torch.manual_seed(42)
    python_state = random.getstate()
    torch_state = torch.get_rng_state()

    with swanlab_run(
        tmp_path,
        {"threshold": 0.8},
        mode="offline",
        group="cdic-msc-test",
        tags=("scope:reproduction", "method:cdic", "data:msc"),
        job_type="train",
    ) as run:
        assert run is not None
        run.log({"train/mean_turn_nll": 2.0}, step=1)

    identity = json.loads((tmp_path / "swanlab.json").read_text(encoding="utf-8"))
    assert identity["mode"] == "offline"
    assert identity["project"] == "latent-working-memory-cdic-repro"
    assert identity["group"] == "cdic-msc-test"
    assert identity["job_type"] == "train"
    assert identity["tags"] == ["data:msc", "method:cdic", "scope:reproduction"]
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_disabled_swanlab_run_has_no_artifacts(tmp_path):
    with swanlab_run(tmp_path, {}, mode="disabled") as run:
        assert run is None
    assert list(tmp_path.iterdir()) == []


def test_enabled_swanlab_run_requires_organization_metadata(tmp_path):
    with pytest.raises(ValueError, match="require a group"):
        with swanlab_run(tmp_path, {}, mode="offline"):
            pass
