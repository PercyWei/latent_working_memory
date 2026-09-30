"""Target stopping, resumability and published-data cost accounting."""

import fcntl
import json
import os
from pathlib import Path

import pytest

from latent_working_memory.data_preparation.fineweb_qa import campaign
from latent_working_memory.data_preparation.fineweb_qa.annotation import STAGES
from latent_working_memory.data_preparation.fineweb_qa.storage import save_json, load_json


def setup_campaign(tmp_path, target=5):
    pool = {
        "pool_id": "pool",
        "split_counts": {"train": 6, "dev": 2, "test": 2},
        "config": {"batch_counts": {"train": 2, "dev": 1, "test": 1}},
    }
    save_json(tmp_path / "pool/source-pool.json", pool)
    prompts = {}
    for stage in STAGES:
        path = tmp_path / f"{stage}.txt"
        path.write_text(stage)
        prompts[stage] = str(path)
    template = {
        "batch_index": 0,
        "source_pool_dir": str(tmp_path / "pool"),
        "artifacts_dir": str(tmp_path / "batches/batch-000"),
        "dataset_dir": str(tmp_path / "data/batch-000"),
        "annotation": {"endpoint": "http://local"},
        "prompts": prompts,
    }
    path = tmp_path / "template.json"
    save_json(path, template)
    config = {
        "batch_template": str(path),
        "target_train_trajectories": target,
        "artifacts_dir": str(tmp_path / "campaign"),
        "dataset_dir": str(tmp_path / "collection"),
    }
    return config, template, pool


def publish(template, index, train=2):
    batch = campaign.batch_config(template, index)
    save_json(Path(batch["artifacts_dir"]) / "config.json", batch)
    save_json(
        Path(batch["dataset_dir"]) / "preparation.json",
        {
            "source_pool_id": "pool",
            "batch_index": index,
            "summary": {
                "by_split": {
                    split: {
                        "complete_documents": train if split == "train" else 1,
                        "qas": 48 * (train if split == "train" else 1),
                    }
                    for split in ("train", "dev", "test")
                }
            },
        },
    )


def fake_workers(monkeypatch, template, fail_stage=None):
    calls = []

    class Process:
        def __init__(self, command, **kwargs):
            config = load_json(Path(command[-1]))
            self.index = config["batch_index"]
            self.stage = command[4]
            calls.append((self.index, self.stage))
            self.pid = 100

        def wait(self):
            if self.stage == fail_stage:
                return 1
            if self.stage == "finalize":
                publish(template, self.index)
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", Process)
    return calls


@pytest.mark.parametrize("target", [5, 6])
def test_existing_data_counts_and_target_stops_after_whole_batch(tmp_path, monkeypatch, target):
    config, template, _ = setup_campaign(tmp_path, target)
    publish(template, 0)
    calls = fake_workers(monkeypatch, template)
    result = campaign.run(config)
    assert result["completed_by_split"]["train"]["trajectories"] == 6
    assert result["status"]["status"] == "complete"
    assert calls == [(i, s) for i in (1, 2) for s in campaign.STAGES]
    assert load_json(tmp_path / "collection/collection.json")["train_trajectories"] == 6
    assert len(load_json(tmp_path / "collection/collection.json")["batches"]) == 3
    assert campaign.run(config)["status"]["status"] == "complete"
    assert len(calls) == 10


def test_failure_stops_dispatch_and_resume_preserves_completed_batches(tmp_path, monkeypatch):
    config, template, _ = setup_campaign(tmp_path)
    publish(template, 0)
    calls = fake_workers(monkeypatch, template, "annotate")
    with pytest.raises(RuntimeError, match="batch 1 stage annotate"):
        campaign.run(config)
    assert calls == [(1, "prepare"), (1, "annotate")]
    assert load_json(tmp_path / "campaign/status.json")["status"] == "failed"
    calls = fake_workers(monkeypatch, template)
    assert campaign.run(config)["completed_by_split"]["train"]["trajectories"] == 6
    assert all(index != 0 for index, _ in calls)
    assert len(load_json(tmp_path / "campaign/status.json")["stages"]) == 12


def request_log(path, rid, input_tokens, output_tokens):
    record = {
        "stage": "generate",
        "request_id": rid,
        "attempts_total": 1,
        "network_attempts": 1,
        "network_attempt_numbers": [1],
        "network_seconds": 1,
        "cache_hit": False,
        "ok": True,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record)
        + "\n"
        + json.dumps(dict(record, network_attempts=0, network_attempt_numbers=[], cache_hit=True))
        + "\n"
    )


def test_report_separates_published_cost_from_ongoing_work_and_deduplicates_cache(tmp_path):
    config, template, _ = setup_campaign(tmp_path)
    publish(template, 0)
    request_log(tmp_path / "batches/batch-000/requests.jsonl", "r0", 100, 20)
    request_log(tmp_path / "batches/batch-001/requests.jsonl", "r1", 1000, 200)
    save_json(
        tmp_path / "batches/batch-001/documents/doc-000.json",
        {"document_index": 0, "finished": True, "assembly": {"ok": True}},
    )
    result = campaign.report(config)
    assert result["completed_by_split"]["train"]["trajectories"] == 2
    assert result["remaining_train_trajectories"] == 3
    assert result["completed_batches_usage"]["total_tokens"] == 120
    assert result["all_batches_usage"]["total_tokens"] == 1320
    assert result["all_batches_usage"]["network_attempts"] == 2
    assert result["completed_batch_tokens_per_train_trajectory"] == 60
    assert result["batches"][1]["annotation_quota_complete_documents"] == 1
    assert not result["batches"][1]["published"]
    assert not (tmp_path / "campaign/status.json").exists()


def test_foreign_pool_is_never_counted(tmp_path):
    config, template, _ = setup_campaign(tmp_path)
    publish(template, 0)
    p = tmp_path / "data/batch-000/preparation.json"
    data = load_json(p)
    data["source_pool_id"] = "foreign"
    save_json(p, data)
    with pytest.raises(ValueError, match="source pool"):
        campaign.report(config)


def test_duplicate_controller_cannot_start_or_overwrite_status(tmp_path, monkeypatch):
    config, template, _ = setup_campaign(tmp_path)
    root = tmp_path / "campaign"
    root.mkdir()
    calls = fake_workers(monkeypatch, template)
    with (root / "campaign.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            campaign.run(config)
    assert not calls and not (root / "status.json").exists()


def test_frozen_campaign_config_rejects_mutation(tmp_path, monkeypatch):
    config, template, _ = setup_campaign(tmp_path, target=2)
    publish(template, 0)
    fake_workers(monkeypatch, template)
    campaign.run(config)
    config["target_train_trajectories"] = 3
    with pytest.raises(ValueError, match="configuration changed"):
        campaign.run(config)


def test_resume_refuses_a_still_live_child(tmp_path, monkeypatch):
    config, template, _ = setup_campaign(tmp_path)
    save_json(
        tmp_path / "campaign/status.json",
        {"status": "failed", "stages": [], "stage_pid": os.getpid()},
    )
    calls = fake_workers(monkeypatch, template)
    with pytest.raises(RuntimeError, match="previous stage process still exists"):
        campaign.run(config)
    assert calls == []
