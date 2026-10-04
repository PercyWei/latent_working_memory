"""Flat publication, local recovery and explicit raw-source exclusion."""

import fcntl
import json
import os
import shutil
from pathlib import Path

import pytest

from latent_working_memory.data_preparation.fineweb_qa import campaign, publication
from latent_working_memory.data_preparation.fineweb_qa.annotation import STAGES
from latent_working_memory.data_preparation.fineweb_qa.storage import save_json, load_json


def setup_campaign(tmp_path, target=5):
    recipe = {
        "source": {"dataset": "fixture", "scan_documents": 10},
        "window": {},
        "batch_counts": {"train": 2, "dev": 1, "test": 1},
    }
    documents = []
    for split, count in (("train", 6), ("dev", 2), ("test", 2)):
        for i in range(count):
            documents.append(
                {
                    "pool_index": len(documents),
                    "document_id": f"{split}-{i}",
                    "dedup_cluster": f"example/{split}/{i}",
                    "split": split,
                    "trajectory_id": f"{split}-{i}:0:100",
                    "segments": [{}] * 6,
                    "source": {"file": "raw.parquet", "row_group": 0, "row_index": len(documents)},
                }
            )
    pool = {
        "pool_id": "pool",
        "config": recipe,
        "excluded_sources": [],
        "previous_datasets": [],
        "documents": documents,
        "statistics": {"frozen_documents": 10},
        "split_counts": {"train": 6, "dev": 2, "test": 2},
    }
    save_json(tmp_path / "dataset/source-pool.json", pool)
    save_json(tmp_path / "pool-config.json", dict(recipe, pool_dir="ignored"))
    prompts = {}
    for stage in STAGES:
        path = tmp_path / f"{stage}.txt"
        path.write_text(stage)
        prompts[stage] = str(path)
    raw_template = {
        "batch_index": 0,
        "source_pool_dir": "ignored",
        "qa": {},
        "artifacts_dir": "ignored",
        "dataset_dir": "ignored",
        "cache_dir": "ignored",
        "annotation": {"endpoint": "http://local"},
        "prompts": prompts,
    }
    save_json(tmp_path / "template.json", raw_template)
    config = {
        "batch_template": str(tmp_path / "template.json"),
        "source_pool_config": str(tmp_path / "pool-config.json"),
        "target_train_trajectories": target,
        "previous_datasets": [],
        "artifacts_dir": str(tmp_path / "campaign"),
        "dataset_dir": str(tmp_path / "dataset"),
    }
    template, _ = campaign._inputs(config)
    return config, template, pool


def publish(template, index, train=2):
    batch = campaign.batch_config(template, index)
    pool = load_json(Path(template["source_pool_dir"]) / "source-pool.json")
    ranges = campaign.batch_ranges(pool, template["source_offsets"], index)
    save_json(Path(batch["artifacts_dir"]) / "config.json", batch)
    dataset = Path(batch["dataset_dir"])
    dataset.mkdir(parents=True, exist_ok=True)
    counts, outcomes = {}, []
    for split, bounds in ranges.items():
        candidates = [d for d in pool["documents"] if d["split"] == split][
            bounds["start"] : bounds["stop"]
        ]
        successful = candidates[:train] if split == "train" else candidates
        rows = [dict(d, qas=[{}] * 48) for d in successful]
        (dataset / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        counts[split] = {
            "frozen_documents": len(candidates),
            "complete_documents": len(rows),
            "qas": 48 * len(rows),
        }
        for d in candidates:
            outcomes.append(
                {
                    "document_index": len(outcomes),
                    "trajectory_id": d["trajectory_id"],
                    "failure_reason": None if d in successful else "quota_shortfall",
                }
            )
    save_json(
        dataset / "preparation.json",
        {
            "source_pool_id": pool["pool_id"],
            "batch_index": index,
            "batch_ranges": ranges,
            "summary": {"by_split": counts},
            "stage_counts": {},
            "documents": outcomes,
            "supplementation": {},
            "prompts": {s: s for s in STAGES},
        },
    )


def fake_workers(monkeypatch, template, fail_stage=None):
    calls = []

    class Process:
        def __init__(self, command, **kwargs):
            config = load_json(Path(command[-1]))
            self.index, self.stage = config["batch_index"], command[4]
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
def test_target_stops_after_whole_batch_and_publishes_flat_dataset(tmp_path, monkeypatch, target):
    config, template, _ = setup_campaign(tmp_path, target)
    publish(template, 0)
    calls = fake_workers(monkeypatch, template)
    result = campaign.run(config)
    assert result["completed_by_split"]["train"]["trajectories"] == 6
    assert result["status"]["status"] == "complete" and result["dataset_published"]
    assert calls == [(i, s) for i in (1, 2) for s in campaign.STAGES]
    dataset = tmp_path / "dataset"
    assert {p.name for p in dataset.iterdir()} == {
        "source-pool.json",
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "preparation.json",
    }
    metadata = load_json(dataset / "preparation.json")
    assert len(metadata["used_sources"]) == 10
    before = (dataset / "train.jsonl").read_bytes()
    expected = b"".join(
        (tmp_path / f"campaign/batches/batch-{i:03d}/dataset/train.jsonl").read_bytes()
        for i in range(3)
    )
    assert before == expected
    assert campaign.run(config)["status"]["status"] == "complete"
    assert (dataset / "train.jsonl").read_bytes() == before and len(calls) == 2 * len(
        campaign.STAGES
    )


def test_failure_resume_preserves_batches_without_publishing_partial_dataset(tmp_path, monkeypatch):
    config, template, _ = setup_campaign(tmp_path)
    publish(template, 0)
    calls = fake_workers(monkeypatch, template, "annotate")
    with pytest.raises(RuntimeError, match="batch 1 stage annotate"):
        campaign.run(config)
    assert calls == [(1, "prepare"), (1, "annotate")]
    assert not (tmp_path / "dataset/preparation.json").exists()
    calls = fake_workers(monkeypatch, template)
    assert campaign.run(config)["completed_by_split"]["train"]["trajectories"] == 6
    assert all(index != 0 for index, _ in calls)


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


def test_report_separates_finalized_batches_publication_and_ongoing_cost(tmp_path):
    config, template, _ = setup_campaign(tmp_path)
    publish(template, 0)
    request_log(tmp_path / "campaign/batches/batch-000/requests.jsonl", "r0", 100, 20)
    request_log(tmp_path / "campaign/batches/batch-001/requests.jsonl", "r1", 1000, 200)
    result = campaign.report(config)
    assert result["completed_batches_usage"]["total_tokens"] == 120
    assert result["all_batches_usage"]["total_tokens"] == 1320
    assert not result["dataset_published"] and not (tmp_path / "campaign/status.json").exists()


def test_report_uses_frozen_contract_and_counts_every_recorded_request_stage(tmp_path, monkeypatch):
    config, template, _ = setup_campaign(tmp_path, target=2)
    publish(template, 0)
    fake_workers(monkeypatch, template)
    campaign.run(config)
    contract_path = tmp_path / "campaign/config.json"
    frozen = load_json(contract_path)
    frozen["batch_template"]["prompts"]["archived_check"] = "unavailable.txt"
    save_json(contract_path, frozen)
    batch_path = tmp_path / "campaign/batches/batch-000/config.json"
    batch = load_json(batch_path)
    batch["prompts"]["archived_check"] = "unavailable.txt"
    save_json(batch_path, batch)
    log = tmp_path / "campaign/batches/batch-000/requests.jsonl"
    request_log(log, "recorded", 123, 45)
    records = [
        dict(json.loads(line), stage="archived_check") for line in log.read_text().splitlines()
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in records))
    Path(config["batch_template"]).unlink()
    Path(config["source_pool_config"]).unlink()
    report = campaign.report(config)
    assert report["completed_by_split"]["train"]["trajectories"] == 2
    assert report["all_batches_usage"]["total_tokens"] == 168


def test_append_excludes_failed_sources_and_resumes_without_previous_directory(
    tmp_path, monkeypatch
):
    config, template, pool = setup_campaign(tmp_path, target=1)
    publish(template, 0, train=1)
    fake_workers(monkeypatch, template)
    campaign.run(config)
    previous = tmp_path / "dataset"
    first = load_json(previous / "preparation.json")
    assert len(first["used_sources"]) == 4  # Includes the failed train source.
    appended = dict(
        config,
        previous_datasets=[str(previous)],
        artifacts_dir=str(tmp_path / "next"),
        dataset_dir=str(tmp_path / "next-data"),
    )
    new_pool = campaign.initialize(appended)
    assert new_pool["pool_id"] != pool["pool_id"]
    assert new_pool["split_counts"] == {"train": 4, "dev": 1, "test": 1}
    first_ids = {d["document_id"] for d in first["used_sources"]}
    assert first_ids.isdisjoint(d["document_id"] for d in new_pool["documents"])
    second_template, _ = campaign._inputs(appended)
    shutil.rmtree(previous)
    fake_workers(monkeypatch, second_template, "annotate")
    with pytest.raises(RuntimeError, match="annotate"):
        campaign.run(appended)
    calls = fake_workers(monkeypatch, second_template)
    result = campaign.run(appended)
    assert result["completed_by_split"]["train"]["trajectories"] == 2
    assert calls == [(0, stage) for stage in campaign.STAGES]
    second = load_json(tmp_path / "next-data/preparation.json")
    assert len(second["used_sources"]) == 4
    assert first_ids.isdisjoint(d["document_id"] for d in second["used_sources"])
    third = dict(
        appended,
        previous_datasets=[str(tmp_path / "next-data")],
        artifacts_dir=str(tmp_path / "third"),
        dataset_dir=str(tmp_path / "third-data"),
    )

    def rebuild(recipe, excluded, previous_dirs):
        assert {d["document_id"] for d in excluded} == {
            d["document_id"] for d in second["used_sources"]
        }
        rebuilt = campaign.pool_after_exclusions(pool, excluded, previous_dirs)
        save_json(Path(recipe["pool_dir"]) / "source-pool.json", rebuilt)
        return rebuilt

    monkeypatch.setattr(campaign, "prepare_pool", rebuild)
    third_pool = campaign.initialize(third)
    assert third_pool["split_counts"] == {"train": 4, "dev": 1, "test": 1}
    assert first_ids <= {d["document_id"] for d in third_pool["documents"]}


def test_fresh_initialization_builds_local_pool_and_roots_override_template(tmp_path, monkeypatch):
    config, template, pool = setup_campaign(tmp_path, target=2)
    (tmp_path / "dataset/source-pool.json").unlink()

    def build(recipe):
        assert recipe["pool_dir"] == config["dataset_dir"]
        save_json(Path(recipe["pool_dir"]) / "source-pool.json", pool)
        return pool

    monkeypatch.setattr(campaign, "prepare_pool", build)
    fake_workers(monkeypatch, template)
    campaign.run(config)
    batch = load_json(tmp_path / "campaign/batch-configs/batch-000.json")
    assert batch["dataset_dir"] == str(tmp_path / "campaign/batches/batch-000/dataset")
    assert batch["source_pool_dir"] == config["dataset_dir"]
    assert batch["cache_dir"] == str(tmp_path / "campaign/requests")


def test_multiple_datasets_merge_only_their_own_sources_and_deduplicate(tmp_path, monkeypatch):
    config, template, _ = setup_campaign(tmp_path, target=2)
    publish(template, 0)
    fake_workers(monkeypatch, template)
    campaign.run(config)
    first_dir = Path(config["dataset_dir"])
    first = load_json(first_dir / "preparation.json")
    second_config = dict(
        config,
        previous_datasets=[str(first_dir)],
        artifacts_dir=str(tmp_path / "second-run"),
        dataset_dir=str(tmp_path / "second-data"),
    )
    campaign.initialize(second_config)
    second_template, _ = campaign._inputs(second_config)
    fake_workers(monkeypatch, second_template)
    campaign.run(second_config)
    second_dir = Path(second_config["dataset_dir"])
    second = load_json(second_dir / "preparation.json")
    assert len(first["used_sources"]) == len(second["used_sources"]) == 4
    first_ids = {s["document_id"] for s in first["used_sources"]}
    second_ids = {s["document_id"] for s in second["used_sources"]}
    assert first_ids.isdisjoint(second_ids)
    third_config = dict(
        config,
        previous_datasets=[str(first_dir), str(second_dir), str(first_dir)],
        artifacts_dir=str(tmp_path / "third-run"),
        dataset_dir=str(tmp_path / "third-data"),
    )
    third_pool = campaign.initialize(third_config)
    assert {s["document_id"] for s in third_pool["excluded_sources"]} == first_ids | second_ids
    assert len(third_pool["excluded_sources"]) == 8
    assert third_pool["split_counts"] == {"train": 2, "dev": 0, "test": 0}


@pytest.mark.parametrize("previous", [None, "dataset", [None], [""]])
def test_previous_datasets_requires_explicit_directory_list(tmp_path, previous):
    config, _, _ = setup_campaign(tmp_path)
    config["previous_datasets"] = previous
    with pytest.raises(ValueError, match="list of dataset directories"):
        campaign.initialize(config)


def test_expanded_scan_passes_all_used_raw_sources_to_builder(tmp_path, monkeypatch):
    config, template, _ = setup_campaign(tmp_path, target=1)
    publish(template, 0, train=1)
    fake_workers(monkeypatch, template)
    campaign.run(config)
    recipe = load_json(Path(config["source_pool_config"]))
    recipe["source"]["scan_documents"] = 20
    save_json(tmp_path / "expanded.json", recipe)
    second = dict(
        config,
        previous_datasets=[str(tmp_path / "dataset")],
        dataset_dir=str(tmp_path / "second"),
        source_pool_config=str(tmp_path / "expanded.json"),
    )

    def build(recipe, excluded, previous_dirs):
        assert len(excluded) == 4 and previous_dirs == [config["dataset_dir"]]
        assert recipe["source"]["scan_documents"] == 20
        return {"checked": True}

    monkeypatch.setattr(campaign, "prepare_pool", build)
    assert campaign.initialize(second) == {"checked": True}


def test_publication_rejects_duplicate_rows_preserving_previous_valid_files(tmp_path):
    config, template, _ = setup_campaign(tmp_path, target=2)
    publish(template, 0)
    campaign.finalize(config)
    marker = tmp_path / "dataset/preparation.json"
    before = marker.read_bytes()
    staged = tmp_path / "campaign/batches/batch-000/dataset/train.jsonl"
    lines = staged.read_text().splitlines(True)
    staged.write_text(lines[0] + lines[0])
    with pytest.raises(ValueError, match="duplicate"):
        campaign.finalize(config)
    assert marker.read_bytes() == before
    assert not list((tmp_path / "dataset").glob(".*.tmp"))


def test_publication_interruption_has_no_completion_marker_and_can_resume(tmp_path, monkeypatch):
    config, template, _ = setup_campaign(tmp_path, target=2)
    publish(template, 0)
    real_replace = publication.os.replace

    def fail_dev(src, dst):
        if Path(dst).name == "dev.jsonl":
            raise OSError("interrupted export")
        return real_replace(src, dst)

    monkeypatch.setattr(publication.os, "replace", fail_dev)
    with pytest.raises(OSError, match="interrupted"):
        campaign.finalize(config)
    assert not (tmp_path / "dataset/preparation.json").exists()
    monkeypatch.setattr(publication.os, "replace", real_replace)
    assert campaign.finalize(config)["summary"]["complete_documents"] == 4


@pytest.mark.parametrize(
    "field,value,error",
    [("source_pool_id", "foreign", "source pool"), ("batch_ranges", {}, "source ranges")],
)
def test_inconsistent_batches_are_not_counted(tmp_path, field, value, error):
    config, template, _ = setup_campaign(tmp_path)
    publish(template, 0)
    path = tmp_path / "campaign/batches/batch-000/dataset/preparation.json"
    altered = load_json(path)
    altered[field] = value
    save_json(path, altered)
    with pytest.raises(ValueError, match=error):
        campaign.report(config)


def test_duplicate_controller_and_live_child_are_rejected(tmp_path, monkeypatch):
    config, template, _ = setup_campaign(tmp_path)
    root = tmp_path / "campaign"
    root.mkdir()
    calls = fake_workers(monkeypatch, template)
    with (root / "campaign.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            campaign.run(config)
    assert not calls
    save_json(root / "status.json", {"status": "failed", "stages": [], "stage_pid": os.getpid()})
    monkeypatch.setattr(
        campaign,
        "initialize",
        lambda *_: pytest.fail("live child must be checked before initialization"),
    )
    with pytest.raises(RuntimeError, match="previous stage process"):
        campaign.run(config)


def test_frozen_configuration_and_self_reference_are_rejected(tmp_path, monkeypatch):
    config, template, _ = setup_campaign(tmp_path, target=2)
    publish(template, 0)
    fake_workers(monkeypatch, template)
    campaign.run(config)
    config["target_train_trajectories"] = 3
    with pytest.raises(ValueError, match="configuration changed"):
        campaign.run(config)
    config["previous_datasets"] = [config["dataset_dir"]]
    with pytest.raises(ValueError, match="new dataset"):
        campaign.initialize(config)
    config["previous_datasets"] = [str(tmp_path / "unrelated")]
    with pytest.raises(ValueError, match="snapshot configuration"):
        campaign.initialize(config)
