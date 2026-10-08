"""Exact split quotas, source replenishment and recovery with real local Parquet."""

import copy
import fcntl
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latent_working_memory.data_preparation.fineweb_factqa import campaign, pipeline, publication
from latent_working_memory.data_preparation.fineweb_factqa import __main__ as stage_cli
from latent_working_memory.data_preparation.fineweb_factqa.config import load_config, run_config
from latent_working_memory.data_preparation.fineweb_factqa.storage import load_json, save_json
from latent_working_memory.data_preparation.fineweb_source import load_previous_sources
from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig
from latent_working_memory.data_preparation.fineweb_multisegment.prepare import prepare_dataset
from latent_working_memory.data_preparation.segmentation import SegmentationConfig


def case(tmp_path, counts=None, size=40, source_batch_size=4):
    source = tmp_path / "raw"
    source.mkdir()
    records = [
        {
            "id": f"document-{i}",
            "url": f"https://example.org/document-{i}",
            "text": " ".join(f"word{i}_{j}" for j in range(90)),
        }
        for i in range(size)
    ]
    pq.write_table(pa.Table.from_pylist(records), source / "000_00000.parquet", row_group_size=5)
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for stage in ("generate", "verify", "document_review"):
        text = "Answer at most {{max_answer_chars}} characters" if stage == "generate" else stage
        (prompts / f"{stage}.txt").write_text(text)
    config = {
        "source_dir": str(source),
        "source_batch_size": source_batch_size,
        "source_seed": 37,
        "selection_seed": 19,
        "split_counts": counts or {"train": 3, "dev": 1, "test": 1},
        "window": {
            "capacity": 8,
            "min_segment_ratio": 1,
            "max_segment_ratio": 2,
            "min_segments": 2,
            "max_segments": 3,
            "content_reserve_ratio": 1.5,
        },
        "batch_split_counts": {"train": 2, "dev": 1, "test": 1},
        "qa": {
            "role_seed": 29,
            "max_answer_chars": 128,
            "max_supplement_rounds": 3,
            "supplement_surplus": 2,
        },
        "annotation": {"endpoint": "http://local", "model": "fixture"},
        "prompts_dir": str(prompts),
    }
    path = tmp_path / "config.json"
    save_json(path, config)
    return run_config(load_config(path), tmp_path / "data", tmp_path / "runs", "test"), path


def workers(monkeypatch, fail_stage=None, reject_first=False):
    calls, rejected = [], set()

    class Process:
        pid = 100

        def __init__(self, command, **kwargs):
            assert command[3] == "latent_working_memory.data_preparation.fineweb_factqa"
            self.stage = command[4]
            self.config = load_json(Path(command[-1]))
            calls.append((self.config["batch_index"], self.stage, self.config))

        def wait(self):
            if self.stage == fail_stage:
                return 1
            cfg = self.config
            root = Path(cfg["artifacts_dir"])
            if self.stage == "prepare":
                pipeline.prepare(cfg)
            elif self.stage == "finalize":
                selection = load_json(root / "selection.json")
                if reject_first and not rejected:
                    rejected.add(selection["documents"][0]["document_id"])
                target = Path(cfg["dataset_dir"])
                target.mkdir(parents=True)
                counts, outcomes = {}, []
                for split in ("train", "dev", "test"):
                    selected = [doc for doc in selection["documents"] if doc["split"] == split]
                    rows = [
                        dict(doc, qas=[{}])
                        for doc in selected
                        if doc["document_id"] not in rejected
                    ]
                    (target / f"{split}.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows)
                    )
                    counts[split] = {
                        "frozen_documents": len(selected),
                        "complete_documents": len(rows),
                        "qas": len(rows),
                    }
                    outcomes.extend(
                        {
                            "document_index": len(outcomes) + i,
                            "trajectory_id": doc["trajectory_id"],
                            "failure_reason": "quota_shortfall"
                            if doc["document_id"] in rejected
                            else None,
                        }
                        for i, doc in enumerate(selected)
                    )
                save_json(
                    target / "preparation.json",
                    {
                        "source_pool_id": selection["source_pool_id"],
                        "batch_index": cfg["batch_index"],
                        "batch_ranges": selection["ranges"],
                        "summary": {"by_split": counts},
                        "stage_counts": {},
                        "documents": outcomes,
                        "supplementation": {},
                        "prompts": load_json(root / "prompts.json"),
                    },
                )
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", Process)
    return calls, rejected


def test_repository_configs_and_prompt_limits():
    root = Path("configs/data_preparation/fineweb-factqa")
    for path in root.glob("*.json"):
        config = load_config(path)
        assert config["split_counts"] == {"train": 1000, "dev": 100, "test": 100}
        prompts = {
            stage: str(Path(config["prompts_dir"]) / f"{stage}.txt")
            for stage in ("generate", "verify", "document_review")
        }
        texts = pipeline._prompt_texts({"qa": config["qa"], "prompts": prompts})
        assert "{{max_answer_chars}}" not in texts["generate"]
        assert "128" in texts["generate"]
        assert not {"dataset_dir", "artifacts_dir", "previous_datasets"} & config.keys()


@pytest.mark.parametrize(
    "field",
    [
        "scan_documents",
        "old_pool_documents",
        "source_glob",
        "split_fractions",
        "target_train_trajectories",
        "batch_counts",
    ],
)
def test_obsolete_user_fields_are_rejected(tmp_path, field):
    _, path = case(tmp_path)
    config = load_json(path)
    config[field] = 1
    save_json(path, config)
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_answer_limit_is_validated_before_sources(tmp_path, limit):
    _, path = case(tmp_path)
    config = load_json(path)
    config["qa"]["max_answer_chars"] = limit
    save_json(path, config)
    with pytest.raises(ValueError, match="max_answer_chars"):
        load_config(path)


def test_exact_quotas_replenish_after_rejected_document_and_record_only_used_sources(
    tmp_path, monkeypatch
):
    config, _ = case(tmp_path, {"train": 5, "dev": 2, "test": 2}, source_batch_size=2)
    calls, rejected = workers(monkeypatch, reject_first=True)
    report = campaign.run(config)
    assert {
        key: value["trajectories"] for key, value in report["completed_by_split"].items()
    } == config["split_counts"]
    assert report["remaining_split_counts"] == {"train": 0, "dev": 0, "test": 0}
    root = Path(config["dataset_dir"])
    pool = load_json(root / "source-pool.json")
    used = load_previous_sources([root])
    assert pool["statistics"]["source_batches"] > 1
    assert len(used) == 10 and rejected <= {row["document_id"] for row in used}
    assert pool["statistics"]["scanned_documents"] >= len(used)
    assert len({row["document_id"] for row in used}) == len(used)
    assert load_json(root / "preparation.json")["used_sources_file"] == "used-sources.jsonl"
    for split, count in config["split_counts"].items():
        assert len((root / f"{split}.jsonl").read_text().splitlines()) == count
    before = {path.name: path.read_bytes() for path in root.glob("*.jsonl")}
    call_count = len(calls)
    campaign.run(config)
    assert len(calls) == call_count
    assert before == {path.name: path.read_bytes() for path in root.glob("*.jsonl")}


def test_recovery_reuses_pending_allocation_and_allows_endpoint_change(tmp_path, monkeypatch):
    config, _ = case(tmp_path, {"train": 3, "dev": 0, "test": 0})
    workers(monkeypatch, fail_stage="annotate")
    with pytest.raises(RuntimeError, match="annotate"):
        campaign.run(config)
    root = Path(config["dataset_dir"])
    pending = load_previous_sources([root])
    assert pending and not (root / "preparation.json").exists()
    changed = copy.deepcopy(config)
    changed["annotation"]["endpoint"] = "http://new-service"
    calls, _ = workers(monkeypatch)
    campaign.run(changed)
    assert all(item[2]["annotation"]["endpoint"] == "http://new-service" for item in calls)
    assert {item["document_id"] for item in pending} <= {
        item["document_id"] for item in load_previous_sources([root])
    }
    assert len(load_previous_sources([root])) == 3
    assert campaign.report(changed)["dataset_published"]


def test_exhaustion_reports_remaining_counts_without_publishing(tmp_path, monkeypatch):
    config, _ = case(tmp_path, {"train": 3, "dev": 0, "test": 0}, size=2)
    workers(monkeypatch)
    with pytest.raises(ValueError, match="remaining trajectories"):
        campaign.run(config)
    assert not (Path(config["dataset_dir"]) / "preparation.json").exists()
    assert len(load_previous_sources([config["dataset_dir"]])) == 2
    assert campaign.report(config)["remaining_split_counts"]["train"] == 1


def test_previous_dataset_only_requires_ledger_and_ignores_unused_scanned_candidates(
    tmp_path, monkeypatch
):
    config, _ = case(tmp_path, {"train": 1, "dev": 0, "test": 0}, size=4, source_batch_size=4)
    workers(monkeypatch)
    campaign.run(config)
    first = Path(config["dataset_dir"])
    used = load_previous_sources([first])
    (first / "source-pool.json").unlink()
    second = dict(
        config,
        dataset_dir=str(tmp_path / "second"),
        artifacts_dir=str(tmp_path / "second-run"),
        previous_datasets=[str(first)],
        split_counts={"train": 3, "dev": 0, "test": 0},
    )
    campaign.run(second)
    assert {item["document_id"] for item in used}.isdisjoint(
        item["document_id"] for item in load_previous_sources([second["dataset_dir"]])
    )
    assert len(load_previous_sources([second["dataset_dir"]])) == 3


def test_factqa_and_multisegment_exclude_each_others_actual_sources(tmp_path, monkeypatch):
    config, _ = case(tmp_path, {"train": 1, "dev": 0, "test": 0}, size=4, source_batch_size=4)
    multisegment = DataPreparationConfig(
        source_dir=config["source_dir"],
        source_batch_size=4,
        source_seed=config["source_seed"],
        selection_seed=config["selection_seed"],
        split_counts=config["split_counts"],
        window=SegmentationConfig(**config["window"], continuation_tokens=1),
    )
    first_meta = prepare_dataset(multisegment, tmp_path / "multi", "first")
    first = tmp_path / "multi" / first_meta["dataset"]
    config["previous_datasets"] = [str(first)]
    workers(monkeypatch)
    campaign.run(config)
    factqa = Path(config["dataset_dir"])
    second_meta = prepare_dataset(multisegment, tmp_path / "multi", "second", [first, factqa])
    second = tmp_path / "multi" / second_meta["dataset"]
    identities = [
        load_previous_sources([root])[0]["document_id"] for root in (first, factqa, second)
    ]
    assert len(set(identities)) == 3


@pytest.mark.parametrize("change", ["target", "model", "qa", "prompt"])
def test_frozen_rules_cannot_change_on_resume(tmp_path, monkeypatch, change):
    config, _ = case(tmp_path, {"train": 1, "dev": 0, "test": 0})
    workers(monkeypatch)
    campaign.run(config)
    changed = copy.deepcopy(config)
    if change == "target":
        changed["split_counts"]["train"] += 1
    elif change == "model":
        changed["annotation"]["model"] = "different"
    elif change == "qa":
        changed["qa"]["max_answer_chars"] += 1
    else:
        (Path(config["prompts_dir"]) / "verify.txt").write_text("different")
    with pytest.raises(ValueError, match="configuration changed"):
        campaign.run(changed)


def test_report_and_finalize_use_frozen_prompt_content(tmp_path, monkeypatch):
    config, _ = case(tmp_path, {"train": 1, "dev": 0, "test": 0})
    workers(monkeypatch)
    campaign.run(config)
    for path in Path(config["prompts_dir"]).glob("*.txt"):
        path.unlink()
    assert campaign.report(config)["dataset_published"]
    assert campaign.finalize(config)["summary"]["complete_documents"] == 1


def test_finalize_rejects_changed_rules_and_mixed_prompts_without_replacing_publication(
    tmp_path, monkeypatch
):
    config, _ = case(tmp_path, {"train": 3, "dev": 0, "test": 0})
    workers(monkeypatch)
    campaign.run(config)
    root = Path(config["dataset_dir"])
    before = {path.name: path.read_bytes() for path in root.glob("*.json*")}
    with pytest.raises(ValueError, match="frozen campaign"):
        campaign.finalize(dict(config, split_counts={"train": 1, "dev": 0, "test": 0}))
    batch = next((Path(config["artifacts_dir"]) / "batches").glob("*/dataset/preparation.json"))
    metadata = load_json(batch)
    metadata["prompts"]["verify"] = "changed"
    save_json(batch, metadata)
    with pytest.raises(ValueError, match="prompts"):
        campaign.finalize(config)
    assert before == {path.name: path.read_bytes() for path in root.glob("*.json*")}


def test_atomic_publication_can_resume_after_interruption(tmp_path, monkeypatch):
    config, _ = case(tmp_path, {"train": 1, "dev": 0, "test": 0})
    workers(monkeypatch)
    replace = publication.os.replace

    def fail_dev(source, destination):
        if Path(destination).name == "dev.jsonl":
            raise OSError("interrupted export")
        return replace(source, destination)

    monkeypatch.setattr(publication.os, "replace", fail_dev)
    with pytest.raises(OSError, match="interrupted"):
        campaign.run(config)
    assert not (Path(config["dataset_dir"]) / "preparation.json").exists()
    monkeypatch.setattr(publication.os, "replace", replace)
    assert campaign.finalize(config)["summary"]["complete_documents"] == 1


def test_concurrent_controller_and_live_child_are_rejected(tmp_path, monkeypatch):
    config, _ = case(tmp_path)
    root = Path(config["artifacts_dir"])
    root.mkdir(parents=True)
    calls, _ = workers(monkeypatch)
    with (root / "campaign.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            campaign.run(config)
    assert not calls
    save_json(root / "status.json", {"stage_pid": os.getpid()})
    with pytest.raises(RuntimeError, match="previous stage process"):
        campaign.run(config)


def test_cli_run_identity_and_prepare_pool_share_one_config(tmp_path, monkeypatch, capsys):
    config, path = case(tmp_path)
    options = [
        "--config",
        str(path),
        "--output-root",
        str(tmp_path / "data"),
        "--artifacts-root",
        str(tmp_path / "runs"),
        "--run-id",
        "test",
    ]
    monkeypatch.setattr("sys.argv", ["factqa", "prepare-pool", *options])
    stage_cli.main()
    assert json.loads(capsys.readouterr().out)["statistics"]["scanned_documents"] == 0
    workers(monkeypatch)
    monkeypatch.setattr("sys.argv", ["campaign", "run", *options])
    campaign.main()
    assert json.loads(capsys.readouterr().out)["dataset_published"]
    assert Path(config["dataset_dir"]).name.endswith("_test")
