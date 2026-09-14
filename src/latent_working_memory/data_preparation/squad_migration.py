"""Explicitly replace an existing SQuAD index and its saved dynamic data identities."""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from zoneinfo import ZoneInfo

import torch
from transformers import AutoTokenizer

from latent_working_memory.data_preparation.squad import SPLITS, prepare_squad
from latent_working_memory.v1.checkpoint import _atomic_torch_save
from latent_working_memory.v1.dynamic.squad import SquadDataset


def same_payload(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(same_payload(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(same_payload(a, b) for a, b in zip(left, right))
        )
    return left == right


def migrate_squad(index_path, record_dir, plan_paths=(), checkpoint_paths=()):
    """Verify equivalent source assignments/lengths before updating saved identities.

    Call only with inactive runs. Backups stay outside the four-file dataset; failures
    restore the original dataset and any modified plan/checkpoint files.
    """
    root = index_path.parent
    if set(root.iterdir()) != {index_path}:
        raise ValueError("migration requires a directory containing only the original index")
    if record_dir.exists() or record_dir.resolve().is_relative_to(root.resolve()):
        raise ValueError("migration records require a new directory outside the dataset")
    if len(set(plan_paths)) != len(plan_paths) or len(set(checkpoint_paths)) != len(
        checkpoint_paths
    ):
        raise ValueError("duplicate migration target")
    original = json.loads(index_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(original["tokenizer"], local_files_only=True)
    staging = Path(tempfile.mkdtemp(prefix=".squad-format-", dir=root.parent))
    output = staging / "dataset"
    old_directory = staging / "original"
    plans, checkpoints, applied = {}, {}, []
    try:
        prepare_squad(
            Path(original["source_files"]["train"]),
            Path(original["source_files"]["dev"]),
            tokenizer,
            output,
            original["seed"],
        )
        data = SquadDataset(output, tokenizer)
        # This comparison covers the original order within each split as well as values.
        all_records = {split: [] for split in (*SPLITS, "excluded")}
        for row in original["articles"]:
            all_records[row["split"]].append(row)
        for split, old_rows in all_records.items():
            new_rows = (
                data.preparation["excluded"]
                if split == "excluded"
                else [r for r in data.records.values() if r["split"] == split]
            )
            old_view = [
                (
                    r["document_id"],
                    r["official_split"],
                    r["article_index"],
                    r["title"],
                    r["group"],
                    r["input_tokens"],
                    r["paragraph_tokens"],
                    r["questions"],
                )
                for r in old_rows
            ]
            new_view = [
                (
                    r["document_id"],
                    r["official_split"],
                    r["article_index"],
                    r["title"],
                    r["group_id"],
                    r["reference_input_tokens"],
                    r["reference_paragraph_tokens"],
                    r["question_count"],
                )
                for r in new_rows
            ]
            if old_view != new_view:
                raise ValueError(
                    "source assignment, order or reference lengths changed; migration aborted"
                )
        for path in plan_paths:
            plan = json.loads(path.read_text())
            if plan["data_index"] != original:
                raise ValueError(
                    f"evaluation plan does not belong to the original SQuAD index: {path}"
                )
            plans[path] = plan
        for path in checkpoint_paths:
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if payload["phase"] != "dynamic":
                raise ValueError("only dynamic checkpoints have a SQuAD evaluation identity")
            plan = payload["progress"]["identity"]["evaluation_plan"]
            if plan["data_index"] != original or plan not in plans.values():
                raise ValueError(f"checkpoint evaluation plan differs: {path}")
            checkpoints[path] = payload
        # Relocate source references from staging to the final dataset location.
        metadata = data.preparation
        import_paths = {
            name: os.path.relpath(Path(path).resolve(), root.resolve())
            for name, path in original["source_files"].items()
        }
        metadata["source_files"] = import_paths
        (output / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
        record_dir.mkdir(parents=True)
        shutil.copy2(index_path, record_dir / "original-index.json")
        for i, path in enumerate([*plans, *checkpoints]):
            shutil.copy2(path, staging / f"backup-{i}")
        root.rename(old_directory)
        output.rename(root)
        if SquadDataset(root, tokenizer).index != data.index:
            raise ValueError("dataset relocation changed runtime identity")
        for path, plan in plans.items():
            updated = deepcopy(plan)
            updated["data_index"] = data.index
            temporary = staging / "plan.json"
            temporary.write_text(json.dumps(updated, indent=2) + "\n")
            applied.append(path)
            temporary.replace(path)
        for path, payload in checkpoints.items():
            # Only the structural data identity changes. Model/optimizer/RNG state is reused.
            payload["progress"]["identity"]["evaluation_plan"]["data_index"] = data.index
            applied.append(path)
            _atomic_torch_save(payload, path)
            actual = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if not same_payload(actual, payload):
                raise ValueError("checkpoint contents differ after migration")
        report = {
            "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "dataset_dir": str(root.resolve()),
            "preparation_id": metadata["preparation_id"],
            "splits": metadata["splits"],
            "evaluation_plans": [str(p) for p in plans],
            "checkpoints": [str(p) for p in checkpoints],
            "source_order_splits_and_reference_lengths_unchanged": True,
            "changed_checkpoint_fields": ["progress.identity.evaluation_plan.data_index"],
        }
        (record_dir / "migration.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except BaseException:
        targets = [*plans, *checkpoints]
        for path in reversed(applied):
            shutil.copy2(staging / f"backup-{targets.index(path)}", path)
        if old_directory.exists():
            if root.exists():
                shutil.rmtree(root)
            old_directory.rename(root)
        raise
    finally:
        shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = migrate_squad(args.index, args.record_dir, args.evaluation_plan, args.checkpoint)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
