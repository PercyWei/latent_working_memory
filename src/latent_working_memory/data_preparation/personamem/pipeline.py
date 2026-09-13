"""Run and verify the complete, bounded-scope factual QA construction workflow."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import json
from pathlib import Path
import random

from latent_working_memory.data_preparation.personamem import (
    artifacts,
    audit,
    construction,
    sources,
)
from latent_working_memory.data_preparation.personamem.blocks import evidence_blocks
from latent_working_memory.data_preparation.personamem.common import now


def logged(run, name, function, *args, **kwargs):
    with (run / name).open("a", buffering=1) as handle:
        with redirect_stdout(handle), redirect_stderr(handle):
            return function(*args, **kwargs)


def candidate_stage_complete(config, stage):
    root, dataset = Path(config["artifacts_dir"]), Path(config["dataset_dir"])
    path = root / f"{stage}.summary.json"
    if not path.exists():
        return False
    candidates = artifacts.read_json(dataset / "candidates.json")
    selected = candidates[: config["pilot_candidates"]] if stage == "pilot" else candidates
    for index, candidate in enumerate(selected):
        result_path = root / "results" / f"candidate-{index:05d}.json"
        if not result_path.exists():
            return False
        result = artifacts.read_json(result_path)
        if result["candidate_id"] != candidate["candidate_id"]:
            raise ValueError("saved candidate order differs from the prepared sources")
        if not result["ok"]:
            return False
    return not artifacts.read_json(path)["interrupted"]


def source_review(config):
    root, dataset = Path(config["artifacts_dir"]), Path(config["dataset_dir"])
    if (root / "source_review.json").exists():
        return
    qas = artifacts.read_rows(dataset / "qas.provisional.jsonl")
    selected = list(qas)
    random.Random(config["seed"]).shuffle(selected)
    panel = selected[:12]
    sources_by_user = {}
    for q in qas:
        sources_by_user.setdefault(q["persona_id"], []).append(q)
    for user, rows in sources_by_user.items():
        evidence_blocks(artifacts.read_json(dataset / "histories" / f"{user}.json"), rows)
    construction.save(root / "source_review_panel.json", panel)
    construction.save(
        root / "source_review.json",
        dict(
            reviewed_at=now(),
            reviewer="programmatic original-message and character-span validation",
            questions=len(qas),
            sample_questions=len(panel),
            validated_spans=len(qas),
            notes="All saved answer and evidence spans reconstructed exactly. This check verifies "
            "positions, not semantic support or human review.",
        ),
    )


def pilot_gate(config):
    root = Path(config["artifacts_dir"])
    path = root / "pilot_gate.json"
    if path.exists():
        gate = artifacts.read_json(path)
    else:
        diagnostic = artifacts.read_json(root / "diagnostic_summary.json")
        review = artifacts.read_json(root / "pilot_review.json")
        progress = artifacts.read_json(root / "progress.json")
        conditions = diagnostic["conditions"]
        gain = conditions["gold"]["f1"] - max(
            conditions[c]["f1"] for c in ("question_only", "wrong")
        )
        passed = (
            progress["failed_candidates"] == 0
            and progress["accepted_qas"] > 0
            and gain > 0.7
            and review["blind_questions"] > 0
            and review["blind_semantically_correct"] == review["blind_questions"]
        )
        gate = dict(
            created_at=now(),
            passed=passed,
            pilot_candidates=config["pilot_candidates"],
            accepted_after_content_review=progress["accepted_qas"],
            raw_diagnostic=diagnostic,
            criteria="All pilot candidates completed; at least one accepted QA; gold F1 exceeds "
            "both controls by >0.7; all sampled blind gold predictions pass model semantic review.",
            limitations="Model-based pilot gate, not independent human or training-model accuracy.",
        )
        construction.save(path, gate)
    if not gate["passed"]:
        raise ValueError("pilot quality gate failed; construction remains incomplete")


def run_all(config):
    artifacts.check_layout(config)
    run = Path(config["artifacts_dir"])
    run.mkdir(parents=True, exist_ok=True)
    with (run / "construction.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = run / "config.json"
        if snapshot.exists() and artifacts.read_json(snapshot) != config:
            raise ValueError("construction config changed; use another run directory")
        if (run / "completion.json").exists():
            return artifacts.verify(config)
        if not snapshot.exists():
            construction.save(snapshot, config)
        artifacts.snapshot_source(run, "source_code")
        if not (run / "provenance.json").exists():
            construction.save(
                run / "provenance.json",
                dict(
                    created_at=now(),
                    source_api=config["source_url"],
                    model=config["model"],
                    reasoning_effort=config["reasoning_effort"],
                    client_request_concurrency_limit=config["concurrency"],
                    artifact_contract="personamem-fact-qa-v1",
                    no_gpu_training_started=True,
                    manual_review="Only actual externally supplied records are used; never fabricated.",
                ),
            )
        logged(run, "pilot.log", sources.prepare, config)
        for stage in ("pilot", "full"):
            if stage == "full":
                pilot_gate(config)
                artifacts.snapshot_source(run, "full_source_code")
            if not candidate_stage_complete(config, stage):
                logged(run, f"{stage}.log", construction.run, config, stage, lock_acquired=True)
            if not candidate_stage_complete(config, stage):
                raise ValueError(
                    f"{stage} did not finish; resume the existing run after resolving errors"
                )
            if stage == "pilot" and not (run / "pilot_gate.json").exists():
                construction.collect(config)
                if not (run / "diagnostic_summary.json").exists():
                    logged(
                        run,
                        "diagnostic.log",
                        audit.diagnose,
                        config,
                        config["diagnostic_questions"],
                    )
                if not (run / "pilot_review.json").exists():
                    logged(run, "pilot_review.log", audit.review_pilot, config)
                construction.collect(config)
                source_review(config)
        if not (run / "final_summary.json").exists():
            construction.collect(config)
            logged(run, "dedup.log", audit.deduplicate, config)
        if not (run / "final_diagnostic/diagnostic_summary.json").exists():
            logged(
                run,
                "final_diagnostic.log",
                audit.diagnose,
                config,
                config["diagnostic_questions"],
                final=True,
            )
        artifacts.snapshot_source(run, "final_source_code")
        artifacts.finalize(config)
        return artifacts.verify(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "verify", "finalize", "prepare", "pilot", "full", "collect"),
        default="all",
    )
    args = parser.parse_args()
    config = artifacts.read_json(args.config)
    if args.stage == "all":
        result = run_all(config)
    elif args.stage == "verify":
        result = artifacts.verify(config)
    elif args.stage == "finalize":
        result = artifacts.finalize(config)
    elif args.stage == "prepare":
        artifacts.check_layout(config)
        result = sources.prepare(config)
    elif args.stage == "collect":
        artifacts.check_layout(config)
        result = construction.collect(config)
    else:
        artifacts.check_layout(config)
        result = construction.run(config, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
