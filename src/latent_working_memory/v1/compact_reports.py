"""Replace legacy dev/test panels using saved reports; retain old panels in Hidden."""

import argparse
import json
from pathlib import Path

import swanlab

from latent_working_memory.v1.reporting import (
    development_overview, evaluation_overview, paired_reconstructions,
)
from latent_working_memory.v1.tracking import swanlab_training_run


def compact_training_reports(training_dir, step):
    identity = json.loads((training_dir / "swanlab.json").read_text())
    publications = training_dir / "evaluation-publications"
    receipt = json.loads((publications / f"test-step-{step:06d}.json").read_text())
    if receipt["checkpoint_step"] != step:
        raise ValueError("publication checkpoint step differs from requested step")
    entries = receipt["reports"]
    paths = [(entry["evaluation_source"], training_dir / entry["evaluation_source"] /
              f"dev-step-{step:06d}.jsonl") for entry in entries]
    values = development_overview(paths, step)
    reports = [(entry["evaluation_source"], json.loads(Path(entry["report"]).read_text()))
               for entry in entries]
    prefix = "evaluation/test/overview"
    values.update(evaluation_overview(reports, prefix))
    values.update(paired_reconstructions(
        [(entry["evaluation_source"], Path(entry["report"]).with_suffix(".jsonl")) for entry in entries],
        prefix,
    ))
    values[f"{prefix}/metadata"] = swanlab.Text(json.dumps(receipt, ensure_ascii=False))
    api = swanlab.Api()
    project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    remote = api.run(f"{project_path}/{identity['id']}")
    if remote.state != "FINISHED":
        raise ValueError("chart compaction requires a finished run")
    base = f"/experiment/{remote.run_id}"

    def get(path, params):
        response = api._get(path, params=params)
        if not response.ok:
            raise RuntimeError(response.errmsg)
        return response.data

    sections = get(f"{base}/sections", {"size": 100})
    backup = publications / "chart-layout-before-compaction.json"
    if backup.exists():
        original = json.loads(backup.read_text())
    else:
        original = {"run_id": identity["id"], "sections": sections,
                    "hidden": get(f"{base}/sections/protected", {"types": "HIDDEN"})}
        backup.write_text(json.dumps(original, ensure_ascii=False, indent=2) + "\n")
    if original["run_id"] != identity["id"]:
        raise ValueError("layout backup belongs to another run")
    old_ids = {chart for section in original["sections"] if section["name"] in {"dev", "evaluation"}
               for chart in section["chartIndex"]}
    with swanlab_training_run(training_dir) as run:
        run.log(values, step=step)
    current = get(f"{base}/sections", {"size": 100})
    hidden = get(f"{base}/sections/protected", {"types": "HIDDEN"})[0]
    updates = [{"index": section["index"],
                "chartIndex": [index for index in section["chartIndex"] if index not in old_ids]}
               for section in current if section["name"] in {"dev", "evaluation"}]
    updates.append({"index": hidden["index"],
                    "chartIndex": list(dict.fromkeys(hidden["chartIndex"] + sorted(old_ids)))})
    # Same two-section reorder operation as the dashboard drag-and-drop API.
    response = api._put(f"{base}/chart/order", data={"sections": updates})
    if not response.ok:
        raise RuntimeError(response.errmsg)
    after = get(f"{base}/sections", {"size": 100})
    hidden_after = get(f"{base}/sections/protected", {"types": "HIDDEN"})[0]
    counts = {section["name"]: len(section["chartIndex"])
              for section in after if section["name"] in {"dev", "evaluation"}}
    assert old_ids.issubset(hidden_after["chartIndex"])
    for section in after:
        if section["name"] in {"dev", "evaluation"}:
            assert not old_ids.intersection(section["chartIndex"])
        else:
            previous = next(s for s in original["sections"] if s["index"] == section["index"])
            assert section == previous
    expected = {section: sum(key.startswith(section + "/") for key in values)
                for section in ("dev", "evaluation")}
    if counts != expected:
        raise RuntimeError(f"unexpected visible chart counts: {counts}, expected {expected}")
    audit = {"run_id": identity["id"], "checkpoint_step": step,
             "before": {s["name"]: len(s["chartIndex"]) for s in original["sections"]
                        if s["name"] in counts}, "after": counts,
             "hidden_legacy_charts": len(old_ids), "published_keys": list(values)}
    (publications / "chart-compaction.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    compact_training_reports(args.training_run, args.step)


if __name__ == "__main__":
    main()
