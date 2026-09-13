"""Replace dev ECharts snapshots with native curves on existing training runs."""

import argparse
import json
from pathlib import Path

import swanlab

from latent_working_memory.v1.dev_scalars import configure_development_panels, development_scalars
from latent_working_memory.v1.reporting import OVERVIEW_METRICS
from latent_working_memory.v1.reupload_training import prepare_replay
from latent_working_memory.v1.tracking import swanlab_training_run


def migrate(training_dir):
    prepared = prepare_replay(training_dir)
    identity = json.loads((training_dir / "swanlab.json").read_text())
    audit_path = training_dir / "evaluation-publications/dev-scalar-migration.json"
    if audit_path.exists():
        raise ValueError("dev scalar migration has already completed")
    api = swanlab.Api()
    project = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    remote = api.run(f"{project}/{identity['id']}")
    base = f"/experiment/{remote.run_id}"

    def checked(response):
        if not response.ok:
            raise RuntimeError(response.errmsg)
        return response.data

    before = checked(api._get(f"{base}/sections", params={"size": 100}))
    backup = audit_path.with_name("layout-before-dev-scalars.json")
    if not backup.exists():
        backup.write_text(json.dumps(before, indent=2) + "\n")
    sources = list(prepared["config"]["evaluation_preparations"])
    with swanlab_training_run(training_dir) as run:
        configure_development_panels(run, sources, "online")
        for step, reports in sorted(prepared["dev"].items()):
            run.log(development_scalars({name: value[0] for name, value in reports.items()}), step=step)
    current = checked(api._get(f"{base}/sections", params={"size": 100}))
    section = next(s for s in current if s["name"] == "dev")
    charts = [checked(api._get(f"{base}/chart/{index}/info")) for index in section["chartIndex"]]
    titles = {f"dev/overview/{task}/{metric}" for task, metric in OVERVIEW_METRICS}
    old = [c["index"] for c in charts if c["type"] == "ECHARTS" and c["title"] in titles]
    hidden = checked(api._get(f"{base}/sections/protected", params={"types": "HIDDEN"}))[0]
    checked(api._put(f"{base}/chart/order", data={"sections": [
        {"index": section["index"], "chartIndex": [i for i in section["chartIndex"] if i not in old]},
        {"index": hidden["index"], "chartIndex": list(dict.fromkeys(hidden["chartIndex"] + old))},
    ]}))
    audit = {"run_id": identity["id"], "dev_steps": sorted(prepared["dev"]),
             "archived_snapshot_panels": old,
             "native_panels": [c["title"] for c in charts if c["type"] == "LINE"]}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    migrate(parser.parse_args().training_run)


if __name__ == "__main__":
    main()
