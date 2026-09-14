"""将补齐空记忆对照的最终 QA 图表发布到已有 run，保留原指标和训练身份。"""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import swanlab

from latent_working_memory.v1.dynamic.empty_memory import (
    EMPTY_CONDITIONS,
    ORIGINAL_CONDITIONS,
    read_identity,
    reuse_empty_rows,
)
from latent_working_memory.v1.dynamic.evaluation import aggregate_qa
from latent_working_memory.v1.dynamic.reporting import qa_media
from latent_working_memory.v1.tracking import swanlab_training_run


PREFIX = "evaluation/test/with-baselines"
OLD_PREFIX = "evaluation/test/overview"


def load_reports(manifest, source_dir):
    entries = json.loads(manifest.read_text())
    if not entries or set(entries) - {"squad", "personamem"}:
        raise ValueError("reports must map evaluation dataset names to supplemented JSON reports")
    provenance = json.loads((source_dir / "provenance.json").read_text())
    step = provenance["target_steps"]
    expected_checkpoint = source_dir / "checkpoints" / f"dynamic-step-{step:06d}.pt"
    reports, origins = {}, {}
    for dataset, value in entries.items():
        path = (manifest.parent / value).resolve()
        info = json.loads((path.parent / "evaluation.json").read_text())
        details = info["supplemental_baselines"]
        if (
            info["dataset"] != dataset
            or info["split"] != "test"
            or info["checkpoint_step"] != step
            or info["config"] != provenance["config"]
            or Path(info["checkpoint"]).resolve() != expected_checkpoint.resolve()
            or Path(details["pretrain_checkpoint"]).resolve()
            != Path(provenance["initial_checkpoint"]).resolve()
        ):
            raise ValueError("supplemented report belongs to a different training run or dataset")
        rows = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
        original_path = Path(details["source_report"])
        original = [
            json.loads(line)
            for line in original_path.with_suffix(".jsonl").read_text().splitlines()
        ]
        old = {(*read_identity(r), r["condition"]): r for r in original}
        retained = {
            (*read_identity(r), r["condition"]): r
            for r in rows
            if r["condition"] in ORIGINAL_CONDITIONS
        }
        if retained != old or len(rows) != len(old) // 5 * 7:
            raise ValueError("original evaluation records changed or conditions are incomplete")
        if any(r["condition"] not in ORIGINAL_CONDITIONS | set(EMPTY_CONDITIONS) for r in rows):
            raise ValueError("unknown supplemental condition")
        reuse_empty_rows(original, rows)
        metrics = aggregate_qa(rows)
        if json.loads(path.read_text()) != metrics:
            raise ValueError("supplemented metrics do not match per-read records")
        reports[dataset] = metrics, rows
        origins[dataset] = {"report": str(path), "evaluation": info}
    return reports, origins, step


def checked(response):
    if not response.ok:
        raise RuntimeError(response.errmsg)
    return response.data


def snapshot_run(remote):
    return {
        "name": remote.name,
        "group": remote.group,
        "job_type": remote.job_type,
        "labels": remote.labels,
        "config": remote.profile["config"],
        "scalars": remote.metrics(
            keys=remote.series().json()["keys"], all=True, ignore_timestamp=True
        ),
    }


def media_contents(remote, keys, step):
    series = {item.key: item for item in remote.series(metric_type="MEDIA")}
    result = {}
    for key in keys:
        data = series[key].metric(media_step=step)
        points = data["metrics"]
        if len(points) != 1 or points[0]["index"] != step:
            raise ValueError(f"unexpected media steps: {key}")
        bodies = [urlopen(item["url"], timeout=30).read().decode("utf-8") for item in points[0]["items"]]
        result[key] = bodies if key.endswith("/examples") else [json.loads(b) for b in bodies]
    return result


def publish_supplement(run_dir, manifest, output_dir, mode):
    binding = json.loads((run_dir / "republication.json").read_text())
    identity = json.loads((run_dir / "swanlab.json").read_text())
    if binding["new_run"]["id"] != identity["id"]:
        raise ValueError("display run identity differs from its source binding")
    source = Path(binding["source"]["training_run"])
    reports, origins, step = load_reports(manifest, source)
    media = qa_media(reports, prefix=PREFIX)
    expected = {
        key: [item.content for item in value]
        if key.endswith("/examples")
        else [json.loads(value.dump_options())]
        for key, value in media.items()
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "media.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n"
    )
    publication = {
        "run": identity,
        "source_training_run": str(source),
        "step": step,
        "datasets": origins,
        "media_keys": list(expected),
        "status": "prepared",
    }
    receipt = output_dir / "publication.json"
    receipt.write_text(json.dumps(publication, indent=2) + "\n")
    if mode == "disabled":
        return
    api = swanlab.Api()
    project = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    remote = api.run(f"{project}/{identity['id']}")
    if remote.state != "FINISHED":
        raise ValueError("supplement requires a finished cloud run")
    existing_media = remote.series(metric_type="MEDIA").json()["keys"]
    if set(expected) & set(existing_media):
        raise ValueError("baseline supplement already exists; do not duplicate the same steps")
    before = snapshot_run(remote)
    base = f"/experiment/{remote.run_id}"
    sections = checked(api._get(f"{base}/sections", params={"size": 100}))
    old_keys = [key.replace(PREFIX, OLD_PREFIX, 1) for key in expected]
    old_media = media_contents(remote, old_keys, step)
    old_charts = []
    for section in sections:
        if section["name"] != "evaluation":
            continue
        for index in section["chartIndex"]:
            chart = checked(api._get(f"{base}/chart/{index}/info"))
            if chart["title"] in old_keys:
                old_charts.append(chart)
    if {c["title"] for c in old_charts} != set(old_keys):
        raise ValueError("original final evaluation panels are missing or ambiguous")
    (output_dir / "before.json").write_text(
        json.dumps(
            {"run": before, "charts": old_charts, "old_media": old_media},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("Publishing new final-evaluation media to existing run", flush=True)
    with swanlab_training_run(run_dir) as run:
        run.log(media, step=step)
    remote = api.run(f"{project}/{identity['id']}")
    if media_contents(remote, expected, step) != expected:
        raise ValueError(
            "uploaded baseline charts or examples differ from the prepared publication"
        )
    if snapshot_run(remote) != before:
        raise ValueError("cloud run identity, configuration or scalar history changed")
    # SwanLab v0 media definitions are immutable. Replace only the old chart definitions,
    # after verifying their replacements; the metric data remain queryable in House.
    for chart in old_charts:
        checked(api._delete(f"{base}/chart/{chart['index']}/hard"))
    remote = api.run(f"{project}/{identity['id']}")
    if media_contents(remote, old_keys, step) != old_media:
        raise ValueError("original media data changed while replacing panel definitions")
    if snapshot_run(remote) != before or remote.state != "FINISHED":
        raise ValueError("cloud run changed unexpectedly during panel replacement")
    after_sections = checked(api._get(f"{base}/sections", params={"size": 100}))
    evaluation = next(s for s in after_sections if s["name"] == "evaluation")
    charts = [checked(api._get(f"{base}/chart/{index}/info")) for index in evaluation["chartIndex"]]
    titles = [c["title"] for c in charts]
    if set(old_keys) & set(titles) or not set(expected) <= set(titles):
        raise ValueError("final evaluation panel replacement is incomplete")
    publication.update(
        status="complete",
        replaced_panels=[c["index"] for c in old_charts],
        original_scalars_unchanged=True,
        original_media_preserved=True,
        uploaded_media_verified=True,
    )
    receipt.write_text(json.dumps(publication, indent=2) + "\n")
    print(json.dumps({"run_id": identity["id"], "status": "complete", "datasets": list(reports)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Current display run directory")
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--swanlab-mode", choices=("disabled", "online"), default="disabled")
    args = parser.parse_args()
    publish_supplement(args.run_dir, args.reports, args.output_dir, args.swanlab_mode)


if __name__ == "__main__":
    main()
