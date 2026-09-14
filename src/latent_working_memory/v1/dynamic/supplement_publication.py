"""将补齐空记忆对照的最终 QA 图表发布到已有 run，保留原指标和训练身份。"""

import argparse
import json
import time
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


def same_run(before, after):
    # House may return slightly different derived min/max/average summaries for
    # unchanged data. Compare the complete actual step/value sequences instead.
    fields = ("name", "group", "job_type", "labels", "config")
    if any(before[k] != after[k] for k in fields):
        return False

    def points(snapshot):
        return {
            row["key"]: sorted((point["step"], point["value"]) for point in row["metrics"])
            for row in snapshot["scalars"]["list"]
        }

    return points(before) == points(after)


def media_contents(remote, keys, step):
    series = {item.key: item for item in remote.series(metric_type="MEDIA")}
    result = {}
    for key in keys:
        data = series[key].metric(media_step=step)
        points = data["metrics"]
        if len(points) != 1 or points[0]["index"] != step:
            raise ValueError(f"unexpected media steps: {key}")
        bodies = [
            urlopen(item["url"], timeout=30).read().decode("utf-8") for item in points[0]["items"]
        ]
        result[key] = bodies if key.endswith("/examples") else [json.loads(b) for b in bodies]
    return result


def wait_for_media(api, path, keys, attempts=16):
    for attempt in range(attempts):
        remote = api.run(path)
        if set(keys) <= set(remote.series(metric_type="MEDIA").json()["keys"]):
            return remote
        if attempt + 1 < attempts:
            time.sleep(2)
    raise RuntimeError(
        "cloud media indexing is not ready; resume publication after it becomes visible"
    )


def publish_supplement(run_dir, manifest, output_dir, mode, resume=False):
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
    receipt = output_dir / "publication.json"
    if resume:
        publication = json.loads(receipt.read_text())
        if (
            publication["run"] != identity
            or publication["datasets"] != origins
            or publication["step"] != step
            or json.loads((output_dir / "media.json").read_text()) != expected
        ):
            raise ValueError("publication inputs changed on resume")
        if publication["status"] == "complete":
            raise ValueError("publication is already complete")
    else:
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
        receipt.write_text(json.dumps(publication, indent=2) + "\n")
    if mode == "disabled":
        return
    api = swanlab.Api()
    project = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    run_path = f"{project}/{identity['id']}"
    remote = api.run(run_path)
    if remote.state != "FINISHED":
        raise ValueError("supplement requires a finished cloud run")
    existing_media = set(remote.series(metric_type="MEDIA").json()["keys"])
    if not resume and set(expected) & existing_media:
        raise ValueError("baseline supplement already exists; resume its publication instead")
    base = f"/experiment/{remote.run_id}"
    old_keys = [key.replace(PREFIX, OLD_PREFIX, 1) for key in expected]
    if resume:
        saved = json.loads((output_dir / "before.json").read_text())
        before, old_charts, old_media = saved["run"], saved["charts"], saved["old_media"]
        present = set(expected) & existing_media
        if present and media_contents(remote, present, step) != {
            key: expected[key] for key in present
        }:
            raise ValueError("existing supplemental media differ from this publication")
    else:
        before = snapshot_run(remote)
        old_media = media_contents(remote, old_keys, step)
        sections = checked(api._get(f"{base}/sections", params={"size": 100}))
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
    if not same_run(before, snapshot_run(remote)):
        raise ValueError("cloud run identity, configuration or scalar history changed")
    pending = {key: value for key, value in media.items() if key not in existing_media}
    if pending:
        publication["status"] = "uploading"
        receipt.write_text(json.dumps(publication, indent=2) + "\n")
        print("Publishing new final-evaluation media to existing run", flush=True)
        with swanlab_training_run(run_dir) as run:
            run.log(pending, step=step)
    remote = wait_for_media(api, run_path, expected)
    if media_contents(remote, expected, step) != expected:
        raise ValueError(
            "uploaded baseline charts or examples differ from the prepared publication"
        )
    if not same_run(before, snapshot_run(remote)):
        raise ValueError("cloud run identity, configuration or scalar history changed")
    publication["status"] = "verified"
    receipt.write_text(json.dumps(publication, indent=2) + "\n")
    # v0 media charts cannot be edited. Replace their definitions only after verification;
    # keep the old House metric data and assert preservation after the change.
    sections = checked(api._get(f"{base}/sections", params={"size": 100}))
    remaining = {index for section in sections for index in section["chartIndex"]}
    for chart in old_charts:
        if chart["index"] in remaining:
            current = checked(api._get(f"{base}/chart/{chart['index']}/info"))
            if current != chart:
                raise ValueError("original panel was edited during publication")
            checked(api._delete(f"{base}/chart/{chart['index']}/hard"))
    remote = api.run(run_path)
    if media_contents(remote, old_keys, step) != old_media:
        raise ValueError("original media data changed while replacing panel definitions")
    if not same_run(before, snapshot_run(remote)) or remote.state != "FINISHED":
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a verified publication without resending existing media",
    )
    args = parser.parse_args()
    publish_supplement(args.run_dir, args.reports, args.output_dir, args.swanlab_mode, args.resume)


if __name__ == "__main__":
    main()
