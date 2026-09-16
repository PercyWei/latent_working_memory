"""SwanLab 运行关联、原生 dev 曲线与最终评估汇总。"""

from contextlib import contextmanager
import fcntl
import secrets

import swanlab

from latent_working_memory.v1.reporting import _bar, _shade, _table
from latent_working_memory.v1.tracking import swanlab_run


def development_panels():
    panels = {}
    for objective in ("ae", "lm"):
        title = f"dev/{objective}/nll"
        keys = [f"{title}/compression-{i}" for i in range(1, 6)]
        keys += [f"{title}/trajectory", f"{title}/one-shot"]
        panels[title] = {
            "title": title,
            "config": {
                "xAxis": {"key": "step", "name": "step", "type": "FLOAT", "class": "SYSTEM"},
                "yAxis": [
                    {"key": key, "name": key, "type": "FLOAT", "class": "CUSTOM"} for key in keys
                ],
                "xName": "optimizer step",
                "yName": "NLL",
            },
        }
    title = "dev/paired_gap"
    panels[title] = {
        "title": title,
        "config": {
            "xAxis": {"key": "step", "name": "step", "type": "FLOAT", "class": "SYSTEM"},
            "yAxis": [
                {"key": f"{title}/{task}", "name": task, "type": "FLOAT", "class": "CUSTOM"}
                for task in ("ae", "lm")
            ],
            "xName": "optimizer step",
            "yName": "multi-compression minus one-shot NLL",
        },
    }
    return panels


def setting_color(config):
    training = config["training"]
    strategy = (
        "static"
        if not training["multiround_epochs"]
        else ("warmup" if training["warmup_epochs"] else "direct")
    )
    palette = {
        ("warmup", "ae"): "#2459A6",
        ("warmup", "ae_lm"): "#3B899A",
        ("direct", "ae"): "#B45B18",
        ("direct", "ae_lm"): "#96751C",
        ("static", "ae"): "#7951A0",
        ("static", "ae_lm"): "#B44B78",
    }
    return palette[strategy, training["objective"]]


def panel_style(panel, run_id, run_name, base_color):
    result = {}
    for axis in panel["config"]["yAxis"]:
        label = axis["key"].rsplit("/", 1)[-1]
        if label.startswith("compression-"):
            color = _shade(base_color, int(label.rsplit("-", 1)[-1]) - 1, 5)
        else:
            color = _shade(base_color, int(label in {"one-shot", "lm"}), 2)
        result[f"{run_id}-{axis['key']}"] = {
            "name": f"{run_name}/{label}",
            "colors": [color, color],
        }
    return result


def configure_development_panels(run, output, color):
    """Update the shared view through SwanLab's current chart API.

    Serialize updates from this series' concurrent runs so their colors accumulate.
    Metric registration is owned by the SDK; this function only manages the three panels.
    """
    with (output.parent / ".swanlab-panels.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        api = swanlab.Api()
        project_path = run.url.split("/@", 1)[1].split("/runs/", 1)[0]
        remote = api.run(f"{project_path}/{run.id}")

        def checked(response):
            if not response.ok:
                raise RuntimeError(response.errmsg)
            return response.data

        project = checked(api._get(f"/project/{project_path}"))
        view = project["viewIndex"][0]
        sections_path = f"/sections/{project_path}/{view}"
        sections = checked(api._get(sections_path, params={"size": 100}))
        section = next((s for s in sections if s["name"] == "dev"), None)
        if section is None:
            section = checked(
                api._post(
                    sections_path,
                    data={
                        "name": "dev",
                        "index": secrets.token_hex(3),
                        "position": "above",
                    },
                )
            )
            section["chartIndex"] = []
        charts_path = f"/charts/{project_path}/{view}"
        charts = [
            checked(api._get(f"{charts_path}/xxxxxx/{index}")) for index in section["chartIndex"]
        ]
        for title, panel in development_panels().items():
            existing = next(
                (c for c in charts if c["title"] == title and c["type"] == "LINE"), None
            )
            custom = dict((existing or {}).get("custom") or {})
            custom.update(panel_style(panel, remote.run_id, output.name, color))
            body = {
                "type": "LINE",
                "title": title,
                "custom": custom,
                "config": {
                    "xAxis": {"key": "step", "type": "SYSTEM", "class": "SCALAR"},
                    "yAxis": [
                        {"key": a["key"], "type": "FLOAT", "class": "SCALAR"}
                        for a in panel["config"]["yAxis"]
                    ],
                },
            }
            if existing:
                checked(api._put(f"{charts_path}/xxxxxx/{existing['index']}", data=body))
            else:
                checked(api._post(f"{charts_path}/{section['index']}", data=body))


@contextmanager
def reconstruction_run(output, config, mode, project, group, tags):
    compression = config["model"]["compression"]
    method = "pooling" if compression == "mean" else compression
    with swanlab_run(
        output,
        config,
        mode=mode,
        project=project,
        group=group,
        tags=tuple(tags),
        job_type="train",
        fixed_tags=("scope:main", f"method:v2-{method}", "data:fineweb"),
    ) as run:
        if run is not None and mode == "online":
            configure_development_panels(run, output, setting_color(config))
        yield run


def log_training(run, record, cursor, learning_rate, epoch_progress):
    if run is None:
        return
    values = {
        "train/loss": record["loss"],
        "train/ae/nll": record["ae"],
        "train/gradient_norm": record["grad_norm"],
        "train/learning_rate": learning_rate,
        "resources/step_seconds": record["seconds"],
        "resources/source_tokens_per_second": record["source_tokens"] / record["seconds"],
        "resources/peak_memory_gib": record["peak_memory_bytes"] / 1024**3,
        "progress/epoch": record["global_epoch"],
        "progress/epoch_fraction": epoch_progress,
        "progress/source_tokens": cursor["source_tokens"],
        "progress/target_tokens": cursor["target_tokens"],
        "progress/sample_visits": cursor["sample_visits"],
    }
    if record["lm"] is not None:
        values["train/lm/nll"] = record["lm"]
    run.log(values, step=record["step"])


def development_scalars(metrics):
    values = {}
    for task in ("ae", "lm"):
        for i in range(1, 6):
            key = f"round/{i}/{task}_nll"
            if key in metrics:
                values[f"dev/{task}/nll/compression-{i}"] = metrics[key]
        for source, label in (
            (f"trajectory_{task}", "trajectory"),
            (f"one_shot_{task}", "one-shot"),
        ):
            if source in metrics:
                values[f"dev/{task}/nll/{label}"] = metrics[source]
        key = f"final_minus_one_shot_{task}"
        if key in metrics:
            values[f"dev/paired_gap/{task}"] = metrics[key]
    return values


def metric_table(metrics):
    return _table([{"metric": key, "value": value} for key, value in metrics.items()])


def log_development(run, metrics, step):
    if run is not None:
        run.log(development_scalars(metrics) | {"dev/details": metric_table(metrics)}, step=step)


def final_evaluation_values(metrics, records):
    values = {"evaluation/details": metric_table(metrics)}
    colors = {"compressed": "#2459A6", "one-shot": "#28764A"}
    for task in ("ae", "lm"):
        points = {
            f"compression-{i}": metrics[f"round/{i}/{task}_nll"]
            for i in range(1, 6)
            if f"round/{i}/{task}_nll" in metrics
        }
        for source, label in (
            (f"trajectory_{task}", "trajectory"),
            (f"one_shot_{task}", "one-shot"),
        ):
            if source in metrics:
                points[label] = metrics[source]
        if points:
            values[f"evaluation/{task}/nll"] = _bar(
                list(points),
                {"FineWeb": list(points.values())},
                ["one-shot" if label == "one-shot" else "compressed" for label in points],
                colors,
                "write",
            )
    em = "generation/final_round_exact_match"
    if em in metrics:
        values["evaluation/ae/final_compression_em"] = _bar(
            ["last-compression"], {"FineWeb": [metrics[em]]}, ["compressed"], colors, "write"
        )
    examples = [
        swanlab.Text(
            f"Reference:\n{row['generation']['reference']}\n\nPrediction:\n{row['generation']['prediction']}",
            caption=f"document={row['document_id']}, compressions={row['depth']}, "
            f"EM={row['generation']['final_round_exact_match']}",
        )
        for row in records
        if "generation" in row
    ]
    for start in range(0, len(examples), 100):
        values[f"evaluation/examples/page-{start // 100 + 1}"] = examples[start : start + 100]
    return values


def log_final_evaluation(run, metrics, records, step):
    if run is not None:
        run.log(final_evaluation_values(metrics, records), step=step)
