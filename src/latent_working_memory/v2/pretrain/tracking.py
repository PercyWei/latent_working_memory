"""SwanLab 运行关联、原生 dev 曲线与最终评估汇总。"""

from contextlib import contextmanager
import fcntl
import secrets

import swanlab

from latent_working_memory.v1.reporting import _bar, _shade, _table
from latent_working_memory.v1.tracking import swanlab_run


def development_panels():
    panels = {}
    for mode, label in (("", "compression"), ("single/", "single")):
        for objective in ("ae", "lm"):
            title = f"dev/{mode}{objective}/nll"
            keys = [f"{title}/{label}-{i}" for i in range(1, 6)]
            keys.append(f"{title}/{label}-trajectory")
            if mode:
                keys.append(f"{title}/single-final")
            panels[title] = {
                "title": title,
                "config": {
                    "xAxis": {"key": "step", "name": "step", "type": "FLOAT", "class": "SYSTEM"},
                    "yAxis": [
                        {"key": key, "name": key, "type": "FLOAT", "class": "CUSTOM"}
                        for key in keys
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
            "yName": "final compression minus final single compression NLL",
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
        if label.startswith(("compression-", "single-")):
            suffix = label.rsplit("-", 1)[-1]
            if suffix.isdigit():
                index = int(suffix) - 1
            else:
                index = 5 if suffix == "trajectory" else 6
            color = _shade(base_color, index, 7)
        else:
            color = _shade(base_color, int(label == "lm"), 2)
        result[f"{run_id}-{axis['key']}"] = {
            "name": f"{run_name}/{label}",
            "colors": [color, color],
        }
    return result


def configure_development_panels(run, output, color):
    """Update the shared view through SwanLab's current chart API.

    Serialize updates from this series' concurrent runs so their colors accumulate.
    Metric registration is owned by the SDK; this function manages compression, single and paired-gap panels.
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
        "train/gradient_norm": record["grad_norm"],
        "train/learning_rate": learning_rate,
        "resources/step_seconds": record["seconds"],
        "resources/source_tokens_per_second": record["source_tokens"] / record["seconds"],
        "resources/peak_memory_gib": record["peak_memory_bytes"] / 1024**3,
        "resources/mean_microbatch_size": record["samples"] / record["microbatches"],
        "resources/mean_active_microbatch_size": record["mean_active_microbatch_size"],
        "progress/epoch": record["global_epoch"],
        "progress/epoch_fraction": epoch_progress,
        "progress/source_tokens": cursor["source_tokens"],
        "progress/target_tokens": cursor["target_tokens"],
        "progress/sample_visits": cursor["sample_visits"],
    }
    for task in ("ae", "lm"):
        if record[task] is not None:
            values[f"train/{task}/nll"] = record[task]
    run.log(values, step=record["step"])


def development_scalars(metrics):
    values = {}
    for task in ("ae", "lm"):
        for i in range(1, 6):
            key = f"multi_compression/round/{i}/{task}_nll"
            if key in metrics:
                values[f"dev/{task}/nll/compression-{i}"] = metrics[key]
        key = f"multi_compression/trajectory_{task}"
        if key in metrics:
            values[f"dev/{task}/nll/compression-trajectory"] = metrics[key]
        for i in range(1, 6):
            key = f"single_compression/round/{i}/{task}_nll"
            if key in metrics:
                values[f"dev/single/{task}/nll/single-{i}"] = metrics[key]
        key = f"single_compression/trajectory_{task}"
        if key in metrics:
            values[f"dev/single/{task}/nll/single-trajectory"] = metrics[key]
        key = f"final_round_single_compression_{task}"
        if key in metrics:
            values[f"dev/single/{task}/nll/single-final"] = metrics[key]
        key = f"final_round_compression_gap_{task}"
        if key in metrics:
            values[f"dev/paired_gap/{task}"] = metrics[key]
    return values


def metric_table(metrics):
    return _table([{"metric": key, "value": value} for key, value in metrics.items()])


def log_development(run, metrics, step):
    if run is not None:
        run.log(
            development_scalars(metrics) | {"dev/metrics_table": metric_table(metrics)}, step=step
        )


def final_evaluation_values(metrics, records):
    values = {"evaluation/metrics_table": metric_table(metrics)}
    colors = {"compression": "#2459A6", "single": "#28764A"}
    for task in ("ae", "lm"):
        points = {}
        series = []
        for i in range(1, 6):
            for key, label, group in (
                (f"multi_compression/round/{i}/{task}_nll", f"compression-{i}", "compression"),
                (f"single_compression/round/{i}/{task}_nll", f"single-{i}", "single"),
            ):
                if key in metrics:
                    points[label] = metrics[key]
                    series.append(group)
        for key, label, group in (
            (f"multi_compression/trajectory_{task}", "compression-trajectory", "compression"),
            (f"single_compression/trajectory_{task}", "single-trajectory", "single"),
        ):
            if key in metrics:
                points[label] = metrics[key]
                series.append(group)
        if not any(f"single_compression/round/{i}/{task}_nll" in metrics for i in range(1, 6)):
            multi_final = f"final_round_multi_compression_{task}"
            single_final = f"final_round_single_compression_{task}"
            if multi_final in metrics and single_final in metrics:
                points["compression-final"] = metrics[multi_final]
                points["single-final"] = metrics[single_final]
                series.extend(("compression", "single"))
        if points:
            values[f"evaluation/{task}/compression_nll"] = _bar(
                list(points),
                {"FineWeb": list(points.values())},
                series,
                colors,
                "write",
            )
    em_points = {}
    for prefix, label in (
        ("multi_compression/generation", "compression"),
        ("single_compression/generation", "single"),
    ):
        key = f"{prefix}/final_round_exact_match"
        if key in metrics:
            em_points[label] = metrics[key]
    if em_points:
        values["evaluation/ae/final_reconstruction_em"] = _bar(
            list(em_points),
            {"FineWeb": list(em_points.values())},
            list(em_points),
            colors,
            "write",
        )
    examples = []
    for row in records:
        if "multi_compression_generation" not in row:
            continue
        sections = [f"Reference:\n{row['multi_compression_generation']['reference']}"]
        for field, label in (
            ("multi_compression_generation", "Compression"),
            ("single_compression_generation", "Single compression"),
        ):
            if field in row:
                sections.append(
                    f"{label} (EM={row[field]['final_round_exact_match']}):\n{row[field]['prediction']}"
                )
        examples.append(
            swanlab.Text(
                "\n\n".join(sections),
                caption=f"document={row['document_id']}, compressions={row['depth']}",
            )
        )
    for start in range(0, len(examples), 100):
        values[f"evaluation/examples/compression/page-{start // 100 + 1}"] = examples[
            start : start + 100
        ]
    return values


def log_final_evaluation(run, metrics, records, step):
    if run is not None:
        run.log(final_evaluation_values(metrics, records), step=step)
