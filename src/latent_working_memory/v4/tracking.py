"""同一训练 run 的 SwanLab 身份和原生增量指标。"""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import secrets

import swanlab
from swanlab.sdk.internal.run.components.config import Config as SwanLabConfig

from latent_working_memory.runtime import capture_rng, restore_rng
from latent_working_memory.tracking_credentials import swanlab_api_key
from latent_working_memory.v4.io import write_json


METRICS = (
    "train/loss",
    "dev/loss",
    "train/write_loss",
    "dev/write_loss",
    "train/write_loss_before",
    "dev/write_loss_before",
    "train/grad_norm",
    "train/epoch",
    "resources/optimizer_step_seconds",
    "resources/peak_memory_allocated_gib",
)


def training_identity(training):
    """训练、恢复与评估共用的运行身份；名称由方法目录和 run-id 派生。"""
    root = Path(training["output_dir"])
    return {
        "project": training["swanlab_project"],
        "group": training["group"],
        "name": f"{root.name}_{root.parent.parent.name}",
        "job_type": "train",
        "tags": sorted({"scope:main", "method:attention-matching", *training["tags"]}),
    }


@contextmanager
def training_tracking(config, run, device, resume=False):
    if config.swanlab_project is None:
        yield None
        return
    if config.group is None:
        raise ValueError("online tracking requires an explicit group")
    root = Path(config.output_dir)
    identity_path = root / "swanlab.json"
    expected = training_identity(run["training"])
    identity = None
    if resume:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if any(identity[name] != value for name, value in expected.items()):
            raise ValueError("SwanLab identity differs from the training run")
    rng = capture_rng(device)
    try:
        api_key = swanlab_api_key()
        workspace = None
        if identity is not None:
            project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
            workspace, project = project_path.split("/")
            if project != config.swanlab_project:
                raise ValueError("SwanLab URL differs from its saved project")
            remote = swanlab.Api(api_key=api_key).run(f"{project_path}/{identity['id']}")
            if remote.state not in {"FINISHED", "CRASHED", "ABORTED"}:
                raise ValueError("SwanLab run is still active; cannot resume training")
        tracking = swanlab.init(
            project=config.swanlab_project,
            workspace=workspace,
            name=expected["name"],
            config=run,
            mode="online",
            public=False,
            job_type="train",
            group=config.group,
            tags=expected["tags"],
            log_dir=str(root / "swanlab"),
            id=identity["id"] if identity is not None else None,
            resume="must" if identity is not None else "never",
            settings=swanlab.Settings(
                api_key=api_key,
                interactive=False,
                terminal={"proxy_type": "none"},
                probe={"git": False, "monitor": False},
            ),
        )
    finally:
        restore_rng(rng, device)
    with tracking:
        if identity is None:
            write_json(identity_path, {**expected, "id": tracking.id, "url": tracking.url})
        project_path = tracking.url.split("/@", 1)[1].split("/runs/", 1)[0]
        with (root.parents[2] / ".swanlab-panels.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            api = swanlab.Api(api_key=api_key)
            configure_write_loss_panels(api, api.run(f"{project_path}/{tracking.id}"))
        yield tracking


def configure_write_loss_panels(api, remote):
    """合并默认视图中的写入前后曲线，删除冗余面板并保留标量数据。"""
    for split in ("train", "dev"):
        title = f"{split}/write_loss"
        _configure_line_panels(
            api,
            remote,
            split,
            [(title, (
                (f"{split}/write_loss_before", "写入前", "#74a9cf"),
                (title, "写入后", "#2b8cbe"),
            ))],
        )


def _configure_line_panels(api, remote, section_name, panels):
    """按同一 optimizer step 合并可比标量，保留已有 run 的配色和图例。"""
    def checked(response):
        if not response.ok:
            raise RuntimeError(response.errmsg)
        return response.data

    project_path = remote.url.split("/@", 1)[1].split("/runs/", 1)[0]
    project = checked(api._get(f"/project/{project_path}"))
    view = project["viewIndex"][0]
    sections_path = f"/sections/{project_path}/{view}"
    sections = checked(api._get(sections_path, params={"size": 100}))
    charts_path = f"/charts/{project_path}/{view}"
    section = next((row for row in sections if row["name"] == section_name), None)
    if section is None:
        section = checked(
            api._post(
                sections_path,
                data={"name": section_name, "index": secrets.token_hex(3), "position": "above"},
            )
        )
        section["chartIndex"] = []
    charts = [
        checked(api._get(f"{charts_path}/xxxxxx/{index}")) for index in section["chartIndex"]
    ]
    for title, series in panels:
        names = {title, *(metric for metric, _, _ in series)}
        existing = next(
            (chart for chart in charts if chart["title"] == title and chart["type"] == "LINE"),
            None,
        )
        custom = {}
        for chart in charts:
            if chart["type"] == "LINE" and chart["title"] in names:
                custom.update(chart.get("custom") or {})
        custom.update((existing or {}).get("custom") or {})
        for metric, label, color in series:
            custom[f"{remote.run_id}-{metric}"] = {
                "name": f"{remote.name}/{label}",
                "colors": [color, color],
            }
        body = {
            "type": "LINE",
            "title": title,
            "custom": custom,
            "config": {
                **((existing or {}).get("config") or {}),
                "xAxis": {"key": "step", "type": "SYSTEM", "class": "SCALAR"},
                "yAxis": [
                    {"key": metric, "type": "FLOAT", "class": "SCALAR"}
                    for metric, _, _ in series
                ],
            },
        }
        if existing is None:
            existing = checked(api._post(f"{charts_path}/{section['index']}", data=body))
        else:
            checked(api._put(f"{charts_path}/xxxxxx/{existing['index']}", data=body))
        for chart in charts:
            if chart["index"] == existing["index"] or chart["type"] != "LINE":
                continue
            keys = {axis["key"] for axis in chart["config"]["yAxis"]}
            if chart["title"] in names and keys and keys <= names:
                checked(api._delete(f"{charts_path}/{section['index']}/{chart['index']}"))


def configure_metrics(tracking, device):
    if tracking is not None:
        for name in METRICS:
            if name == "resources/peak_memory_allocated_gib" and device.type != "cuda":
                continue
            tracking.define_metric(name, x_axis="_step", section_name=name.split("/", 1)[0])


def log_metrics(tracking, record):
    if tracking is None:
        return
    values = {name: record[name] for name in METRICS if name in record}
    if "resources/peak_memory_allocated_bytes" in record:
        values["resources/peak_memory_allocated_gib"] = (
            record["resources/peak_memory_allocated_bytes"] / 1024**3
        )
    tracking.log(values, step=record["step"])


def evaluation_media(summary):
    """单个 checkpoint 的质量和写入诊断；样本、容量及资源明细留在本地。"""
    metrics = summary["metrics"]
    panels = [
        ("nll", ["答案 NLL"], [metrics["nll"]], "NLL"),
        ("em_f1", ["EM", "F1"], [metrics["em"], metrics["f1"]], "score"),
        (
            "write_loss",
            ["写入前", "写入后"],
            [metrics["write_loss_before"], metrics["write_loss"]],
            "Attention Matching loss",
        )
    ]
    values = {}
    for name, labels, measurements, unit in panels:
        chart = swanlab.echarts.Bar().add_xaxis(labels)
        chart.add_yaxis(
            "attention-matching",
            measurements,
            label_opts={"show": False},
            itemstyle_opts={"color": "#31688e"},
        )
        chart.set_global_opts(
            tooltip_opts={"trigger": "axis"},
            yaxis_opts={"name": unit, "min": 0, **({"max": 1} if name == "em_f1" else {})},
        )
        values[f"evaluation/{name}"] = chart
    return values


def _evaluation_checkpoint_series(summary):
    """多 checkpoint 仅追加当前步的核心标量；benchmark 任务明细留在本地。"""
    metrics = summary["metrics"]
    prefix = "evaluation/checkpoints"
    if "benchmark" in summary:
        benchmark = summary["benchmark"]
        prefix = f"{prefix}/{benchmark}"
        values = {f"{prefix}/score": metrics["score"]}
        series = [(f"{prefix}/score", "总体", "#1d3f55")]
        groups = metrics["by_length"] if benchmark == "ruler" else metrics["groups"]
        for index, row in enumerate(groups):
            if benchmark == "ruler":
                name, label = f"length_{row['length']}", f"{row['length'] / 1024:g}K"
            else:
                name, label = f"category_{row['category']}", row["category_name"]
            fraction = 0.6 * (1 - index / max(1, len(groups) - 1))
            color = "#" + "".join(
                f"{round(channel + (255 - channel) * fraction):02x}" for channel in (49, 104, 142)
            )
            values[f"{prefix}/{name}"] = row["score"]
            series.append((f"{prefix}/{name}", label, color))
        return values, [(prefix, series)]
    values = {
        f"{prefix}/{name}": metrics[name]
        for name in ("nll", "em", "f1", "write_loss_before", "write_loss")
    }
    return values, [
        (f"{prefix}/nll", [(f"{prefix}/nll", "答案 NLL", "#31688e")]),
        (f"{prefix}/em_f1", [
            (f"{prefix}/em", "EM", "#74a9cf"),
            (f"{prefix}/f1", "F1", "#2b8cbe"),
        ]),
        (f"{prefix}/write_loss", [
            (f"{prefix}/write_loss_before", "写入前", "#74a9cf"),
            (f"{prefix}/write_loss", "写入后", "#2b8cbe"),
        ]),
    ]


def validate_evaluation_publication(run, checkpoint_path, step):
    """在线评估前核对已完成训练的身份和成功保存 checkpoint；不访问远端。"""
    training = run["training"]
    if training["swanlab_project"] is None:
        raise ValueError("evaluation publication requires a checkpoint configured for SwanLab")
    root = Path(training["output_dir"]).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if json.loads((root / "run.json").read_text(encoding="utf-8")) != run:
        raise ValueError("checkpoint run differs from the original training directory")
    result = json.loads((root / "training-result.json").read_text(encoding="utf-8"))
    if (
        not result["complete"]
        or type(step) is not int
        or not 0 < step <= result["completed_steps"]
        or checkpoint_path != root / "checkpoints" / f"global_step_{step}"
        or not (checkpoint_path / "state.pt").is_file()
        or Path(result["checkpoint"]).resolve()
        != root / "checkpoints" / f"global_step_{result['completed_steps']}"
    ):
        raise ValueError("publication requires a saved checkpoint from the completed training run")
    identity = json.loads((root / "swanlab.json").read_text(encoding="utf-8"))
    expected = training_identity(training)
    if any(identity[name] != value for name, value in expected.items()):
        raise ValueError("SwanLab identity differs from the original training run")
    project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    _, project = project_path.split("/")
    if project != identity["project"]:
        raise ValueError("training run URL differs from its project")
    return root, identity, project_path


def append_evaluation(
    run, checkpoint_path, summary, output_dir, media=None, checkpoint_series=False
):
    """将 test 评估追加到原训练 run；多 checkpoint 使用真实步数的原生曲线。"""
    if (
        summary["split"] != "test"
        or Path(summary["checkpoint"]["path"]).resolve() != Path(checkpoint_path).resolve()
    ):
        raise ValueError("publication summary must match the requested test checkpoint")
    root, identity, project_path = validate_evaluation_publication(
        run, checkpoint_path, summary["checkpoint"]["step"]
    )
    workspace, project = project_path.split("/")
    api_key = swanlab_api_key()
    api = swanlab.Api(api_key=api_key)
    remote = api.run(f"{project_path}/{identity['id']}")
    if remote.state != "FINISHED":
        raise ValueError("evaluation append requires a finished training run")
    remote_config = {
        name: entry["value"]
        for name, entry in sorted(
            remote.profile["config"].items(), key=lambda pair: pair[1]["sort"]
        )
    }
    expected_config = SwanLabConfig()
    expected_config.update(run)
    if any(remote_config.get(name) != value for name, value in expected_config.items()):
        raise ValueError("SwanLab configuration differs from the saved training run")
    if checkpoint_series:
        values, panels = _evaluation_checkpoint_series(summary)
    else:
        values = evaluation_media(summary) if media is None else media
        panels = []
    with swanlab.init(
        project=project,
        workspace=workspace,
        name=remote.name,
        config=remote_config,
        job_type=identity["job_type"],
        group=identity["group"],
        tags=identity["tags"],
        id=identity["id"],
        resume="must",
        mode="online",
        log_dir=str(root / "swanlab"),
        settings=swanlab.Settings(
            api_key=api_key,
            interactive=False,
            terminal={"proxy_type": "none"},
            probe={"git": False, "monitor": False},
        ),
    ) as tracking:
        if checkpoint_series:
            with (root.parents[2] / ".swanlab-panels.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                _configure_line_panels(api, remote, "evaluation", panels)
        for name in values:
            if checkpoint_series:
                tracking.define_metric(name, x_axis="_step", section_name="evaluation")
            else:
                tracking.define_metric(name, section_name="evaluation")
        tracking.log(values, step=summary["checkpoint"]["step"])
    publication = {
        "training_dir": str(root),
        "run_id": identity["id"],
        "url": identity["url"],
        "project": project,
        "group": identity["group"],
        "checkpoint": summary["checkpoint"],
        "split": summary["split"],
        "checkpoint_series": checkpoint_series,
        "panels": [title for title, _ in panels] if checkpoint_series else list(values),
        "metric_names": list(values),
        "x_axis": "_step" if checkpoint_series else None,
        "metrics": summary["metrics"],
        "summary_file": str(Path(output_dir).resolve() / "summary.json"),
    }
    if "benchmark" in summary:
        publication["benchmark"] = summary["benchmark"]
    write_json(Path(output_dir) / "publication.json", publication)
    return publication
