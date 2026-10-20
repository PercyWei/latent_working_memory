"""训练 run 的身份、train/dev 与资源指标、续训历史重放。"""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import secrets

import swanlab

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
    "train/stage",
    "resources/optimizer_step_seconds",
    "resources/peak_memory_allocated_gib",
)


def training_identity(training):
    """训练与恢复共用的运行身份；名称由方法目录和 run-id 派生。"""
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
        configure_metrics(tracking, device)
        if not resume and "continuation" in run:
            rng = capture_rng(device)
            try:
                for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    values = {
                        name: value for name, value in record.items()
                        if name.startswith(("train/", "dev/"))
                    }
                    if values:
                        log_metrics(tracking, {"step": record["step"], **values})
            finally:
                restore_rng(rng, device)
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
