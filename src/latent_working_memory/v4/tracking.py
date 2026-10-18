"""同一训练 run 的身份、训练指标与累计评估报告。"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import secrets
import signal
import time

import torch

import swanlab
from swanlab.sdk.internal.run.components.config import Config as SwanLabConfig

from latent_working_memory.runtime import capture_rng, restore_rng
from latent_working_memory.tracking_credentials import swanlab_api_key
from latent_working_memory.v4.evaluation_records import EvaluationHistory
from latent_working_memory.v4.io import write_json
from latent_working_memory.v4.reporting import evaluation_media


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


def validate_evaluation_publication(run, checkpoint_path, step):
    """在线评估前核对已完成训练的身份和成功保存 checkpoint；不访问远端。"""
    training = run["training"]
    if training["swanlab_project"] is None:
        raise ValueError("evaluation publication requires a checkpoint configured for SwanLab")
    root = Path(training["output_dir"]).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if json.loads((root / "run.json").read_text(encoding="utf-8")) != run:
        raise ValueError("checkpoint run differs from the original training directory")
    result = json.loads((root / "summary.json").read_text(encoding="utf-8"))
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



def _verify_report(api_key, project_path, run_id, media_keys, step):
    """核对完整报告的媒体已到达云端，再提交本地发布记录。"""
    for attempt in range(5):
        remote = swanlab.Api(api_key=api_key).run(f"{project_path}/{run_id}")
        media = {
            item["key"]: item
            for item in remote.medias(keys=list(media_keys), step=step)["list"]
        }
        missing = [
            key for key in media_keys
            if key not in media
            or not any(row["index"] == step and row["items"] for row in media[key]["metrics"])
        ]
        if not missing and remote.state == "FINISHED":
            return
        if attempt < 4:
            time.sleep(2)
    raise RuntimeError(
        f"SwanLab report verification failed at publication {step}: "
        f"state={remote.state}, missing={missing}"
    )


def _configure_report_layout(api, remote, media_keys):
    """完整报告使用媒体面板，清理当前评估类型的旧增量面板。"""
    def checked(response):
        if not response.ok:
            raise RuntimeError(response.errmsg)
        return response.data

    project_path = remote.url.split("/@", 1)[1].split("/runs/", 1)[0]
    project = checked(api._get(f"/project/{project_path}"))
    view = project["viewIndex"][0]
    sections_path = f"/sections/{project_path}/{view}"
    charts_path = f"/charts/{project_path}/{view}"
    sections = checked(api._get(sections_path, params={"size": 100}))
    target = next((row for row in sections if row["name"] == "evaluation"), None)
    if target is None:
        target = checked(api._post(
            sections_path,
            data={"name": "evaluation", "index": secrets.token_hex(3), "position": "above"},
        ))
        target["chartIndex"] = []
        sections.append(target)
    for kind in ("PINNED", "HIDDEN"):
        protected = checked(api._get(f"{sections_path}/protected", params={"type": kind}))
        if protected is not None:
            sections.append(protected)
    benchmark = next((key.rsplit("/", 1)[1] for key in media_keys if key in {
        "evaluation/ruler", "evaluation/locomo"
    }), None)
    legacy = "evaluation/checkpoints" + (f"/{benchmark}" if benchmark else "")
    found = {}
    for section in sections:
        for index in list(section["chartIndex"]):
            chart = checked(api._get(f"{charts_path}/xxxxxx/{index}"))
            axis = "yAxis" if chart["type"] == "LINE" else "keys"
            keys = {row["key"] for row in chart["config"].get(axis, [])}
            retired = keys and all(
                key.startswith(legacy + "/")
                and (benchmark is not None or not key.startswith((
                    "evaluation/checkpoints/ruler/", "evaluation/checkpoints/locomo/"
                )))
                for key in keys
            )
            if chart["type"] in {"LINE", "BAR", "SCALAR"} and retired:
                checked(api._delete(f"{charts_path}/{section['index']}/{index}"))
                if index in target["chartIndex"]:
                    target["chartIndex"].remove(index)
                continue
            if len(keys) == 1 and next(iter(keys)) in media_keys:
                key = next(iter(keys))
                if chart["type"] != "ECHARTS":
                    raise ValueError(f"unexpected SwanLab report panel type: {key}")
                if key in found:
                    raise ValueError(f"multiple SwanLab report panels for {key}")
                found[key] = (section["index"], index)
    missing = set(media_keys).difference(found)
    if missing:
        raise RuntimeError(f"SwanLab report panels are missing: {sorted(missing)}")
    for key in media_keys:
        section, index = found[key]
        if section != target["index"]:
            checked(api._post(
                f"{charts_path}/{section}/{index}/move", data={"index": target["index"]},
            ))
            target["chartIndex"].append(index)
    order = [found[key][1] for key in media_keys]
    order.extend(index for index in target["chartIndex"] if index not in order)
    checked(api._put(f"{sections_path}/{target['index']}", data={"chartIndex": order, "folded": False}))
    restored = checked(api._get(f"{sections_path}/{target['index']}"))
    if restored["chartIndex"] != order or restored.get("folded", False):
        raise RuntimeError("SwanLab report layout verification failed")
    return {
        "name": "evaluation", "index": view,
        "url": (
            f"{remote.url.split('/@', 1)[0]}/@{project_path}/v1/{view}"
            f"/runs/{remote.run_id}/chart"
        ),
    }


def publish_evaluations(run, evaluations):
    """合并已发布缓存与本次选择，按真实 step 重建完整报告。"""
    history = EvaluationHistory(evaluations[0][2], evaluations[0][1])
    for checkpoint, summary, output_dir in evaluations:
        if (
            summary["split"] != "test"
            or Path(summary["checkpoint"]["path"]).resolve() != Path(checkpoint).resolve()
            or Path(output_dir).resolve().parent != history.root
        ):
            raise ValueError("publication results must belong to the requested checkpoint and task")
        EvaluationHistory(output_dir, summary).validate_summary(summary)
        root, identity, project_path = validate_evaluation_publication(
            run, checkpoint, summary["checkpoint"]["step"]
        )
    with history.publication_lock():
        merged = {
            saved["checkpoint"]["path"]: (directory, saved)
            for directory, saved in history.published_summaries()
        }
        for _, summary, output_dir in evaluations:
            merged[summary["checkpoint"]["path"]] = (Path(output_dir).resolve(), summary)
        ordered = sorted(merged.values(), key=lambda pair: pair[1]["checkpoint"]["step"])
        summaries = [summary for _, summary in ordered]
        for directory, summary in ordered:
            EvaluationHistory(directory, summary).validate_summary(summary)
        signature = hashlib.sha256(json.dumps(
            summaries, ensure_ascii=False, sort_keys=True, allow_nan=False
        ).encode("utf-8")).hexdigest()
        publication = history.report_publication()
        if publication is not None and any(
            publication[name] != value for name, value in (
                ("run_id", identity["id"]), ("project", identity["project"]),
                ("group", identity["group"]),
            )
        ):
            raise ValueError("report publication differs from the original training run")
        values = evaluation_media(summaries)
        api_key = swanlab_api_key()
        api = swanlab.Api(api_key=api_key)
        remote = api.run(f"{project_path}/{identity['id']}")
        if remote.state != "FINISHED":
            raise ValueError("evaluation publication requires a finished training run")
        remote_config = {
            name: entry["value"] for name, entry in sorted(
                remote.profile["config"].items(), key=lambda pair: pair[1]["sort"]
            )
        }
        expected_config = SwanLabConfig()
        expected_config.update(run)
        if any(remote_config.get(name) != value for name, value in expected_config.items()):
            raise ValueError("SwanLab configuration differs from the saved training run")
        already_published = (
            publication is not None and publication["signature"] == signature
            and publication["panels"] == list(values)
        )
        if already_published:
            publication_step = publication["publication_step"]
        else:
            rng = capture_rng(torch.device("cpu"))
            interrupt_handler = signal.getsignal(signal.SIGINT)
            try:
                workspace, project = project_path.split("/")
                tracking = swanlab.init(
                    project=project, workspace=workspace, name=remote.name, config=remote_config,
                    job_type=identity["job_type"], group=identity["group"], tags=identity["tags"],
                    id=identity["id"], resume="must", mode="online", log_dir=str(root / "swanlab"),
                    settings=swanlab.Settings(
                        api_key=api_key, interactive=False, terminal={"proxy_type": "none"},
                        probe={"git": False, "monitor": False},
                    ),
                )
            finally:
                restore_rng(rng, torch.device("cpu"))
            try:
                signal.signal(signal.SIGINT, interrupt_handler)
                resumed_step = max((
                    metric.min_step for key, metric in tracking._ctx.core._metrics._metrics.items()
                    if key in values
                ), default=0)
                publication_step = max(
                    0, resumed_step, publication["publication_step"] if publication else -1
                ) + 1
                for key in values:
                    tracking.define_metric(key, section_name="evaluation")
                tracking.log(values, step=publication_step)
            finally:
                tracking.finish(state="success")
            _verify_report(api_key, project_path, identity["id"], values, publication_step)
        with (root.parents[2] / ".swanlab-panels.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            view = _configure_report_layout(api, remote, values)
        publication = {
            "training_dir": str(root), "run_id": identity["id"], "project": identity["project"],
            "group": identity["group"], "signature": signature,
            "summaries": [str(directory / "summary.json") for directory, _ in ordered],
            "publication_step": publication_step, "view": view, "panels": list(values),
        }
        write_json(history.publication_path, publication)
        print(
            f"[evaluation] report published steps={[row['checkpoint']['step'] for row in summaries]} "
            f"publication={publication_step} record={history.publication_path}", flush=True,
        )
        return publication
