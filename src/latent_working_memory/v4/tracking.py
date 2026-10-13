"""同一训练 run 的 SwanLab 身份和原生增量指标。"""

from contextlib import contextmanager
import json
from pathlib import Path

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


@contextmanager
def training_tracking(config, run, device, resume=False):
    if config.swanlab_project is None:
        yield None
        return
    root = Path(config.output_dir)
    identity_path = root / "swanlab.json"
    tags = sorted({"scope:main", "method:attention-matching", *config.tags})
    expected = {
        "project": config.swanlab_project,
        "group": config.group,
        "name": config.run_name or root.name,
        "job_type": "train",
        "tags": tags,
    }
    identity = None
    if resume:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if any(identity[name] != value for name, value in expected.items()):
            raise ValueError("SwanLab identity differs from the training run")
    rng = capture_rng(device)
    try:
        api_key = swanlab_api_key()
        api = swanlab.Api(api_key=api_key)
        workspace = api.username
        if identity is not None:
            project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
            workspace, project = project_path.split("/")
            if project != config.swanlab_project:
                raise ValueError("SwanLab URL differs from its saved project")
            remote = api.run(f"{project_path}/{identity['id']}")
            if remote.state not in {"FINISHED", "CRASHED", "ABORTED"}:
                raise ValueError("SwanLab run is still active; cannot resume training")
        if not api.project(f"{workspace}/{config.swanlab_project}").project_id:
            raise ValueError("online tracking requires an existing, explicitly selected project")
        tracking = swanlab.init(
            project=config.swanlab_project,
            workspace=workspace,
            name=expected["name"],
            config=run,
            mode="online",
            public=False,
            job_type="train",
            group=config.group,
            tags=tags,
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
        yield tracking


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
    """最终 checkpoint 的质量和写入诊断；样本、容量及资源明细留在本地。"""
    metrics = summary["metrics"]
    panels = [("nll", ["答案 NLL"], [metrics["nll"]], "NLL")]
    if "em" in metrics:
        panels.append(("em_f1", ["EM", "F1"], [metrics["em"], metrics["f1"]], "score"))
    panels.append(
        (
            "write_loss",
            ["写入前", "写入后"],
            [metrics["write_loss_before"], metrics["write_loss"]],
            "Attention Matching loss",
        )
    )
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


def validate_evaluation_publication(run, checkpoint_path, step):
    """在线评估前核对本地训练身份和最终 checkpoint；不访问远端。"""
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
        or checkpoint_path.parent != root / "checkpoints"
        or checkpoint_path != Path(result["checkpoint"]).resolve()
        or step != result["completed_steps"]
    ):
        raise ValueError("publication requires the completed training run's final checkpoint")
    identity = json.loads((root / "swanlab.json").read_text(encoding="utf-8"))
    expected = {
        "project": training["swanlab_project"],
        "group": training["group"],
        "name": training["run_name"] or root.name,
        "job_type": "train",
        "tags": sorted({"scope:main", "method:attention-matching", *training["tags"]}),
    }
    if any(identity[name] != value for name, value in expected.items()):
        raise ValueError("SwanLab identity differs from the original training run")
    project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    _, project = project_path.split("/")
    if project != identity["project"]:
        raise ValueError("training run URL differs from its project")
    return root, identity, project_path


def append_evaluation(run, checkpoint_path, summary, output_dir):
    """将最终 test 评估追加到已完成的原训练 run，保留其配置和运行身份。"""
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
    remote = swanlab.Api(api_key=api_key).run(f"{project_path}/{identity['id']}")
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
    values = evaluation_media(summary)
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
        for name in values:
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
        "panels": list(values),
        "metrics": summary["metrics"],
        "summary_file": str(Path(output_dir).resolve() / "summary.json"),
    }
    write_json(Path(output_dir) / "publication.json", publication)
    return publication
