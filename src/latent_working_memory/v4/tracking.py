"""同一训练 run 的 SwanLab 身份和原生增量指标。"""

from contextlib import contextmanager
import json
from pathlib import Path

import swanlab

from latent_working_memory.runtime import capture_rng, restore_rng
from latent_working_memory.tracking_credentials import swanlab_api_key


METRICS = (
    "train/loss",
    "dev/loss",
    "train/write_loss",
    "dev/write_loss",
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
            temporary = identity_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {**expected, "id": tracking.id, "url": tracking.url},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(identity_path)
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
