"""训练与 dev 共用原生增量曲线，恢复沿用同一 SwanLab run。"""

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json

import swanlab
import torch.distributed as dist

from latent_working_memory.runtime import capture_rng, restore_rng
from latent_working_memory.tracking_credentials import swanlab_api_key
from ..evaluation.records import write_json


@contextmanager
def preserve_training_rng(device, boundary_generator=None):
    rng = capture_rng(device)
    boundary = None if boundary_generator is None else boundary_generator.get_state()
    try:
        yield
    finally:
        restore_rng(rng, device)
        if boundary_generator is not None:
            boundary_generator.set_state(boundary)


@contextmanager
def tracking_run(config, output_dir, resume, device, boundary_generator=None):
    training = config.training
    if training.tracking == "disabled":
        yield None
        return
    with (output_dir / ".swanlab-publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output_dir / "swanlab.json"
        identity = json.loads(path.read_text(encoding="utf-8")) if resume else None
        workspace = None
        if identity is not None:
            project_path, run_id = identity["url"].split("/@", 1)[1].split("/runs/", 1)
            workspace, project = project_path.split("/")
            if project != training.swanlab_project or run_id != identity["id"]:
                raise ValueError("SwanLab URL differs from the saved project or run id")
        world = dist.get_world_size() if dist.is_initialized() else 1
        with preserve_training_rng(device, boundary_generator):
            run = swanlab.init(
                project=training.swanlab_project, workspace=workspace,
                name=None if resume else output_dir.parent.name,
                config=None if resume else {
                    **asdict(config), "world_size": world,
                    "effective_batch_size": world * training.micro_batch_size
                    * training.gradient_accumulation_steps,
                },
                group=training.group, job_type="train",
                tags=sorted({"method:adaptive-boundary-grpo", f"stage:{config.stage}", *training.tags}),
                mode="online", public=False, log_dir=str(output_dir / "swanlab"),
                id=identity["id"] if identity else None,
                resume="must" if identity else "never",
                settings=swanlab.Settings(
                    api_key=swanlab_api_key(), interactive=False,
                    terminal={"proxy_type": "none"}, probe={"git": False, "monitor": False},
                ),
            )
            if identity is None:
                write_json(path, {"id": run.id, "url": run.url})
        try:
            yield run
        finally:
            with preserve_training_rng(device, boundary_generator):
                run.finish()


def log_metrics(run, record, step, registered):
    """只注册实际出现的指标，不为未执行的目标和 dev 创建空曲线。"""
    metrics = {name: value for name, value in record.items() if "/" in name}
    for name in metrics:
        if name not in registered:
            run.define_metric(name, x_axis="_step", section_name=name.split("/", 1)[0])
            registered.add(name)
    run.log(metrics, step=step)
