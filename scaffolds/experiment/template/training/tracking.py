"""训练与 dev 共用一个 SwanLab run；评估不经过这里。"""

from contextlib import contextmanager
import json

import swanlab

from ..execution.runtime import capture_rng, restore_rng
from ..execution.state import write_json


@contextmanager
def tracking_run(config, output_dir, resume, device, world_size):
    training = config["training"]
    if training["tracking"] == "disabled":
        yield None
        return
    path = output_dir / "swanlab.json"
    identity = json.loads(path.read_text(encoding="utf-8")) if resume else None
    if identity is not None and identity["project"] != training["swanlab_project"]:
        raise ValueError("SwanLab project differs from the original run")
    workspace = identity["url"].split("/@", 1)[1].split("/", 1)[0] if identity else None
    rng = capture_rng(device)
    try:
        run = swanlab.init(
            project=training["swanlab_project"],
            workspace=workspace,
            name=None if resume else output_dir.parent.name,
            config=None
            if resume
            else {
                **config,
                "world_size": world_size,
                "effective_batch_size": world_size
                * training["micro_batch_size"]
                * training["gradient_accumulation_steps"],
            },
            group=training["group"],
            job_type="train",
            tags=["method:experiment-template"],
            mode="online",
            public=False,
            log_dir=str(output_dir / "swanlab"),
            id=identity["id"] if identity else None,
            resume="must" if identity else "never",
            settings=swanlab.Settings(interactive=False),
        )
    finally:
        restore_rng(rng, device)
    with run:
        if identity is None:
            write_json(path, {"id": run.id, "url": run.url, "project": training["swanlab_project"]})
        yield run
