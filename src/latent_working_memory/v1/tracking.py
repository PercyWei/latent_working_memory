"""跨阶段 SwanLab 会话、运行身份与随机状态保护。"""

from __future__ import annotations
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import swanlab
from latent_working_memory.v1.checkpoint import capture_rng_state, restore_rng_state


@contextmanager
def swanlab_run(
    output_dir: Path,
    config: dict[str, Any],
    mode: str = "disabled",
    project: str = "latent-working-memory",
    run_id: str | None = None,
    job_type: str = "train",
    group: str | None = None,
    tags: tuple[str, ...] = (),
    fixed_tags: tuple[str, ...] = ("scope:main", "method:latent-working-memory"),
    new_run: bool = False,
) -> Iterator[swanlab.Run | None]:
    if mode == "disabled":
        yield None
        return
    if not group:
        raise ValueError("enabled SwanLab runs require a group")
    fixed_tags = set(fixed_tags)
    tags = tuple(sorted(fixed_tags | set(tags)))
    identity_path = output_dir / "swanlab.json"
    if new_run and run_id is not None:
        raise ValueError("a new run cannot reuse an explicit run ID")
    if identity_path.exists() and not new_run:
        identity = json.loads(identity_path.read_text())
        if any(
            identity[k] != v
            for k, v in {
                "project": project,
                "group": group,
                "tags": list(tags),
                "job_type": job_type,
            }.items()
        ) or (run_id is not None and identity["id"] != run_id):
            raise ValueError("SwanLab project/run differs from the output directory")
        run_id = identity["id"]
    rng_state = capture_rng_state()
    try:
        run = swanlab.init(
            project=project,
            name=output_dir.name,
            config=config,
            mode=mode,
            public=False,
            job_type=job_type,
            group=group,
            tags=list(tags),
            log_dir=str(output_dir / "swanlab"),
            id=run_id,
            resume="allow" if run_id is not None else "never",
            settings=swanlab.Settings(
                interactive=False,
                terminal={"proxy_type": "none"},
                probe={"git": False, "monitor": False},
            ),
        )
    finally:
        restore_rng_state(rng_state)
    with run:
        identity_path.write_text(
            json.dumps(
                {
                    "id": run.id,
                    "project": project,
                    "group": group,
                    "tags": list(tags),
                    "job_type": job_type,
                    "mode": mode,
                    "url": run.url if mode == "online" else None,
                },
                indent=2,
            )
            + "\n"
        )
        yield run


@contextmanager
def swanlab_training_run(training_dir):
    """Resume a finished training run while preserving its identity and configuration."""
    identity = json.loads((training_dir / "swanlab.json").read_text())
    if identity["job_type"] != "train" or identity["mode"] != "online":
        raise ValueError("evaluation append requires an online training run")
    # The saved URL identifies the workspace as well as the project; IDs alone are not global.
    project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    workspace, project = project_path.split("/")
    if project != identity["project"]:
        raise ValueError("training run URL differs from its project")
    remote = swanlab.Api().run(f"{project_path}/{identity['id']}")
    if remote.state != "FINISHED":
        raise ValueError("append requires a finished training run; do not resume active training")
    # SwanLab's canonical API config is {key: {value, desc, sort}}.
    config = {
        key: item["value"]
        for key, item in sorted(remote.profile["config"].items(), key=lambda pair: pair[1]["sort"])
    }
    rng_state = capture_rng_state()
    try:
        run = swanlab.init(
            project=project,
            workspace=workspace,
            name=remote.name,
            config=config,
            id=identity["id"],
            resume="must",
            mode="online",
            log_dir=str(training_dir / "swanlab"),
            settings=swanlab.Settings(
                interactive=False,
                terminal={"proxy_type": "none"},
                probe={"git": False, "monitor": False},
            ),
        )
    finally:
        restore_rng_state(rng_state)
    with run:
        yield run
