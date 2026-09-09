"""SwanLab 实验初始化、续接与本地运行身份记录。"""

from __future__ import annotations

import json
import random
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import swanlab
import torch


SWANLAB_MODES = ("disabled", "offline", "online")


@contextmanager
def swanlab_run(
    output_dir: Path,
    config: dict[str, Any],
    mode: str = "disabled",
    project: str = "latent-working-memory",
    group: str | None = None,
    tags: tuple[str, ...] = (),
    run_id: str | None = None,
    job_type: str | None = None,
) -> Generator[swanlab.Run | None, None, None]:
    if mode not in SWANLAB_MODES:
        raise ValueError(f"SwanLab mode must be one of {SWANLAB_MODES}")
    if mode == "disabled":
        yield None
        return
    if not group:
        raise ValueError("enabled SwanLab runs require a group")
    if not job_type:
        raise ValueError("enabled SwanLab runs require a job type")
    if not tags:
        raise ValueError("enabled SwanLab runs require tags")
    tags = tuple(sorted(set(tags)))

    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "swanlab.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "project": project,
            "group": group,
            "job_type": job_type,
            "tags": list(tags),
        }
        if any(identity[key] != value for key, value in expected_metadata.items()) or (
            run_id is not None and identity["id"] != run_id
        ):
            raise ValueError("SwanLab metadata differs from the output directory")
        run_id = identity["id"]

    python_rng_state = random.getstate()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
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
        random.setstate(python_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_states:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    with run:
        identity = {
            "id": run.id,
            "project": project,
            "group": group,
            "job_type": job_type,
            "tags": list(tags),
            "mode": mode,
            "url": run.url if mode == "online" else None,
        }
        temporary = identity_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(identity_path)
        yield run
