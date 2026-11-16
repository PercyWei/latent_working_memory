"""训练与 dev 共用原生增量曲线，恢复沿用同一 SwanLab run。"""

from contextlib import contextmanager
import fcntl
import json
import secrets

import swanlab
import torch.distributed as dist

from latent_working_memory.runtime import capture_rng, restore_rng
from latent_working_memory.tracking_credentials import swanlab_api_key
from ..config import config_to_dict
from ..io import write_json


def configure_post_panels(api, remote):
    """沿用 v4.1 的原生面板配置，只整理后训练图表，不删除历史标量。"""
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
    section = next((row for row in sections if row["name"] == "train"), None)
    if section is None:
        section = checked(api._post(
            sections_path,
            data={"name": "train", "index": secrets.token_hex(3), "position": "above"},
        ))
        section["chartIndex"] = []
    charts = [
        checked(api._get(f"{charts_path}/xxxxxx/{index}"))
        for index in section["chartIndex"]
    ]
    panels = (
        ("train/reward", "reward", (
            ("train/reward", "reward", "#2b8cbe"),
            ("train/reward_std", "reward_std", "#d95f0e"),
        ), ()),
        ("train/memory_usage", "blocks = memory_slots / K", (
            ("train/blocks", "blocks", "#2b8cbe"),
        ), ("train/memory_slots",)),
        ("train/compression", "压缩倍率 = segment_tokens / K", (
            ("train/compression_ratio", "compression_ratio", "#2b8cbe"),
        ), ("train/segment_tokens",)),
        ("train/learning_rate", "learning_rate", (
            ("train/compressor_learning_rate", "compressor", "#2b8cbe"),
            ("train/boundary_learning_rate", "boundary", "#d95f0e"),
        ), ()),
    )
    for title, axis_name, series, replaced in panels:
        metrics = {metric for metric, _, _ in series}
        names = {title, *metrics, *replaced}
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
                "name": f"{remote.name}/{label}", "colors": [color, color],
            }
        body = {
            "type": "LINE", "title": title, "custom": custom,
            "config": {
                **((existing or {}).get("config") or {}),
                "xAxis": {"key": "step", "type": "SYSTEM", "class": "SCALAR"},
                "yName": axis_name,
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
            if chart["title"] in names and keys and keys <= metrics | set(replaced):
                checked(api._delete(f"{charts_path}/{section['index']}/{chart['index']}"))
    # train/samples 还用于预训练，保留共享面板；后训练不再记录该项。
    for chart in charts:
        name = chart["title"]
        if chart["type"] == "LINE" and name in {
            "train/commit_rate", "train/zero_advantage_groups",
        } and {axis["key"] for axis in chart["config"]["yAxis"]} == {name}:
            checked(api._delete(f"{charts_path}/{section['index']}/{chart['index']}"))


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
    if not training.swanlab.tracking:
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
            if project != training.swanlab.project or run_id != identity["id"]:
                raise ValueError("SwanLab URL differs from the saved project or run id")
        world = dist.get_world_size() if dist.is_initialized() else 1
        with preserve_training_rng(device, boundary_generator):
            api_key = swanlab_api_key()
            run = swanlab.init(
                project=training.swanlab.project, workspace=workspace,
                name=None if resume else output_dir.parent.name,
                config=None if resume else {
                    **config_to_dict(config), "world_size": world,
                    "effective_batch_size": world * training.loop.micro_batch_size
                    * training.loop.gradient_accumulation_steps,
                },
                group=training.swanlab.group, job_type="train",
                tags=sorted({"method:adaptive-boundary-grpo", f"stage:{config.stage}", *training.swanlab.tags}),
                mode="online", public=False, log_dir=str(output_dir / "swanlab"),
                id=identity["id"] if identity else None,
                resume="must" if identity else "never",
                settings=swanlab.Settings(
                    api_key=api_key, interactive=False,
                    terminal={"proxy_type": "none"}, probe={"git": False, "monitor": False},
                ),
            )
            if identity is None:
                write_json(path, {"id": run.id, "url": run.url})
        try:
            if config.stage == "post":
                with preserve_training_rng(device, boundary_generator):
                    with (output_dir.parents[1] / ".swanlab-panels.lock").open("a") as panel_lock:
                        fcntl.flock(panel_lock, fcntl.LOCK_EX)
                        api = swanlab.Api(api_key=api_key)
                        project_path = run.url.split("/@", 1)[1].split("/runs/", 1)[0]
                        configure_post_panels(api, api.run(f"{project_path}/{run.id}"))
            yield run
        finally:
            with preserve_training_rng(device, boundary_generator):
                run.finish()


def log_metrics(run, record, step, registered):
    """只注册实际出现的指标，不为未执行的目标和 dev 创建空曲线。"""
    metrics = {name: value for name, value in record.items()
               if "/" in name and name not in {
                   "train/commit_rate", "train/memory_slots", "train/segment_tokens",
               }}
    for name in metrics:
        if name not in registered:
            run.define_metric(name, x_axis="_step", section_name=name.split("/", 1)[0])
            registered.add(name)
    run.log(metrics, step=step)
