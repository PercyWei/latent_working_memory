"""训练与 dev 共用原生增量曲线，恢复沿用同一 SwanLab run。"""

from contextlib import contextmanager
import fcntl
import json

import swanlab
import torch.distributed as dist

from latent_working_memory.runtime import capture_rng, restore_rng
from latent_working_memory.tracking_credentials import swanlab_api_key
from ..config import config_to_dict
from ..io import write_json


def configure_post_panels(api, remote, metrics):
    """为实际出现的当前指标注册原生图表并配置展示。"""
    def checked(response):
        if not response.ok:
            raise RuntimeError(response.errmsg)
        return response.data

    project_path = remote.url.split("/@", 1)[1].split("/runs/", 1)[0]
    # 与 SDK 的原生列注册使用同一接口；同步完成后再整理图表，避免首次 log 另建默认图。
    checked(api._post(f"/projects/{project_path}/series", data={"series": [
        {"key": metric, "type": "FLOAT", "sectionName": metric.split("/", 1)[0]}
        for metric in sorted(metrics)
    ]}))
    project = checked(api._get(f"/project/{project_path}"))
    view = project["viewIndex"][0]
    sections_path = f"/sections/{project_path}/{view}"
    sections = checked(api._get(sections_path, params={"size": 100}))
    charts_path = f"/charts/{project_path}/{view}"
    panels = {"train": (
        ("train/reward", "reward", (
            ("train/reward", "reward", "#2b8cbe"),
            ("train/reward_std", "reward_std", "#d95f0e"),
            ("train/baseline_reward", "greedy_reward", "#74a9cf"),
        )),
        ("train/memory_usage", "memory_blocks = memory_slots / K", (
            ("train/blocks", "memory_blocks", "#2b8cbe"),
        )),
        ("train/compression", "compression_rate = segment_tokens / K", (
            ("train/compression_ratio", "compression_rate", "#2b8cbe"),
        )),
        ("train/retention", "retention_ratio", (
            ("train/prefix_retention_ratio", "prefix_retention_ratio", "#2b8cbe"),
            ("train/retention_ratio", "retention_ratio", "#9ecae1"),
        )),
        ("train/learning_rate", "learning_rate", (
            ("train/compressor_learning_rate", "compressor", "#2b8cbe"),
            ("train/boundary_learning_rate", "boundary", "#d95f0e"),
            ("train/value_learning_rate", "value", "#31a354"),
        )),
    ), "dev": (
        ("dev/memory_usage", "memory_blocks = memory_slots / K", (
            ("dev/blocks", "memory_blocks", "#2b8cbe"),
        )),
        ("dev/compression", "compression_rate = segment_tokens / K", (
            ("dev/compression_ratio", "compression_rate", "#2b8cbe"),
        )),
    )}
    def condition_order(name):
        return name != "policy", int(name[6:]) if name.startswith("ratio_") else 0, name

    def lighter_color(color):
        return "#" + "".join(
            f"{round(int(color[start:start + 2], 16) * 0.55 + 255 * 0.45):02x}"
            for start in (1, 3, 5)
        )

    conditions = sorted(
        {name.split("/")[1] for name in metrics
         if name.startswith("dev/") and name.count("/") == 2},
        key=condition_order,
    )
    colors = {"policy": "#d62728", "single": "#2b8cbe", "sampled": "#2b8cbe"}
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b")
    ratios = [name for name in conditions if name.startswith("ratio_")]
    ratio_colors = dict(zip((2, 4, 8, 16), palette))
    colors.update({name: ratio_colors.get(int(name[6:]), palette[int(name[6:]) % len(palette)])
                   for name in ratios})
    light_colors = {name: lighter_color(color) for name, color in colors.items()}
    comparisons = []
    for metric in ("qa_nll", "ae_nll", "lm_nll", "em", "f1", "blocks", "objective"):
        series = tuple((f"dev/{condition}/{metric}", condition, colors[condition])
                       for condition in conditions if f"dev/{condition}/{metric}" in metrics)
        if series:
            comparisons.append((f"dev/conditions/{metric}", metric, series))
    retention_labels = {"prefix_retention_ratio": "prefix", "retention_ratio": "full"}
    retention_series = tuple(
        (f"dev/{condition}/{metric}", f"{condition}/{label}",
         (colors if label == "prefix" else light_colors)[condition])
        for condition in conditions for metric, label in retention_labels.items()
        if f"dev/{condition}/{metric}" in metrics
    )
    if retention_series:
        comparisons.append(("dev/conditions/retention", "retention_ratio", retention_series))
    panels["dev"] += tuple(comparisons)
    for split, split_panels in panels.items():
        if not any(metric.startswith(f"{split}/") for metric in metrics):
            continue
        section = next(row for row in sections if row["name"] == split)
        charts = [
            checked(api._get(f"{charts_path}/xxxxxx/{index}"))
            for index in section["chartIndex"]
        ]
        for title, axis_name, series in split_panels:
            keys = {metric for metric, _, _ in series}
            if not keys & metrics:
                continue
            names = {title, *keys}
            merged_retention = title == "dev/conditions/retention"
            if merged_retention:
                names.update(f"dev/conditions/{metric}" for metric in retention_labels)
            # 首次配置使用原生主指标图，后续运行复用同一项目中的当前面板。
            matching = [chart for chart in charts if chart["type"] == "LINE"
                        and chart["title"] in names]
            existing = next((chart for chart in matching if chart["title"] == title), matching[0])
            if title.startswith("dev/conditions/"):
                retained = {axis["key"] for chart in matching
                            for axis in chart["config"]["yAxis"]} - keys
                series += tuple((
                    key, (f"{key.split('/')[1]}/{retention_labels[key.rsplit('/', 1)[1]]}"
                          if merged_retention else key.split("/")[1]), None,
                ) for key in sorted(retained))
                series = tuple(sorted(series, key=lambda row: (
                    *condition_order(row[0].split("/")[1]), row[0],
                )))
                keys.update(retained)
            custom = {}
            for chart in charts:
                if chart["type"] == "LINE" and chart["title"] in names:
                    custom.update(chart.get("custom") or {})
            custom.update(existing.get("custom") or {})
            if merged_retention:
                # 旧分图的图例只有条件名；合图后历史运行也需要区分两类比例。
                for metric, label, _ in series:
                    for custom_key, style in custom.items():
                        if (custom_key.endswith(f"-{metric}")
                            and style["name"].endswith(f"/{metric.split('/')[1]}")):
                            custom[custom_key] = {
                                **style, "name": f"{style['name']}/{label.rsplit('/', 1)[1]}",
                                "colors": ([lighter_color(color) for color in style["colors"]]
                                           if label.endswith("/full") else style["colors"]),
                            }
            # 项目面板跨运行共享：保留其他策略已有的序列，只添加实际出现的新指标。
            displayed = keys & (set(metrics) | {
                axis["key"] for chart in charts
                if chart["type"] == "LINE" and chart["title"] in names
                for axis in chart["config"]["yAxis"]
            })
            for metric, label, color in series:
                if metric in metrics:
                    custom[f"{remote.run_id}-{metric}"] = {
                        "name": f"{remote.name}/{label}", "colors": [color, color],
                    }
            body = {
                "type": "LINE", "title": title, "custom": custom,
                "config": {
                    **(existing.get("config") or {}),
                    "xAxis": {"key": "step", "type": "SYSTEM", "class": "SCALAR"},
                    "yName": axis_name,
                    "yAxis": [
                        {"key": metric, "type": "FLOAT", "class": "SCALAR"}
                        for metric, _, _ in series if metric in displayed
                    ],
                },
            }
            checked(api._put(f"{charts_path}/xxxxxx/{existing['index']}", data=body))
            for chart in charts:
                if chart["index"] == existing["index"] or chart["type"] != "LINE":
                    continue
                chart_keys = {axis["key"] for axis in chart["config"]["yAxis"]}
                if chart["title"] in names and chart_keys and chart_keys <= keys:
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
        tags = {f"stage:{config.stage}", *training.swanlab.tags}
        if config.stage == "post":
            tags.add(f"method:update-{training.update_mode}")
        if config.uses_boundary_policy:
            tags.add("method:compression-policy")
            tags.add("method:tail-fixed-recent")
            tags.add(f"method:adaptive-boundary-{training.objective.strategy}")
        else:
            tags.update((f"method:compression-{training.compression.mode}",
                         f"method:objective-{training.objective.type}"))
            if training.compression.mode == "multi":
                tags.add("method:tail-fixed-recent")
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
                tags=sorted(tags),
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
            yield run
        finally:
            with preserve_training_rng(device, boundary_generator):
                run.finish()


def log_metrics(run, record, step, registered, stage, output_dir, objective=None):
    """只注册实际出现的指标，不为未执行的目标和 dev 创建空曲线。"""
    auxiliary = {
        "memory_slots", "recent_tokens", "pending_tokens", "segment_tokens", "memory_cost", "samples",
        "rollout_step", "actor_optimizer_step", "critic_optimizer_step", "optimizer_updates",
        "actor_loss", "critic_loss",
    }
    metrics = {name: value for name, value in record.items()
               if "/" in name and name.rsplit("/", 1)[-1] not in auxiliary
               and (stage != "post" or name not in {
                   "train/commit_rate", "train/memory_slots", "train/segment_tokens",
                   "dev/loss", "dev/samples", "dev/commit_rate",
                   "dev/memory_slots", "dev/segment_tokens",
               })}
    if (objective is not None and objective.strategy in {"scst", "ppo"} and objective.group_size == 1
        and "train/reward_std" in metrics):
        del metrics["train/reward_std"]
    new_metrics = metrics.keys() - registered
    if new_metrics:
        if stage == "post" or any(name.startswith("dev/") and name.count("/") == 2
                                  for name in new_metrics):
            with (output_dir.parents[1] / ".swanlab-panels.lock").open("a") as panel_lock:
                fcntl.flock(panel_lock, fcntl.LOCK_EX)
                api = swanlab.Api(api_key=swanlab_api_key())
                project_path = run.url.split("/@", 1)[1].split("/runs/", 1)[0]
                configure_post_panels(api, api.run(f"{project_path}/{run.id}"), new_metrics)
        for name in sorted(new_metrics):
            run.define_metric(name, x_axis="_step", section_name=name.split("/", 1)[0])
        registered.update(new_metrics)
    run.log(metrics, step=step)
