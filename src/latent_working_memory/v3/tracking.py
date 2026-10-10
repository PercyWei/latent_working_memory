"""v3 方法级 SwanLab 身份、跨阶段增量曲线与最终评估汇总。"""

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import secrets

import swanlab
from swanlab.sdk.internal.run.components.config import Config as SwanLabConfig

from latent_working_memory.v1.reporting import _shade
from latent_working_memory.v3.config import DYNAMIC_METHODS, DYNAMIC_PRETRAIN_METHODS
from latent_working_memory.v4.checkpoint import capture_rng, restore_rng


TRAINING_METRICS = (
    "train/epoch",
    "train/grad_norm",
    "train/slots_final",
    "dev/slots_final",
    "resources/optimizer_step_seconds",
    "resources/peak_memory_allocated_gib",
)
METHOD_COLORS = {
    "icae_single": "#2459A6",
    "icae_multi": "#28764A",
    "autocompressors": "#7951A0",
    "memory_change": "#B45B18",
    "information_loss": "#A53651",
}
QUALITY_GROUPS = ("all", "old", "new")
STAGE_NUMBERS = {"pretrain": 1, "lm": 1, "qa": 2, "warmup": 2, "policy": 3}
DEV_QA_METRICS = ("dev/qa_old_nll", "dev/qa_new_nll")
APPEND_RATIO_METRICS = ("train/append_ratio", "dev/append_ratio")


def experiment_directory(training):
    """方法的持久化根目录；独立 train 命令默认使用阶段输出目录。"""
    return Path(training.experiment_dir or training.output_dir).resolve()


def experiment_name(training):
    """云端展示名与本地目录分开；未指定时沿用目录名称。"""
    return training.experiment_name or experiment_directory(training).name


def tracking_method(config, previous=None):
    if config.objective.method in DYNAMIC_PRETRAIN_METHODS and config.objective.stage == "pretrain":
        # 旧预训练会话继续沿用已保存的身份，不改名或改写云端配置。
        if previous is not None and previous["method"] == "shared-pretrain":
            return previous["method"]
        return "dynamic-pretrain"
    return config.objective.method


def _stage_record(run):
    record = deepcopy(run)
    sources = record.pop("pretraining_sources")
    record["pretraining_source_counts"] = {key: len(values) for key, values in sources.items()}
    if record["initialization"] is not None:
        record["initialization"].pop("pretraining_sources")
    return record


def _experiment_config(config, run, previous):
    """只增加新阶段；已有阶段配置与原始预训练来源必须保持不变。"""
    fixed = {
        "experiment_id": config.training.experiment_id,
        "method": tracking_method(config, previous),
        "model": run["config"]["model"],
        "resolved_model_revision": run["resolved_model_revision"],
    }
    if previous is not None:
        if any(previous[name] != value for name, value in fixed.items()):
            raise ValueError("method experiment identity or model configuration changed")
        if previous["pretraining"] is not None and previous["pretraining"] != run["pretraining"]:
            raise ValueError("method experiment cannot replace its original pretraining source")
    stages = {} if previous is None else dict(previous["stages"])
    stage = config.objective.stage
    current = _stage_record(run)
    if stage in stages and stages[stage] != current:
        raise ValueError(f"existing stage {stage} configuration cannot be replaced")
    if stages:
        latest = next(reversed(stages))
        if stage in stages and latest != stage:
            raise ValueError("cannot resume an earlier stage after a successor has started")
        if stage not in stages:
            previous_dir = Path(stages[latest]["config"]["training"]["output_dir"])
            result = json.loads((previous_dir / "training-result.json").read_text(encoding="utf-8"))
            initialization = run["initialization"]
            if (
                initialization is None
                or initialization["stage"] != latest
                or Path(initialization["checkpoint"]).resolve()
                != Path(result["checkpoint"]).resolve()
                or run["step_offset"] != result["global_step"]
            ):
                raise ValueError("new stage must continue the method's last completed checkpoint")
    stages[stage] = current
    return {**fixed, "pretraining": run["pretraining"], "stages": stages}


def _save_record(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _configure_append_ratio_panel(tracking, root, method, name, api_key):
    """在当前默认视图中合并训练／验证追加比例，保留其他 runs 的配色。"""
    with (root.parent / ".swanlab-panels.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        api = swanlab.Api(api_key=api_key)
        project_path = tracking.url.split("/@", 1)[1].split("/runs/", 1)[0]
        remote = api.run(f"{project_path}/{tracking.id}")

        def checked(response):
            if not response.ok:
                raise RuntimeError(response.errmsg)
            return response.data

        project = checked(api._get(f"/project/{project_path}"))
        view = project["viewIndex"][0]
        sections_path = f"/sections/{project_path}/{view}"
        sections = checked(api._get(sections_path, params={"size": 100}))
        section = next((row for row in sections if row["name"] == "train"), None)
        if section is None:
            section = checked(
                api._post(
                    sections_path,
                    data={"name": "train", "index": secrets.token_hex(3), "position": "above"},
                )
            )
            section["chartIndex"] = []
        charts_path = f"/charts/{project_path}/{view}"
        charts = [
            checked(api._get(f"{charts_path}/xxxxxx/{index}")) for index in section["chartIndex"]
        ]
        title = "train/append_ratio"
        existing = next(
            (chart for chart in charts if chart["title"] == title and chart["type"] == "LINE"), None
        )
        custom = dict((existing or {}).get("custom") or {})
        for index, metric in enumerate(APPEND_RATIO_METRICS):
            color = _shade(METHOD_COLORS[method], index, len(APPEND_RATIO_METRICS))
            custom[f"{remote.run_id}-{metric}"] = {
                "name": f"{name}/{metric.split('/')[0]}",
                "colors": [color, color],
            }
        body = {
            "type": "LINE",
            "title": title,
            "custom": custom,
            "config": {
                "xAxis": {"key": "step", "type": "SYSTEM", "class": "SCALAR"},
                "yAxis": [
                    {"key": metric, "type": "FLOAT", "class": "SCALAR"}
                    for metric in APPEND_RATIO_METRICS
                ],
                "yRange": [0, 1],
            },
        }
        if existing:
            checked(api._put(f"{charts_path}/xxxxxx/{existing['index']}", data=body))
        else:
            checked(api._post(f"{charts_path}/{section['index']}", data=body))


@contextmanager
def method_tracking_run(config, run, device, api_key=None):
    """建立方法级会话，跨阶段复用；已存在的 run 必须成功恢复。"""
    root = experiment_directory(config.training)
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "experiment.json"
    previous = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else None
    combined = _experiment_config(config, run, previous)
    if config.training.swanlab_project is None:
        _save_record(record_path, combined)
        yield None
        return
    method = combined["method"]
    data = (
        ("fineweb",)
        if method in {"dynamic-pretrain", "shared-pretrain", "autocompressors"}
        else (
            ("fineweb", "fineweb-factqa")
            if method in {"icae_single", "icae_multi"}
            else ("fineweb-factqa",)
        )
    )
    tags = sorted(
        {
            "scope:main",
            f"method:{method}",
            *(f"data:{name}" for name in data),
            *config.training.tags,
        }
    )
    expected_identity = {
        "project": config.training.swanlab_project,
        "group": config.training.group,
        "job_type": "train",
        "tags": tags,
        "mode": "online",
    }
    identity_path = root / "swanlab.json"
    identity = (
        json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else None
    )
    workspace = None
    upload_config = combined
    rng = capture_rng(device)
    try:
        if identity is not None:
            if previous is None:
                raise ValueError("existing SwanLab identity requires its method experiment.json")
            if any(identity[name] != value for name, value in expected_identity.items()):
                raise ValueError("SwanLab identity differs from the method experiment")
            project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
            workspace, project = project_path.split("/")
            if project != identity["project"]:
                raise ValueError("training run URL differs from its project")
            remote = swanlab.Api(api_key=api_key).run(f"{project_path}/{identity['id']}")
            # 同一会话可能在前驱阶段完成、后继阶段登记前失败；阶段衔接已由本地记录验证。
            if remote.state not in {"FINISHED", "CRASHED", "ABORTED"}:
                raise ValueError("SwanLab session state does not permit this stage continuation")
            remote_config = {
                name: entry["value"] for name, entry in remote.profile["config"].items()
            }
            # SDK 对顶层 None 等值有转换；仅在云端比较边界使用相同序列化规则。
            expected_config = SwanLabConfig()
            expected_config.update({name: previous[name] for name in combined})
            differences = [
                name
                for name, value in expected_config.items()
                if name not in remote_config or remote_config[name] != value
            ]
            if differences:
                raise ValueError(
                    "SwanLab configuration differs from the saved method experiment: "
                    + ", ".join(differences)
                )
            # 云端附加字段只保留在上传配置中，不写入本地方法的固定记录。
            upload_config = {**remote_config, **combined}
        tracking = swanlab.init(
            project=config.training.swanlab_project,
            workspace=workspace,
            name=experiment_name(config.training),
            config=upload_config,
            mode="online",
            public=False,
            job_type="train",
            group=config.training.group,
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
            _save_record(
                identity_path, {**expected_identity, "id": tracking.id, "url": tracking.url}
            )
        _save_record(record_path, combined)
        if method in DYNAMIC_METHODS and config.objective.stage in {"warmup", "policy"}:
            rng = capture_rng(device)
            try:
                _configure_append_ratio_panel(
                    tracking, root, method, experiment_name(config.training), api_key
                )
            finally:
                restore_rng(rng, device)
        yield tracking


def update_method_tracking(config, run, tracking):
    """验证阶段衔接并更新同一活跃 run，不结束会话或重新访问云端身份。"""
    record_path = experiment_directory(config.training) / "experiment.json"
    previous = json.loads(record_path.read_text(encoding="utf-8"))
    combined = _experiment_config(config, run, previous)
    if tracking is not None:
        tracking.config.update(combined)
    _save_record(record_path, combined)


def _loss_name(method, stage):
    if method == "autocompressors" and stage in {"pretrain", "lm"}:
        return "lm_loss"
    return "ae_lm_loss" if stage == "pretrain" else "qa_loss"


def configure_training_metrics(tracking, method, stage):
    """目标不同的损失分开展示，所有曲线沿用累计真实 optimizer step。"""
    if tracking is None:
        return
    names = (
        *TRAINING_METRICS,
        "train/stage",
        f"train/{_loss_name(method, stage)}",
        f"dev/{_loss_name(method, stage)}",
        *(DEV_QA_METRICS if stage in {"qa", "warmup", "policy"} else ()),
    )
    for name in names:
        tracking.define_metric(name, x_axis="_step", section_name=name.split("/", 1)[0])


def training_metrics(record, method):
    """只上传当前实际计算的指标，warmup 与 policy 共享 QA 目标曲线。"""
    values = {name: record[name] for name in TRAINING_METRICS if record.get(name) is not None}
    values["train/stage"] = STAGE_NUMBERS[record["stage"]]
    for section in ("train", "dev"):
        if record.get(f"{section}/loss") is not None:
            values[f"{section}/{_loss_name(method, record['stage'])}"] = record[f"{section}/loss"]
    if "resources/peak_memory_allocated_bytes" in record:
        values["resources/peak_memory_allocated_gib"] = (
            record["resources/peak_memory_allocated_bytes"] / 1024**3
        )
    for age in ("old", "new"):
        name = f"dev/qa_{age}_nll"
        if record.get(name) is not None and record[f"dev/qa_{age}_count"] > 0:
            values[name] = record[name]
    if method in DYNAMIC_METHODS and record["stage"] in {"warmup", "policy"}:
        for section in ("train", "dev"):
            if f"{section}/appends" in record:
                appends = record[f"{section}/appends"]
                decisions = appends + record[f"{section}/overwrites"]
                if decisions > 0:
                    values[f"{section}/append_ratio"] = appends / decisions
    return values


def evaluation_media(summary):
    """汇总质量、容量与每条轨迹的构建耗时；明细保留在本地产物中。"""
    method = summary["method"]
    color = METHOD_COLORS[method]
    groups = [name for name in QUALITY_GROUPS if summary["quality"][name]["questions"]]
    values = {}
    for metric in ("nll", "em", "f1"):
        chart = swanlab.echarts.Bar().add_xaxis(groups)
        chart.add_yaxis(
            method,
            [
                {
                    "value": summary["quality"][group][metric],
                    "itemStyle": {"color": _shade(color, index, len(groups))},
                }
                for index, group in enumerate(groups)
            ],
            label_opts={"show": False},
            itemstyle_opts={"color": color},
        )
        chart.set_global_opts(
            tooltip_opts={"trigger": "axis"},
            legend_opts={"type": "scroll", "top": 0},
            yaxis_opts={"name": metric, **({"min": 0, "max": 1} if metric != "nll" else {})},
        )
        values[f"evaluation/{metric}"] = chart
    summaries = (
        (
            "capacity",
            ["final_slots", "mean_slots"],
            [summary["capacity"][name] for name in ("final_slots", "mean_slots")],
            "slots",
        ),
        (
            "build_seconds_per_trajectory",
            ["build"],
            [summary["costs"]["build_seconds"] / summary["trajectories"]],
            "seconds / trajectory",
        ),
    )
    for name, labels, measurements, unit in summaries:
        chart = swanlab.echarts.Bar().add_xaxis(labels)
        chart.add_yaxis(
            method,
            [
                {
                    "value": value,
                    "itemStyle": {"color": _shade(color, index, len(labels))},
                }
                for index, value in enumerate(measurements)
            ],
            label_opts={"show": False},
            itemstyle_opts={"color": color},
        )
        chart.set_global_opts(
            tooltip_opts={"trigger": "axis"},
            legend_opts={"type": "scroll", "top": 0},
            yaxis_opts={"name": unit, "min": 0},
        )
        values[f"evaluation/{name}"] = chart
    return values
