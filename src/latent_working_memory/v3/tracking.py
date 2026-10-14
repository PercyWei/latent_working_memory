"""v3 方法级 SwanLab 身份、跨阶段增量曲线与最终评估汇总。"""

from contextlib import contextmanager
from copy import deepcopy
import fcntl
from html import escape
import json
import math
from pathlib import Path
import secrets
import signal
import time

import swanlab
import torch
from swanlab.sdk.internal.run.components.config import Config as SwanLabConfig

from latent_working_memory.v1.reporting import _shade
from latent_working_memory.v3.config import DYNAMIC_METHODS, DYNAMIC_PRETRAIN_METHODS
from latent_working_memory.runtime import capture_rng, restore_rng


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
STAGE_NUMBERS = {"pretrain": 1, "lm": 1, "qa": 2, "warmup": 2, "policy": 3}
DEV_QA_METRICS = ("dev/qa_old_nll", "dev/qa_new_nll")
APPEND_RATIO_METRICS = ("train/append_ratio", "dev/append_ratio")
EVALUATION_STEP_METRIC = "checkpoint_comparison/checkpoint_step"
EVALUATION_VIEW_NAME = "checkpoint_comparison"
EVALUATION_METRICS = (
    "checkpoint_comparison/overall_nll",
    "checkpoint_comparison/overall_em",
    "checkpoint_comparison/overall_f1",
    "checkpoint_comparison/final_slots",
    "checkpoint_comparison/mean_slots",
    "checkpoint_comparison/mean_build_seconds",
)


def experiment_directory(training):
    """方法的持久化根目录；独立 train 命令默认使用阶段输出目录。"""
    return Path(training.experiment_dir or training.output_dir).resolve()


def experiment_name(training):
    """云端展示名与本地目录分开；未指定时沿用目录名称。"""
    return training.experiment_name or experiment_directory(training).name


def tracking_method(config, previous=None):
    if config.objective.method in DYNAMIC_PRETRAIN_METHODS and config.objective.stage == "pretrain":
        # 旧预训练会话继续沿用已保存的身份，不改名或改写云端配置。
        if previous is not None and previous["method"] in {"shared-pretrain", "dynamic-pretrain"}:
            return previous["method"]
        if config.objective.method == "dynamic":
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


def _configure_line_panels(
    tracking, root, method, name, section_name, panels, x_axis, api_key,
    view_name=None, previous_view_name=None,
):
    """配置共享原生曲线，指定名称时使用空白公共自定义视图。"""
    with (root.parent / ".swanlab-panels.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        api = swanlab.Api(api_key=api_key)
        project_path = tracking.url.split("/@", 1)[1].split("/runs/", 1)[0]
        remote = api.run(f"{project_path}/{tracking.id}")

        def checked(response):
            if not response.ok:
                raise RuntimeError(response.errmsg)
            return response.data

        if view_name is None:
            project = checked(api._get(f"/project/{project_path}"))
            view = project["viewIndex"][0]
        else:
            views_path = f"/views/{project_path}"
            matches = [
                row
                for row in checked(api._get(views_path, params={"name": view_name, "size": 50}))
                if row["name"] == view_name and row["type"] == "PUBLIC"
            ]
            if len(matches) > 1:
                raise ValueError(f"multiple public SwanLab views named {view_name}")
            if matches:
                record = checked(api._get(f"{views_path}/{matches[0]['index']}"))
            else:
                previous = (
                    [
                        row
                        for row in checked(
                            api._get(views_path, params={"name": previous_view_name, "size": 50})
                        )
                        if row["name"] == previous_view_name and row["type"] == "PUBLIC"
                        and row["locked"]
                    ]
                    if previous_view_name is not None else []
                )
                if len(previous) > 1:
                    raise ValueError(f"multiple public SwanLab views named {previous_view_name}")
                if previous:
                    view_path = f"{views_path}/{previous[0]['index']}"
                else:
                    # 官方网页创建自定义视图使用 locked=True；公开范围单独设置。
                    created = checked(api._post(views_path, data={"name": view_name, "locked": True}))
                    view_path = f"{views_path}/{created['index']}"
                checked(api._put(view_path, data={"name": view_name, "type": "PUBLIC"}))
                record = checked(api._get(view_path))
            if (
                not record["locked"]
                or record["type"] != "PUBLIC"
                or record["viewPermission"] != "WRITE"
            ):
                raise ValueError(f"SwanLab view {view_name} must be a writable public custom view")
            view = record["index"]
        sections_path = f"/sections/{project_path}/{view}"
        sections = checked(api._get(sections_path, params={"size": 100}))
        section = next((row for row in sections if row["name"] == section_name), None)
        if section is None and previous_view_name is not None:
            section = next((row for row in sections if row["name"] == previous_view_name), None)
            if section is not None:
                checked(api._put(f"{sections_path}/{section['index']}", data={"name": section_name}))
        if section is None:
            section = checked(
                api._post(
                    sections_path,
                    data={"name": section_name, "index": secrets.token_hex(3), "position": "above"},
                )
            )
            section["chartIndex"] = []
        charts_path = f"/charts/{project_path}/{view}"
        charts = [
            checked(api._get(f"{charts_path}/xxxxxx/{index}")) for index in section["chartIndex"]
        ]
        configured = []
        for title, metrics, labels, y_range in panels:
            existing = next(
                (
                    chart for chart in charts
                    if chart["type"] == "LINE" and (
                        chart["title"] == title
                        or previous_view_name is not None and chart["title"] == title.replace(
                            f"{section_name}/", f"{previous_view_name}/", 1
                        )
                    )
                ),
                None,
            )
            custom = dict((existing or {}).get("custom") or {})
            for index, (metric, label) in enumerate(zip(metrics, labels, strict=True)):
                color = _shade(METHOD_COLORS[method], index, len(metrics))
                custom[f"{remote.run_id}-{metric}"] = {
                    "name": f"{name}/{label}",
                    "colors": [color, color],
                }
            body = {
                "type": "LINE",
                "title": title,
                "custom": custom,
                "config": {
                    "xAxis": x_axis,
                    "fixedXAxis": True,
                    "yAxis": [
                        {"key": metric, "type": "FLOAT", "class": "SCALAR"}
                        for metric in metrics
                    ],
                    "yRange": y_range,
                },
            }
            if existing is None:
                existing = checked(api._post(f"{charts_path}/{section['index']}", data=body))
            configured.append((existing["index"], body))
        for index, body in configured:
            checked(api._put(f"{charts_path}/xxxxxx/{index}", data=body))
        return {
            "name": view_name,
            "index": view,
            "url": (
                f"{tracking.url.split('/@', 1)[0]}/@{project_path}/v1/{view}"
                f"/runs/{remote.run_id}/chart"
            ),
        }


def _configure_append_ratio_panel(tracking, root, method, name, api_key):
    _configure_line_panels(
        tracking,
        root,
        method,
        name,
        "train",
        [("train/append_ratio", APPEND_RATIO_METRICS, ("train", "dev"), [0, 1])],
        {"key": "step", "type": "SYSTEM", "class": "SCALAR"},
        api_key,
    )


def configure_evaluation_metrics(tracking):
    """以真实 checkpoint step 展示原生曲线，媒体发布步数独立递增。"""
    if tracking is None:
        return
    for metric in (EVALUATION_STEP_METRIC, *EVALUATION_METRICS):
        tracking.define_metric(
            metric,
            x_axis=EVALUATION_STEP_METRIC if metric != EVALUATION_STEP_METRIC else "_step",
            section_name=EVALUATION_VIEW_NAME,
            hidden=True,
        )


def configure_evaluation_layout(tracking, root, method, api_key):
    """checkpoint 对比区域配置四张汇总曲线；单个 checkpoint 保留单点。"""
    return _configure_line_panels(
        tracking,
        Path(root),
        method,
        tracking.name,
        EVALUATION_VIEW_NAME,
        [
            ("checkpoint_comparison/overall_nll", (EVALUATION_METRICS[0],), ("NLL",), [0, None]),
            ("checkpoint_comparison/overall_em_f1", EVALUATION_METRICS[1:3], ("EM", "F1"), [0, 1]),
            (
                "checkpoint_comparison/slots",
                EVALUATION_METRICS[3:5],
                ("final slots", "mean slots"),
                [0, None],
            ),
            (
                "checkpoint_comparison/mean_build_seconds",
                (EVALUATION_METRICS[5],),
                ("seconds / trajectory",),
                [0, None],
            ),
        ],
        {"key": EVALUATION_STEP_METRIC, "type": "FLOAT", "class": "SCALAR"},
        api_key,
        view_name=EVALUATION_VIEW_NAME,
        previous_view_name="evaluation",
    )


def configure_evaluation_visibility(training_dir, api_key):
    """将自动视图的旧结果面板和重复标量图移入隐藏分组，保留底层记录。"""
    identity = json.loads((training_dir / "swanlab.json").read_text(encoding="utf-8"))
    project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    retired = {
        EVALUATION_STEP_METRIC,
        *EVALUATION_METRICS,
        *(f"evaluation/{key.rsplit('/', 1)[-1]}" for key in (EVALUATION_STEP_METRIC, *EVALUATION_METRICS)),
        "evaluation/nll_delta_by_age",
        "evaluation/capacity",
        "evaluation/build_seconds_per_trajectory",
    }
    with (training_dir.parent / ".swanlab-panels.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        api = swanlab.Api(api_key=api_key)

        def checked(response):
            if not response.ok:
                raise RuntimeError(response.errmsg)
            return response.data

        views_path = f"/views/{project_path}"
        for row in checked(api._get(views_path, params={"size": 50})):
            if row["locked"]:
                continue
            view = checked(api._get(f"{views_path}/{row['index']}"))
            if view["viewPermission"] != "WRITE":
                continue
            sections_path = f"/sections/{project_path}/{view['index']}"
            charts_path = f"/charts/{project_path}/{view['index']}"
            sections = checked(api._get(sections_path, params={"size": 100}))
            hidden = None
            for section in sections:
                if section["name"] not in {"evaluation", EVALUATION_VIEW_NAME}:
                    continue
                for index in section["chartIndex"]:
                    chart = checked(api._get(f"{charts_path}/xxxxxx/{index}"))
                    columns = chart["config"]["yAxis"] if chart["type"] == "LINE" else chart["config"]["keys"]
                    keys = {column["key"] for column in columns}
                    if not keys or not keys.issubset(retired):
                        continue
                    if hidden is None:
                        hidden = checked(
                            api._get(f"{sections_path}/protected", params={"type": "HIDDEN"})
                        )["index"]
                    checked(
                        api._post(
                            f"{charts_path}/{section['index']}/{index}/move",
                            data={"index": hidden},
                        )
                    )


@contextmanager
def evaluation_tracking_run(training_dir, api_key):
    """追加评估并保留已完成训练的身份、配置与完成状态。"""
    identity = json.loads((training_dir / "swanlab.json").read_text(encoding="utf-8"))
    if identity["job_type"] != "train" or identity["mode"] != "online":
        raise ValueError("evaluation append requires an online training run")
    project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    workspace, project = project_path.split("/")
    if project != identity["project"]:
        raise ValueError("training run URL differs from its project")
    remote = swanlab.Api(api_key=api_key).run(f"{project_path}/{identity['id']}")
    if remote.state != "FINISHED":
        raise ValueError(
            f"evaluation append requires a finished training run, got {remote.state}: "
            f"{identity['url']}. If an earlier evaluation publication changed its state, "
            "confirm training succeeded and restore the completed state before retrying."
        )
    config = {
        key: item["value"]
        for key, item in sorted(remote.profile["config"].items(), key=lambda pair: pair[1]["sort"])
    }
    # 补传本地结果不需要初始化 CUDA；SDK 初始化只保护 CPU 随机状态。
    device = torch.device("cpu")
    rng = capture_rng(device)
    interrupt_handler = signal.getsignal(signal.SIGINT)
    try:
        tracking = swanlab.init(
            project=project,
            workspace=workspace,
            name=remote.name,
            config=config,
            id=identity["id"],
            resume="must",
            mode="online",
            log_dir=str(training_dir / "swanlab"),
            settings=swanlab.Settings(
                api_key=api_key,
                interactive=False,
                terminal={"proxy_type": "none"},
                probe={"git": False, "monitor": False},
            ),
        )
    finally:
        restore_rng(rng, device)
    try:
        # SDK 的 SIGINT hook 会先标记 ABORTED；交由本上下文统一结束补传。
        signal.signal(signal.SIGINT, interrupt_handler)
        yield tracking
    finally:
        # 发布异常继续向外抛出；不要让 SDK 的 __exit__ 把原训练标记为失败。
        tracking.finish(state="success")


def verify_evaluation_publication(training_dir, metrics, media_keys, step, api_key):
    """确认本次标量与图表已上传后，才能保存成功发布记录。"""
    identity = json.loads((training_dir / "swanlab.json").read_text(encoding="utf-8"))
    project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    for attempt in range(5):
        remote = swanlab.Api(api_key=api_key).run(f"{project_path}/{identity['id']}")
        scalars = remote.summary(keys=list(metrics))
        media = {
            item["key"]: item
            for item in remote.medias(keys=list(media_keys), step=step)["list"]
        }
        missing = [
            key
            for key, expected in metrics.items()
            if key not in scalars
            or scalars[key]["step"] != step
            or not math.isclose(scalars[key]["value"], expected, rel_tol=1e-6, abs_tol=1e-8)
        ]
        missing.extend(
            key
            for key in media_keys
            if key not in media
            or not any(
                item["index"] == step and item["items"] for item in media[key]["metrics"]
            )
        )
        if not missing and remote.state == "FINISHED":
            return
        if attempt < 4:
            time.sleep(2)
    raise RuntimeError(
        f"SwanLab publication verification failed at step {step}: "
        f"state={remote.state}, missing or mismatched metrics={missing}. "
        "Cached evaluation results are retained; rerun the same evaluation to publish them."
    )


def evaluation_scalars(summary):
    """最终质量按题平均，容量和构建耗时按轨迹平均。"""
    return {
        EVALUATION_STEP_METRIC: summary["metadata"]["global_step"],
        **{
            f"checkpoint_comparison/overall_{metric}": summary["quality"]["all"][metric]
            for metric in ("nll", "em", "f1")
        },
        **{
            f"checkpoint_comparison/{metric}": summary["capacity"][metric]
            for metric in ("final_slots", "mean_slots")
        },
        "checkpoint_comparison/mean_build_seconds": summary["costs"]["build_seconds"]
        / summary["trajectories"],
    }


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
            or method in DYNAMIC_METHODS
            and "pretrain" in combined["stages"]
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
        configure_posttraining_panel(config, tracking, device, api_key)
        yield tracking


def configure_posttraining_panel(config, tracking, device, api_key):
    """进入动态后训练前配置追加比例面板，保留训练随机状态。"""
    if config.objective.method not in DYNAMIC_METHODS or config.objective.stage not in {
        "warmup",
        "policy",
    }:
        return
    rng = capture_rng(device)
    try:
        _configure_append_ratio_panel(
            tracking,
            experiment_directory(config.training),
            config.objective.method,
            experiment_name(config.training),
            api_key,
        )
    finally:
        restore_rng(rng, device)


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
        *(DEV_QA_METRICS if method in DYNAMIC_METHODS and stage in {"warmup", "policy"} else ()),
    )
    for name in names:
        tracking.define_metric(name, x_axis="_step", section_name=name.split("/", 1)[0])


def training_metrics(record, method, qa_observations=None):
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
    if method in DYNAMIC_METHODS:
        for age in ("old", "new"):
            name = f"dev/qa_{age}_nll"
            if record.get(name) is not None and record[f"dev/qa_{age}_count"] > 0:
                values[name] = record[name]
    elif qa_observations:
        groups = {}
        for observation in qa_observations:
            groups.setdefault(observation["introduced_at"], []).append(observation["nll"])
        values["dev/nll"] = _introduction_nll_chart(
            method,
            {step: sum(losses) / len(losses) for step, losses in groups.items()},
            sum(row["nll"] for row in qa_observations) / len(qa_observations),
        )
    if method in DYNAMIC_METHODS and record["stage"] in {"warmup", "policy"}:
        for section in ("train", "dev"):
            if f"{section}/appends" in record:
                appends = record[f"{section}/appends"]
                decisions = appends + record[f"{section}/overwrites"]
                if decisions > 0:
                    values[f"{section}/append_ratio"] = appends / decisions
    return values


def _introduction_nll_chart(method, introduction, overall):
    steps = sorted(introduction, key=int)
    chart = swanlab.echarts.Bar().add_xaxis([str(step) for step in steps] + ["总体"])
    chart.add_yaxis(
        method,
        [introduction[step] for step in steps] + [overall],
        label_opts={"show": False},
        itemstyle_opts={"color": METHOD_COLORS[method]},
    )
    chart.set_global_opts(
        tooltip_opts={"trigger": "axis"},
        xaxis_opts={"name": "QA 首次引入段"},
        yaxis_opts={"name": "NLL", "min": 0},
    )
    return chart


def _evaluation_statistics_table(summary, title):
    rows = (
        ("最终记忆容量", summary["capacity"]["final_slots"], "slots"),
        ("平均记忆容量", summary["capacity"]["mean_slots"], "slots"),
        (
            "平均记忆构建耗时",
            summary["costs"]["build_seconds"] / summary["trajectories"],
            "秒 / 轨迹",
        ),
    )
    body = "".join(
        f"<tr><th scope='row'>{escape(label)}</th>"
        f"<td class='number'>{escape(f'{value:.6g}')}</td><td>{escape(unit)}</td></tr>"
        for label, value, unit in rows
    )
    return swanlab.Html(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title>"
        "<style>body{font:14px system-ui,sans-serif;margin:20px;color:#222}"
        "h1{font-size:15px;margin:0 0 16px}table{border-collapse:collapse;width:100%}"
        "th,td{padding:10px 12px;border-bottom:1px solid #ddd;text-align:left}"
        "thead th{background:#f5f5f5}tbody th{font-weight:normal}"
        ".number{font-variant-numeric:tabular-nums;text-align:right}"
        "p{font-size:12px;color:#666}</style></head><body>"
        f"<h1>{escape(title)}</h1><table><thead><tr><th>指标</th>"
        "<th class='number'>数值</th><th>单位</th></tr></thead>"
        f"<tbody>{body}</tbody></table><p>容量与构建耗时均按轨迹平均。</p></body></html>",
        caption=title,
    )


def evaluation_media(summary):
    """展示当前 checkpoint 的最终质量、动态保留热图及资源统计。"""
    method = summary["method"]
    color = METHOD_COLORS[method]
    introduction = summary["quality"]["by_introduction"]
    steps = sorted(introduction, key=int)
    labels = [str(step) for step in steps] + ["总体"]
    rows = [introduction[step] for step in steps] + [summary["quality"]["all"]]
    section = "dev" if summary["split"] == "dev" else "evaluation"
    values = {
        f"{section}/nll": _introduction_nll_chart(
            method,
            {step: row["nll"] for step, row in introduction.items()},
            summary["quality"]["all"]["nll"],
        )
    }
    if summary["split"] == "test":
        chart = swanlab.echarts.Bar().add_xaxis(labels)
        for index, metric in enumerate(("em", "f1")):
            chart.add_yaxis(
                metric.upper(),
                [row[metric] for row in rows],
                yaxis_index=0,
                label_opts={"show": False},
                itemstyle_opts={"color": _shade(color, index, 2)},
            )
        chart.set_global_opts(
            tooltip_opts={"trigger": "axis"},
            legend_opts={"top": 0},
            xaxis_opts={"name": "QA 首次引入段"},
            yaxis_opts={"name": "EM / F1", "min": 0, "max": 1},
        )
        values[f"{section}/em_f1"] = chart
    if method in DYNAMIC_METHODS:
        retention = summary["nll_retention"]
        count = max(cell["read_step"] for cell in retention["cells"])
        axis = [str(step) for step in range(1, count + 1)]
        chart = swanlab.echarts.HeatMap().add_xaxis(axis)
        chart.add_yaxis(
            method,
            axis,
            [
                [cell["read_step"] - 1, cell["introduced_at"] - 1, cell["nll"]]
                for cell in retention["cells"]
            ],
            label_opts={"show": False},
        )
        chart.set_global_opts(
            xaxis_opts={"name": "已读来源段", "type": "category"},
            yaxis_opts={"name": "问题引入段", "type": "category"},
            visualmap_opts={
                "min": 0,
                "max": max(cell["nll"] for cell in retention["cells"]),
                "orient": "horizontal",
                "bottom": 0,
                "inRange": {"color": [_shade(color, 1, 2), color]},
            },
        )
        values[f"{section}/nll_retention"] = chart
    metadata = summary.get("metadata", {})
    title = f"{method} · {summary['split']} 统计"
    if summary["split"] == "test" and "checkpoint" in metadata:
        title = (
            f"{summary['stage']} · stage step {metadata['stage_step']} "
            f"· global step {metadata['global_step']}"
        )
        for chart in values.values():
            chart.options["title"] = {
                "text": title,
                "left": "center",
                "top": 0,
                "textStyle": {"fontSize": 12},
            }
            for legend in chart.options["legend"]:
                legend["top"] = 24
    values[f"{section}/statistics"] = _evaluation_statistics_table(summary, title)
    return values
