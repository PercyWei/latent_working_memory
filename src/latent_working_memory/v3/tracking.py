"""v3 方法级 SwanLab 身份、跨阶段增量曲线与最终评估汇总。"""

from colorsys import hsv_to_rgb
from contextlib import contextmanager
from copy import deepcopy
import fcntl
from hashlib import sha256
from html import escape
import json
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
EVALUATION_MEDIA_TYPES = {
    "evaluation/nll": "ECHARTS",
    "evaluation/em": "ECHARTS",
    "evaluation/f1": "ECHARTS",
    "evaluation/statistics": "HTML",
}


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


def _default_view(api, project_path):
    """按项目视图顺序选择可写的自动视图，跳过空白自定义视图。"""
    response = api._get(f"/project/{project_path}")
    if not response.ok:
        raise RuntimeError(response.errmsg)
    for index in response.data["viewIndex"]:
        response = api._get(f"/views/{project_path}/{index}")
        if not response.ok:
            raise RuntimeError(response.errmsg)
        view = response.data
        if not view["locked"] and view["viewPermission"] == "WRITE":
            return view["index"]
    raise ValueError("SwanLab project requires a writable default view")


def _configure_line_panels(
    root, method, project_path, run_id, name, section_name, panels, x_axis, api_key,
    view_name=None, previous_view_name=None,
):
    """配置共享原生曲线，指定名称时使用空白公共自定义视图。"""
    with (root.parent / ".swanlab-panels.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        api = swanlab.Api(api_key=api_key)
        remote = api.run(f"{project_path}/{run_id}")
        name = remote.name if name is None else name

        def checked(response):
            if not response.ok:
                raise RuntimeError(response.errmsg)
            return response.data

        if view_name is None:
            view = _default_view(api, project_path)
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
                checked(api._post(f"{charts_path}/{section['index']}", data=body))
            elif (
                any(existing.get(key) != body[key] for key in ("type", "title", "custom"))
                or any(existing["config"].get(key) != value for key, value in body["config"].items())
            ):
                checked(api._put(f"{charts_path}/xxxxxx/{existing['index']}", data=body))
        return {
            "name": view_name,
            "index": view,
            "url": (
                f"{remote.url.split('/@', 1)[0]}/@{project_path}/v1/{view}"
                f"/runs/{remote.run_id}/chart"
            ),
        }


def _configure_append_ratio_panel(tracking, root, method, name, api_key):
    _configure_line_panels(
        root,
        method,
        tracking.url.split("/@", 1)[1].split("/runs/", 1)[0],
        tracking.id,
        name,
        "train",
        [("train/append_ratio", APPEND_RATIO_METRICS, ("train", "dev"), [0, 1])],
        {"key": "step", "type": "SYSTEM", "class": "SCALAR"},
        api_key,
    )


def configure_evaluation_layout(training_dir, media_types, api_key):
    """恢复 evaluation 媒体面板、清理旧布局，并按 checkpoint 顺序排列热图。"""
    training_dir = Path(training_dir)
    identity = json.loads((training_dir / "swanlab.json").read_text(encoding="utf-8"))
    project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    # 这些 key 仅用于清理已有云端面板；当前发布路径只记录媒体。
    retired_scalars = {
        f"{prefix}/{suffix}"
        for prefix in ("evaluation", "checkpoint_comparison")
        for suffix in (
            "checkpoint_step", "overall_nll", "overall_em", "overall_f1",
            "final_slots", "mean_slots", "mean_build_seconds",
        )
    }
    retired_media = {
        "evaluation/em_f1", "evaluation/nll_retention", "evaluation/nll_delta_by_age",
        "evaluation/capacity", "evaluation/build_seconds_per_trajectory",
    }
    retired = retired_scalars | retired_media
    with (training_dir.parent / ".swanlab-panels.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        api = swanlab.Api(api_key=api_key)

        def checked(response):
            if not response.ok:
                raise RuntimeError(response.errmsg)
            return response.data

        def panels(view):
            sections_path = f"/sections/{project_path}/{view}"
            sections = checked(api._get(sections_path, params={"size": 100}))
            hidden = None
            for kind in ("PINNED", "HIDDEN"):
                protected = checked(api._get(f"{sections_path}/protected", params={"type": kind}))
                if protected is not None:
                    sections.append(protected)
                if kind == "HIDDEN":
                    hidden = protected
            charts_path = f"/charts/{project_path}/{view}"
            return sections, [
                (section, checked(api._get(f"{charts_path}/xxxxxx/{index}")))
                for section in sections for index in section["chartIndex"]
            ], hidden

        def keys(chart):
            columns = chart["config"]["yAxis" if chart["type"] == "LINE" else "keys"]
            return {column["key"] for column in columns}

        # 删除旧的专用趋势视图以及原生重复面板，底层指标数据仍保留。
        default_view = _default_view(api, project_path)
        views_path = f"/views/{project_path}"
        for row in checked(api._get(views_path, params={"size": 50})):
            view = checked(api._get(f"{views_path}/{row['index']}"))
            if view["viewPermission"] != "WRITE":
                continue
            sections, charts, hidden = panels(view["index"])
            old_charts = [
                (section, chart)
                for section, chart in charts
                if chart["type"] in {"LINE", "BAR", "SCALAR", "ECHARTS", "HTML"}
                and keys(chart) and keys(chart).issubset(retired)
            ]
            if (
                view["index"] != default_view
                and view["locked"] and view["type"] == "PUBLIC"
                and view["name"] in {"checkpoint_comparison", "evaluation"}
                and len(old_charts) == len(charts)
            ):
                checked(api._delete(f"{views_path}/{view['index']}"))
                continue
            charts_path = f"/charts/{project_path}/{view['index']}"
            for section, chart in old_charts:
                if chart["type"] in {"LINE", "BAR", "SCALAR"}:
                    checked(api._delete(f"{charts_path}/{section['index']}/{chart['index']}"))
                elif hidden is None or section["index"] != hidden["index"]:
                    # 官方媒体菜单只提供隐藏，按已确认的 move 接口保留原始记录。
                    if hidden is None:
                        hidden = checked(
                            api._get(
                                f"/sections/{project_path}/{view['index']}/protected",
                                params={"type": "HIDDEN"},
                            )
                        )
                    checked(
                        api._post(
                            f"{charts_path}/{section['index']}/{chart['index']}/move",
                            data={"index": hidden["index"]},
                        )
                    )
            sections_path = f"/sections/{project_path}/{view['index']}"
            for section in checked(api._get(sections_path, params={"size": 100})):
                if section["name"] == "checkpoint_comparison" and not section["chartIndex"]:
                    checked(api._delete(f"{sections_path}/{section['index']}"))

        sections_path = f"/sections/{project_path}/{default_view}"
        charts_path = f"/charts/{project_path}/{default_view}"
        sections, charts, _ = panels(default_view)
        found = {}
        for section, chart in charts:
            if chart["type"] not in {"ECHARTS", "HTML"}:
                continue
            columns = chart["config"]["keys"]
            if len(columns) != 1 or columns[0]["key"] not in media_types:
                continue
            key = columns[0]["key"]
            if chart["type"] != media_types[key]:
                raise ValueError(f"SwanLab evaluation panel has an unexpected type: {key}")
            if key in found:
                raise ValueError(f"multiple SwanLab evaluation panels for {key}")
            found[key] = (section["index"], chart["index"])
        missing = set(media_types).difference(found)
        if missing:
            raise RuntimeError(f"SwanLab evaluation media panels are missing: {sorted(missing)}")
        matches = [section for section in sections if section["name"] == "evaluation"]
        if len(matches) > 1:
            raise ValueError("multiple SwanLab sections named evaluation")
        if matches:
            target = matches[0]
        else:
            target = checked(
                api._post(
                    sections_path,
                    data={"name": "evaluation", "index": secrets.token_hex(3), "position": "above"},
                )
            )
            target["chartIndex"] = []
        for key in media_types:
            source, index = found[key]
            if source == target["index"]:
                continue
            target["chartIndex"].append(index)
            checked(
                api._post(
                    f"{charts_path}/{source}/{index}/move",
                    data={"index": target["index"], "chartIndex": target["chartIndex"]},
                )
            )
        expected = [found[key][1] for key in media_types]
        order = expected + [index for index in target["chartIndex"] if index not in expected]
        checked(
            api._put(
                f"{sections_path}/{target['index']}",
                data={"chartIndex": order, "folded": False},
            )
        )
        restored = checked(api._get(f"{sections_path}/{target['index']}"))
        if (
            restored["name"] != "evaluation" or restored["chartIndex"] != order
            or restored.get("folded", False)
        ):
            raise RuntimeError("SwanLab evaluation layout verification failed")
        return {
            "name": "evaluation",
            "index": default_view,
            "url": (
                f"{identity['url'].split('/@', 1)[0]}/@{project_path}/v1/{default_view}"
                f"/runs/{identity['id']}/chart"
            ),
        }


def evaluation_training_run(training_dir, api_key):
    """校验原在线训练身份和完成状态，返回可追加评估的云端 run。"""
    identity = json.loads((training_dir / "swanlab.json").read_text(encoding="utf-8"))
    if identity["job_type"] != "train" or identity["mode"] != "online":
        raise ValueError("evaluation append requires an online training run")
    project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    _, project = project_path.split("/")
    if project != identity["project"]:
        raise ValueError("training run URL differs from its project")
    remote = swanlab.Api(api_key=api_key).run(f"{project_path}/{identity['id']}")
    if remote.state != "FINISHED":
        raise ValueError(
            f"evaluation append requires a finished training run, got {remote.state}: "
            f"{identity['url']}. If an earlier evaluation publication changed its state, "
            "confirm training succeeded and restore the completed state before retrying."
        )
    return remote


@contextmanager
def evaluation_tracking_run(training_dir, api_key):
    """追加评估并保留已完成训练的身份、配置与完成状态。"""
    identity = json.loads((training_dir / "swanlab.json").read_text(encoding="utf-8"))
    project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    workspace, project = project_path.split("/")
    remote = evaluation_training_run(training_dir, api_key)
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


def verify_evaluation_publication(training_dir, media_keys, step, api_key):
    """确认汇总图和各 checkpoint 热图已上传，才能保存成功发布记录。"""
    identity = json.loads((training_dir / "swanlab.json").read_text(encoding="utf-8"))
    project_path = identity["url"].split("/@", 1)[1].split("/runs/", 1)[0]
    for attempt in range(5):
        remote = swanlab.Api(api_key=api_key).run(f"{project_path}/{identity['id']}")
        media = {
            item["key"]: item
            for item in remote.medias(keys=list(media_keys), step=step)["list"]
        }
        missing = [
            key for key in media_keys
            if key not in media
            or not any(item["index"] == step and item["items"] for item in media[key]["metrics"])
        ]
        if not missing and remote.state == "FINISHED":
            return
        if attempt < 4:
            time.sleep(2)
    raise RuntimeError(
        f"SwanLab publication verification failed at step {step}: "
        f"state={remote.state}, missing media={missing}. "
        "Cached evaluation results are retained; rerun the same evaluation to publish them."
    )


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


def _checkpoint_label(summary):
    metadata = summary["metadata"]
    return (
        f"{summary['stage']} · stage step {metadata['stage_step']} "
        f"· global step {metadata['global_step']}"
    )


def _checkpoint_color(summary):
    """颜色只取决于方法和真实 step，补评其他 checkpoint 时保持不变。"""
    identity = f"{summary['method']}:{summary['metadata']['global_step']}"
    hue = int.from_bytes(sha256(identity.encode()).digest()[:4], "big") / 2**32
    channels = hsv_to_rgb(hue, 0.65, 0.75)
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in channels)


def _evaluation_quality_chart(summaries, metric, introduction_steps):
    labels = [str(step) for step in introduction_steps] + ["总体"]
    chart = swanlab.echarts.Line().add_xaxis(labels)
    for summary in summaries:
        introduction = summary["quality"]["by_introduction"]
        rows = [introduction.get(str(step)) for step in introduction_steps]
        rows.append(summary["quality"]["all"])
        color = _checkpoint_color(summary)
        chart.add_yaxis(
            _checkpoint_label(summary),
            [row[metric] if row is not None else None for row in rows],
            color=color,
            is_symbol_show=True,
            symbol="circle",
            symbol_size=6,
            label_opts={"show": False},
            linestyle_opts={"color": color, "width": 2},
            itemstyle_opts={"color": color},
        )
    name = metric.upper()
    chart.set_global_opts(
        tooltip_opts={"trigger": "axis"},
        legend_opts={"type": "scroll", "top": 28},
        xaxis_opts={"name": "QA 首次引入段", "type": "category"},
        yaxis_opts={"name": name, "min": 0, **({"max": 1} if metric != "nll" else {})},
    )
    chart.options["title"] = {
        "text": f"最终压缩后 {name}",
        "left": "center",
        "top": 0,
        "textStyle": {"fontSize": 14},
    }
    chart.options["grid"] = {"top": 85, "bottom": 60, "left": 65, "right": 100}
    return chart


def _evaluation_statistics_table(summaries):
    headers = (
        "阶段", "阶段 step", "累计 step", "QA 数", "NLL", "EM", "F1",
        "最终 slots", "平均 slots", "构建秒 / 轨迹",
    )
    body = []
    for summary in summaries:
        metadata = summary["metadata"]
        quality = summary["quality"]["all"]
        values = (
            metadata["stage_step"], metadata["global_step"], quality["questions"],
            quality["nll"], quality["em"], quality["f1"],
            summary["capacity"]["final_slots"], summary["capacity"]["mean_slots"],
            summary["costs"]["build_seconds"] / summary["trajectories"],
        )
        cells = "".join(
            f"<td class='number'>{escape(str(value) if isinstance(value, int) else format(value, '.6g'))}</td>"
            for value in values
        )
        body.append(f"<tr><th scope='row'>{escape(summary['stage'])}</th>{cells}</tr>")
    head = "".join(f"<th>{escape(label)}</th>" for label in headers)
    title = "Checkpoint 评估统计"
    return swanlab.Html(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title>"
        "<style>body{font:14px system-ui,sans-serif;margin:20px;color:#222}"
        "h1{font-size:15px;margin:0 0 16px}.table-wrap{overflow-x:auto}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{padding:10px 12px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:left}"
        "thead th{background:#f5f5f5}tbody th{font-weight:normal}"
        ".number{font-variant-numeric:tabular-nums;text-align:right}"
        "p{font-size:12px;color:#666}</style></head><body>"
        f"<h1>{escape(title)}</h1><div class='table-wrap'><table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
        "<p>质量按题平均；容量与构建耗时按轨迹平均。所有 checkpoint 按累计 step 排序。</p>"
        "</body></html>",
        caption=title,
    )


def evaluation_panel_types(summaries):
    """最终评估的完整面板顺序：质量合图、统计表、逐 checkpoint 热图。"""
    return {
        **EVALUATION_MEDIA_TYPES,
        **{
            f"evaluation/nll_retention/step_{summary['metadata']['global_step']}": "ECHARTS"
            for summary in sorted(summaries, key=lambda row: row["metadata"]["global_step"])
            if summary["method"] in DYNAMIC_METHODS
        },
    }


def evaluation_media(summaries):
    """合并全部 checkpoint 的最终质量；动态热图共享坐标域和色阶并逐张展开。"""
    summaries = sorted(summaries, key=lambda row: row["metadata"]["global_step"])
    introduction_steps = sorted(
        {int(step) for summary in summaries for step in summary["quality"]["by_introduction"]}
    )
    values = {
        f"evaluation/{metric}": _evaluation_quality_chart(summaries, metric, introduction_steps)
        for metric in ("nll", "em", "f1")
    }
    values["evaluation/statistics"] = _evaluation_statistics_table(summaries)
    dynamic = [summary for summary in summaries if summary["method"] in DYNAMIC_METHODS]
    if not dynamic:
        return values
    cells = [cell for summary in dynamic for cell in summary["nll_retention"]["cells"]]
    axis = [str(step) for step in range(1, max(cell["read_step"] for cell in cells) + 1)]
    maximum = max(cell["nll"] for cell in cells)
    color = METHOD_COLORS[dynamic[0]["method"]]
    for summary in dynamic:
        chart = swanlab.echarts.HeatMap().add_xaxis(axis)
        chart.add_yaxis(
            "NLL",
            axis,
            [
                [cell["read_step"] - 1, cell["introduced_at"] - 1, cell["nll"]]
                for cell in summary["nll_retention"]["cells"]
            ],
            label_opts={"show": False},
        )
        chart.set_global_opts(
            legend_opts={"show": False},
            xaxis_opts={"name": "已读来源段", "type": "category"},
            yaxis_opts={"name": "问题引入段", "type": "category"},
            visualmap_opts={
                "min": 0,
                "max": maximum,
                "orient": "horizontal",
                "bottom": 0,
                "inRange": {"color": [_shade(color, 1, 2), color]},
            },
        )
        chart.options["title"] = {
            "text": _checkpoint_label(summary),
            "left": "center",
            "top": 0,
            "textStyle": {"fontSize": 12},
        }
        values[f"evaluation/nll_retention/step_{summary['metadata']['global_step']}"] = chart
    return values
