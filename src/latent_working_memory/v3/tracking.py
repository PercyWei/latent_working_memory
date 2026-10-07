"""v3 方法级 SwanLab 身份、跨阶段增量曲线与最终评估汇总。"""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import swanlab

from latent_working_memory.v1.reporting import _shade, _table
from latent_working_memory.v3.config import DYNAMIC_METHODS
from latent_working_memory.v4.checkpoint import capture_rng, restore_rng


TRAINING_METRICS = (
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


def experiment_directory(training):
    """方法的持久化根目录；独立 train 命令默认使用阶段输出目录。"""
    return Path(training.experiment_dir or training.output_dir).resolve()


def tracking_method(config):
    if config.objective.method in DYNAMIC_METHODS and config.objective.stage == "pretrain":
        return "shared-pretrain"
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
        "method": tracking_method(config),
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
    method = tracking_method(config)
    data = (
        ("fineweb",)
        if method in {"shared-pretrain", "autocompressors"}
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
            if any(remote_config.get(name) != value for name, value in previous.items()):
                raise ValueError("SwanLab configuration differs from the saved method experiment")
            # 保留云端已有附加配置；只合入经过本地一致性校验的阶段记录。
            combined = {**remote_config, **combined}
        tracking = swanlab.init(
            project=config.training.swanlab_project,
            workspace=workspace,
            name=root.name,
            config=combined,
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
        yield tracking


def update_method_tracking(config, run, tracking):
    """验证阶段衔接并更新同一活跃 run，不结束会话或重新访问云端身份。"""
    record_path = experiment_directory(config.training) / "experiment.json"
    previous = json.loads(record_path.read_text(encoding="utf-8"))
    combined = _experiment_config(config, run, previous)
    if tracking is not None:
        tracking.config.update(combined)
    _save_record(record_path, combined)


def _loss_name(stage):
    return "ae_lm_loss" if stage == "pretrain" else "lm_loss" if stage == "lm" else "qa_loss"


def configure_training_metrics(tracking, stage):
    """目标不同的损失分开展示，所有曲线沿用累计真实 optimizer step。"""
    if tracking is None:
        return
    names = (*TRAINING_METRICS, f"train/{_loss_name(stage)}", f"dev/{_loss_name(stage)}")
    for name in names:
        tracking.define_metric(name, x_axis="_step", section_name=name.split("/", 1)[0])


def training_metrics(record):
    """只上传当前实际计算的指标，warmup 与 policy 共享 QA 目标曲线。"""
    values = {name: record[name] for name in TRAINING_METRICS if record.get(name) is not None}
    for section in ("train", "dev"):
        if record.get(f"{section}/loss") is not None:
            values[f"{section}/{_loss_name(record['stage'])}"] = record[f"{section}/loss"]
    if "resources/peak_memory_allocated_bytes" in record:
        values["resources/peak_memory_allocated_gib"] = (
            record["resources/peak_memory_allocated_bytes"] / 1024**3
        )
    return values


def stage_progress(config):
    """从各阶段实际结果构建边界表，不把共享预训练步数计入下游 run。"""
    root = experiment_directory(config.training)
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    rows = []
    for stage, record in manifest["stages"].items():
        path = Path(record["config"]["training"]["output_dir"]) / "training-result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "stage": stage,
                "objective": _loss_name(stage),
                "start_global_step": record["step_offset"] + 1
                if result["completed_steps"]
                else None,
                "end_global_step": result["global_step"],
                "stage_steps": result["completed_steps"],
                "complete": result["complete"],
            }
        )
    return _table(rows)


def evaluation_media(summary, rows):
    """每项核心质量指标合并新旧问题；辅助统计与少量生成样例集中入表。"""
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
    values["evaluation/summary"] = _table(
        [
            {
                "method": method,
                "split": summary["split"],
                "offline_oracle": summary["offline_oracle"],
                "trajectories": summary["trajectories"],
                "group": group,
                **summary["quality"][group],
                **summary["capacity"],
            }
            for group in groups
        ]
    )
    details = [
        {"category": "age", "name": age, **metrics}
        for age, metrics in summary["quality"]["by_age"].items()
    ]
    details.extend(
        {"category": "cost", "name": name, "value": value}
        for name, value in summary["costs"].items()
    )
    details.extend(
        {"category": "selection", "name": name, "value": value}
        for name, value in summary["metadata"].get("selection", {}).items()
    )
    values["evaluation/details"] = _table(details)
    examples = [
        {
            "trajectory_id": row["trajectory_id"],
            "qa_id": question["qa_id"],
            "age": question["age"],
            "question": question["question"],
            "answer": question["answer"],
            "prediction": question["prediction"],
            **{name: question[name] for name in ("nll", "em", "f1")},
        }
        for row in rows
        for question in row["questions"]
    ][:8]
    if examples:
        values["evaluation/examples"] = _table(examples)
    return values
