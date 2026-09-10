"""将多组 MSC 评估结果汇总为一个 SwanLab 展示 run。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import swanlab

from cdic_repro.experiments.msc import MSC_SWANLAB_TAGS
from cdic_repro.experiments.msc.evaluate import SCOPES, write_json
from cdic_repro.experiments.tracking import SWANLAB_MODES, swanlab_run


SCOPE_LABELS = {
    "all_turns_s2_s5": "all turns",
    "session_final_s2_s5": "session final",
    "episode_final_s5": "session 5 final",
}
GENERATION_SCOPES = ("all_turns_s2_s5", "episode_final_s5")
EVALUATION_METRICS = (
    "ppl_including_eos",
    "bleu",
    "rouge_l_f1",
    "on_topic_rate",
    "mean_retrieved_states",
    "mean_memory_states_before",
)


@dataclass(frozen=True, slots=True)
class MscEvaluationSummaryConfig:
    artifact_dir: Path
    evaluations: tuple[tuple[str, Path], ...]
    comparisons: tuple[tuple[str, Path], ...]


def load_config(path: Path) -> MscEvaluationSummaryConfig:
    payload = _read_json_object(path)
    expected_fields = {"artifact_dir", "evaluations", "comparisons"}
    if set(payload) != expected_fields:
        raise ValueError(
            "MSC evaluation summary config fields must be "
            f"{sorted(expected_fields)}"
        )
    artifact_dir = payload["artifact_dir"]
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")
    return MscEvaluationSummaryConfig(
        artifact_dir=Path(artifact_dir),
        evaluations=_load_named_paths(payload["evaluations"], "evaluations", required=True),
        comparisons=_load_named_paths(payload["comparisons"], "comparisons", required=False),
    )


def build_report(config: MscEvaluationSummaryConfig) -> dict[str, Any]:
    evaluations = []
    for name, path in config.evaluations:
        summary = _read_json_object(path)
        _validate_evaluation_summary(summary, path)
        evaluations.append(
            {
                "name": name,
                "source": str(path),
                "condition": summary["condition"],
                "scopes": summary["scopes"],
            }
        )

    comparisons = []
    for name, path in config.comparisons:
        comparison = _read_json_object(path)
        _validate_comparison_report(comparison, path)
        comparisons.append(
            {
                "name": name,
                "source": str(path),
                "scopes": comparison["scopes"],
            }
        )
    return {"evaluations": evaluations, "comparisons": comparisons}


def run(
    config: MscEvaluationSummaryConfig,
    swanlab_mode: str = "disabled",
    swanlab_project: str = "latent-working-memory",
    swanlab_group: str | None = None,
    swanlab_tags: tuple[str, ...] = MSC_SWANLAB_TAGS,
    swanlab_run_id: str | None = None,
) -> None:
    report = build_report(config)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.artifact_dir / "report.json", report)
    tracking_config = {
        "evaluations": {name: str(path) for name, path in config.evaluations},
        "comparisons": {name: str(path) for name, path in config.comparisons},
    }
    with swanlab_run(
        config.artifact_dir,
        tracking_config,
        mode=swanlab_mode,
        project=swanlab_project,
        group=swanlab_group,
        tags=swanlab_tags,
        run_id=swanlab_run_id,
        job_type="evaluation-summary",
    ) as tracking:
        if tracking is not None:
            tracking.log(_tracking_media(report))


def _tracking_media(report: dict[str, Any]) -> dict[str, object]:
    evaluations = report["evaluations"]
    condition_names = [evaluation["name"] for evaluation in evaluations]
    media: dict[str, object] = {
        "evaluation_summary/ppl": _metric_bar(
            condition_names,
            evaluations,
            "ppl_including_eos",
            SCOPES,
        ),
        "evaluation_summary/bleu": _metric_bar(
            condition_names,
            evaluations,
            "bleu",
            GENERATION_SCOPES,
        ),
        "evaluation_summary/rouge_l_f1": _metric_bar(
            condition_names,
            evaluations,
            "rouge_l_f1",
            GENERATION_SCOPES,
        ),
        "retrieval_summary/on_topic_rate": _metric_bar(
            condition_names,
            evaluations,
            "on_topic_rate",
            ("all_turns_s2_s5",),
        ),
        "retrieval_summary/mean_retrieved_states": _metric_bar(
            condition_names,
            evaluations,
            "mean_retrieved_states",
            ("all_turns_s2_s5",),
        ),
        "retrieval_summary/mean_memory_states": _metric_bar(
            condition_names,
            evaluations,
            "mean_memory_states_before",
            ("all_turns_s2_s5",),
        ),
        "evaluation_summary/metrics_table": _evaluation_table(evaluations),
    }
    comparisons = report["comparisons"]
    if comparisons:
        comparison_names = [comparison["name"] for comparison in comparisons]
        media["comparison_summary/nll_delta"] = _comparison_bar(
            comparison_names,
            comparisons,
            "token_weighted_nll_delta",
        )
        media["comparison_summary/improved_fraction"] = _comparison_bar(
            comparison_names,
            comparisons,
            "improved_fraction",
        )
    return media


def _metric_bar(
    names: list[str],
    evaluations: list[dict[str, Any]],
    metric: str,
    scopes: tuple[str, ...],
) -> object:
    series = [
        (
            SCOPE_LABELS[scope],
            [float(evaluation["scopes"][scope][metric]) for evaluation in evaluations],
        )
        for scope in scopes
    ]
    return _horizontal_bar(names, series)


def _comparison_bar(
    names: list[str],
    comparisons: list[dict[str, Any]],
    metric: str,
) -> object:
    series = []
    for scope in SCOPES:
        values = []
        for comparison in comparisons:
            result = comparison["scopes"][scope]
            value = (
                int(result["paired_turns_improved"]) / int(result["paired_turns"])
                if metric == "improved_fraction"
                else float(result[metric])
            )
            values.append(value)
        series.append((SCOPE_LABELS[scope], values))
    return _horizontal_bar(names, series)


def _horizontal_bar(
    names: list[str],
    series: list[tuple[str, list[float]]],
) -> object:
    chart = swanlab.echarts.Bar()
    chart.add_xaxis(names)
    for series_name, values in series:
        chart.add_yaxis(series_name, values)
    chart.reversal_axis()
    return chart


def _evaluation_table(evaluations: list[dict[str, Any]]) -> object:
    headers = [
        "condition",
        "scope",
        "PPL (including EOS)",
        "BLEU",
        "ROUGE-L F1",
        "on-topic rate",
        "mean retrieved states",
        "mean memory states",
    ]
    rows = []
    for evaluation in evaluations:
        for scope in SCOPES:
            metrics = evaluation["scopes"][scope]
            rows.append(
                [
                    evaluation["name"],
                    SCOPE_LABELS[scope],
                    *_display_values(metrics, EVALUATION_METRICS),
                ]
            )
    table = swanlab.echarts.Table()
    table.add(headers, rows)
    return table


def _display_values(metrics: dict[str, Any], names: tuple[str, ...]) -> list[float]:
    return [round(float(metrics[name]), 6) for name in names]


def _load_named_paths(
    value: object,
    field_name: str,
    required: bool,
) -> tuple[tuple[str, Path], ...]:
    if not isinstance(value, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(path, str)
        or not path
        for name, path in value.items()
    ):
        raise ValueError(f"{field_name} must map non-empty names to non-empty paths")
    if required and not value:
        raise ValueError(f"{field_name} must not be empty")
    return tuple((name, Path(path)) for name, path in value.items())


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON value must be an object: {path}")
    return value


def _validate_evaluation_summary(summary: dict[str, Any], path: Path) -> None:
    if summary.get("condition") not in {"initialization", "final"}:
        raise ValueError(f"invalid evaluation condition: {path}")
    scopes = summary.get("scopes")
    if not isinstance(scopes, dict) or any(scope not in scopes for scope in SCOPES):
        raise ValueError(f"evaluation summary is missing required scopes: {path}")
    for scope in SCOPES:
        metrics = scopes[scope]
        if not isinstance(metrics, dict):
            raise ValueError(f"evaluation scope must be an object: {path}, {scope}")
        for metric in EVALUATION_METRICS:
            _require_finite_number(metrics, metric, path)


def _validate_comparison_report(report: dict[str, Any], path: Path) -> None:
    scopes = report.get("scopes")
    if not isinstance(scopes, dict) or any(scope not in scopes for scope in SCOPES):
        raise ValueError(f"comparison report is missing required scopes: {path}")
    for scope in SCOPES:
        result = scopes[scope]
        if not isinstance(result, dict):
            raise ValueError(f"comparison scope must be an object: {path}, {scope}")
        _require_finite_number(result, "token_weighted_nll_delta", path)
        improved = result.get("paired_turns_improved")
        turns = result.get("paired_turns")
        if (
            not isinstance(improved, int)
            or isinstance(improved, bool)
            or not isinstance(turns, int)
            or isinstance(turns, bool)
            or not 0 <= improved <= turns
            or turns == 0
        ):
            raise ValueError(f"invalid paired turn counts: {path}, {scope}")


def _require_finite_number(mapping: dict[str, Any], key: str, path: Path) -> None:
    value = mapping.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{key} must be a finite number: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--swanlab-mode", choices=SWANLAB_MODES, default="disabled")
    parser.add_argument("--swanlab-project", default="latent-working-memory")
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-tag", action="append", default=[])
    parser.add_argument("--swanlab-run-id")
    arguments = parser.parse_args()
    run(
        load_config(arguments.config),
        swanlab_mode=arguments.swanlab_mode,
        swanlab_project=arguments.swanlab_project,
        swanlab_group=arguments.swanlab_group,
        swanlab_tags=MSC_SWANLAB_TAGS + tuple(arguments.swanlab_tag),
        swanlab_run_id=arguments.swanlab_run_id,
    )


if __name__ == "__main__":
    main()
