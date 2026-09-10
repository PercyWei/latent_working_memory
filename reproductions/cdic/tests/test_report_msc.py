from __future__ import annotations

import json

import pytest

from cdic_repro.experiments.msc.evaluate import SCOPES
from cdic_repro.experiments.msc.report import (
    _tracking_media,
    build_report,
    load_config,
)


def _metrics(value: float) -> dict[str, float]:
    return {
        "ppl_including_eos": value + 10,
        "bleu": value + 0.1,
        "rouge_l_f1": value + 0.2,
        "on_topic_rate": value + 0.3,
        "mean_retrieved_states": value + 1,
        "mean_memory_states_before": value + 2,
    }


def test_summary_report_builds_custom_bar_charts_and_exact_table(tmp_path):
    summary_paths = []
    for index, condition in enumerate(("initialization", "final")):
        path = tmp_path / f"{condition}.json"
        path.write_text(
            json.dumps(
                {
                    "condition": condition,
                    "scopes": {
                        scope: _metrics(index + scope_index / 10)
                        for scope_index, scope in enumerate(SCOPES)
                    },
                }
            ),
            encoding="utf-8",
        )
        summary_paths.append(path)

    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "scopes": {
                    scope: {
                        "token_weighted_nll_delta": -0.1,
                        "paired_turns_improved": 3,
                        "paired_turns": 4,
                    }
                    for scope in SCOPES
                }
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "report_config.json"
    config_path.write_text(
        json.dumps(
            {
                "artifact_dir": str(tmp_path / "summary-run"),
                "evaluations": {
                    "initialization-th0.80": str(summary_paths[0]),
                    "final-lr2e-4-th0.80": str(summary_paths[1]),
                },
                "comparisons": {"lr2e-4-th0.80": str(comparison_path)},
            }
        ),
        encoding="utf-8",
    )

    report = build_report(load_config(config_path))
    media = _tracking_media(report)

    assert len(media) == 9
    assert all(not isinstance(value, (int, float)) for value in media.values())
    ppl = json.loads(media["evaluation_summary/ppl"].dump_options())
    assert ppl["yAxis"][0]["data"] == [
        "initialization-th0.80",
        "final-lr2e-4-th0.80",
    ]
    assert [series["name"] for series in ppl["series"]] == [
        "all turns",
        "session final",
        "session 5 final",
    ]
    improved = json.loads(media["comparison_summary/improved_fraction"].dump_options())
    assert improved["series"][0]["data"] == [0.75]
    table = json.loads(media["evaluation_summary/metrics_table"].dump_options())
    assert len(table["rowData"]) == 6


def test_summary_report_rejects_missing_evaluation_metric(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "condition": "initialization",
                "scopes": {scope: _metrics(0.0) for scope in SCOPES},
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    del payload["scopes"][SCOPES[0]]["bleu"]
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    config_path = tmp_path / "report_config.json"
    config_path.write_text(
        json.dumps(
            {
                "artifact_dir": str(tmp_path / "summary-run"),
                "evaluations": {"initialization": str(summary_path)},
                "comparisons": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bleu must be a finite number"):
        build_report(load_config(config_path))
