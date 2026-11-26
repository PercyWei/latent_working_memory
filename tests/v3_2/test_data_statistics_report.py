"""Parameter scans keep document weighting and independent recent denominators."""

import pytest

from latent_working_memory.v3_2.exploration.data_statistics.report import build_summary


def test_scan_groups_mask_only_the_infeasible_combination_and_do_not_repeat_recent():
    records = []
    for doc_id, cuts, tails in (("a", [True, False], [True, False]), ("b", [False], [True])):
        records.append(
            {
                "doc_id": doc_id,
                "dataset": "fixture",
                "split": "dev",
                "unit_kind": "paragraph",
                "status": "retained",
                "source_ref": {},
                "metrics": {"doc_tokens": 8},
                "qas": [
                    {
                        "qa_id": str(index),
                        "evidence_granularity": "span",
                        "status": "retained",
                        "metrics": {"evidence_tokens": 2},
                        "blocks": [
                            {"blocks": 2, "evidence_cut": cut, "evidence_block_count": 2},
                            {"blocks": 8, "evidence_cut": None, "evidence_block_count": None},
                        ],
                        "recent": [
                            {"recent_tokens": 0, "tail_overlap": False},
                            {"recent_tokens": 2, "tail_overlap": tail},
                        ],
                    }
                    for index, (cut, tail) in enumerate(zip(cuts, tails, strict=True))
                ],
                "segmentations": [
                    {
                        "blocks": 2,
                        "min_block_tokens": 1,
                        "max_block_tokens": None,
                        "uniform_feasible": True,
                        "natural_feasible": True,
                    },
                    {
                        "blocks": 2,
                        "min_block_tokens": 4,
                        "max_block_tokens": None,
                        "uniform_feasible": doc_id == "a",
                        "natural_feasible": False,
                    },
                    {
                        "blocks": 8,
                        "min_block_tokens": 1,
                        "max_block_tokens": 2,
                        "uniform_feasible": False,
                        "natural_feasible": False,
                    },
                ],
            }
        )
    records.append(
        {
            "doc_id": "excluded",
            "dataset": "fixture",
            "split": "dev",
            "unit_kind": "paragraph",
            "status": "too_short",
            "source_ref": {},
            "metrics": {},
            "qas": [],
        }
    )
    summary = build_summary(records)
    first, constrained, impossible = summary["segmentation_groups"]
    assert first["doc_count"] == 2
    assert first["evidence_cut"]["mean"] == pytest.approx(0.25)
    assert first["evidence_cut"]["qa_count"] == 3
    assert constrained["uniform_feasible_count"] == 1
    assert constrained["natural_feasible_count"] == 0
    cut = constrained["evidence_cut"]
    assert cut["mean"] == 0.5
    assert (cut["count"], cut["missing"], cut["qa_count"], cut["qa_missing"]) == (1, 1, 2, 1)
    assert impossible["evidence_cut"]["mean"] is None
    assert impossible["evidence_block_count"]["qa_missing"] == 3
    zero, recent = summary["recent_groups"]
    assert zero["tail_overlap"]["mean"] == 0
    assert recent["tail_overlap"]["mean"] == pytest.approx(0.75)
    assert recent["tail_overlap"]["qa_count"] == 3
    assert recent["doc_count"] == summary["qa_groups"][0]["document_count"] == 2
    assert summary["qa_ledger"]["scanned"] == 3
    assert "feasibility" not in summary
