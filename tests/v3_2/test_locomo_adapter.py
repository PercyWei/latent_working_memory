"""LoCoMo evidence must refer to the unchanged public benchmark history."""

import json

from benchmarks.data import _locomo_history
from latent_working_memory.v3_2.exploration.data_statistics.adapters.locomo import iter_locomo


def test_locomo_preserves_history_and_resolves_complete_evidence(tmp_path):
    conversation = {
        "speaker_a": "Alex",
        "speaker_b": "Blair",
        "session_10_date_time": "10 May 2023",
        "session_10": [{"speaker": "Alex", "dia_id": "D10:1", "text": "Same words."}],
        "session_2_date_time": "2 May 2023",
        "session_2": [
            {"speaker": "Alex", "dia_id": "D2:1", "text": "Same words."},
            {
                "speaker": "Blair",
                "dia_id": "D2:2",
                "text": "Here it is.",
                "blip_caption": "a blue bicycle",
            },
        ],
    }
    row = {
        "sample_id": "example",
        "conversation": conversation,
        "qa": [{"category": 1, "evidence": ["D2:2; D10:1 D2:1"]}],
    }
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    document = next(iter_locomo(tmp_path))
    assert document.split is None
    assert document.source_ref["path"] == str(path)
    assert document.text == _locomo_history(conversation, "fixture")[0]
    assert [document.text[start:end] for start, end in document.unit_spans] == [
        'Alex said, "Same words."',
        'Blair said, "Here it is." and shared a blue bicycle.',
        'Alex said, "Same words."',
    ]
    assert document.qas[0].evidence_spans == (
        document.unit_spans[1],
        document.unit_spans[2],
        document.unit_spans[0],
    )
    assert document.qas[0].exclusion_reason is None
    assert document.source_ref["row_index"] == 0
    assert document.source_ref["qa_selection_rule"] == "category:1-4"


def test_locomo_preserves_missing_and_unlocatable_qa_in_selection_ledger(tmp_path):
    row = {
        "sample_id": "example",
        "conversation": {
            "speaker_a": "Alex",
            "speaker_b": "Blair",
            "session_1_date_time": "1 May 2023",
            "session_1": [{"speaker": "Alex", "dia_id": "D1:1", "text": "Hello."}],
        },
        "qa": [
            {"category": 1, "evidence": []},
            {"category": 2, "evidence": ["D1:1", "D1:2"]},
            {"category": 3, "evidence": ["D1:01"]},
            {"category": 4, "evidence": ["D:1:1"]},
            {"category": 5, "evidence": ["D1:1"]},
        ],
    }
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    document = next(iter_locomo(tmp_path))
    assert document.split is None
    assert [qa.exclusion_reason for qa in document.qas] == [
        "missing_evidence_annotation",
        "evidence_locator_failed",
        "evidence_locator_failed",
        "evidence_locator_failed",
    ]
    assert all(qa.evidence_spans == () for qa in document.qas)
    assert document.source_ref["source_qa_count"] == 5
    assert document.source_ref["selected_qa_count"] == 4
