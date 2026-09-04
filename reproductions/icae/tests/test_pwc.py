from __future__ import annotations

import json
from pathlib import Path

import pytest

from icae_repro.pwc import (
    BatchConfig,
    Condition,
    iter_records,
    load_completed_ids,
    parse_record,
)


def make_config(tmp_path: Path, **overrides: object) -> BatchConfig:
    values: dict[str, object] = {
        "condition": Condition.FULL_CONTEXT,
        "model_path": Path("model"),
        "input_path": tmp_path / "input.jsonl",
        "output_path": tmp_path / "output.jsonl",
    }
    values.update(overrides)
    return BatchConfig(**values)  # type: ignore[arg-type]


def test_icae_condition_requires_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        make_config(tmp_path, condition=Condition.ICAE_128)


def test_parse_record_uses_stable_index_id() -> None:
    record = parse_record(
        {"input": "context", "prompt": "question", "answer": "answer"},
        sample_index=17,
    )

    assert record.sample_id == "pwc-000017"
    assert record.context == "context"


def test_parse_record_rejects_missing_text_fields() -> None:
    with pytest.raises(ValueError, match="answer"):
        parse_record({"input": "context", "prompt": "question"}, sample_index=0)


def test_iter_records_applies_shard_resume_and_limit(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        shard_index=1,
        num_shards=2,
        max_samples=1,
    )
    rows = [
        {"id": f"sample-{index}", "input": "context", "prompt": "prompt", "answer": "a"}
        for index in range(6)
    ]
    config.input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    records = list(iter_records(config, {"sample-1"}))

    assert [record.sample_id for record in records] == ["sample-3"]
    assert records[0].sample_index == 3


def test_load_completed_ids_rejects_duplicates(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text('{"id":"same"}\n{"id":"same"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_completed_ids(output)
