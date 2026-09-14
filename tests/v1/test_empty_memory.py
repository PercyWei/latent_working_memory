from copy import deepcopy
import json

import pytest
import torch

from latent_working_memory.v1.dynamic.empty_memory import (
    EMPTY_CONDITIONS,
    ORIGINAL_CONDITIONS,
    evaluate_empty_memory,
    original_requests,
)
from latent_working_memory.v1.dynamic.evaluation import aggregate_qa
from latent_working_memory.v1.dynamic.reporting import qa_media


def source_rows():
    return [
        dict(
            capacity=k,
            episode_id="doc",
            read_id="q",
            prefix_end=12,
            document_id="source",
            condition=c,
            question="What?",
            references=["First"],
            target_tokens=2,
            nll_sum=4.0,
            em=0.0,
            f1=0.0,
            prediction="Other",
            hit_limit=False,
            kind="delayed",
            delay_tokens=6,
        )
        for k in (4, 8)
        for c in sorted(ORIGINAL_CONDITIONS)
    ]


def test_empty_baselines_disable_adapter_and_never_write_memory(components, tokenizer, monkeypatch):
    backbone, _ = components
    torch.manual_seed(8)
    with torch.no_grad():
        for name, parameter in backbone.language_model.named_parameters():
            if "lora_" in name:
                parameter.normal_(0, 0.15)
    calls = []
    original = backbone.read_batch

    def read(memories, tokens, use_reader_lora):
        assert all(len(m) == 0 for m in memories)
        calls.append(use_reader_lora)
        return original(memories, tokens, use_reader_lora=use_reader_lora)

    def forbidden(*args, **kwargs):
        pytest.fail("empty-memory baseline must not read or encode source text")

    monkeypatch.setattr(backbone, "read_batch", read)
    monkeypatch.setattr(backbone, "text_features", forbidden)
    source = source_rows()
    saved = deepcopy(source)
    added = evaluate_empty_memory(backbone, tokenizer, source, 2, 256, torch.device("cpu"))
    assert source == saved
    assert calls == [False, True]  # Cached across capacities and read positions.
    assert len(added) == 4 and {r["condition"] for r in added} == set(EMPTY_CONDITIONS)
    assert added[0]["nll_sum"] != pytest.approx(added[1]["nll_sum"])
    assert added[0]["nll_sum"] == added[2]["nll_sum"]


def test_original_conditions_must_be_unique_complete_and_paired():
    rows = source_rows()
    with pytest.raises(ValueError, match="original five"):
        original_requests(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        original_requests(rows + [rows[0]])
    rows[0]["references"] = ["Different"]
    with pytest.raises(ValueError, match="references"):
        original_requests(rows)


def test_final_charts_group_seven_conditions_by_dataset():
    rows = source_rows()
    for condition in EMPTY_CONDITIONS:
        rows += [
            dict(r, condition=condition) for r in source_rows() if r["condition"] == "no_memory"
        ]
    metrics = aggregate_qa(rows)
    assert "paired/no_memory-minus-no_memory_pretrain" in metrics
    media = qa_media({"personamem": (metrics, rows), "squad": (metrics, rows)})
    chart = json.loads(media["evaluation/test/overview/f1"].dump_options())
    assert chart["xAxis"][0]["data"] == ["personamem", "squad"]
    assert len(chart["series"]) == 7
    assert all(len(s["data"]) == 2 for s in chart["series"])
