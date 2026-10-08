"""多段构造与 FactQA 定稿必须保留同一份正文、分段及来源契约。"""

from dataclasses import asdict
import re

from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig
from latent_working_memory.data_preparation.fineweb_multisegment.prepare import (
    Document,
    build_samples,
)
from latent_working_memory.data_preparation.fineweb_factqa.assembly import (
    assemble_document,
    qa_quotas,
)


def test_multisegment_text_layout_is_preserved_by_factqa_assembly():
    source_text = "".join(f"Fact {index:06d} has value-{index:06d}.\n" for index in range(2000))
    document = Document(
        document_id="contract-document",
        text=source_text,
        split="train",
        dedup_cluster="example.org/contract-document",
        source={"file": "synthetic.parquet", "row_group": 2, "row_index": 7},
    )
    config = DataPreparationConfig(source_dir="unused")
    (sample,) = build_samples([document], config, "train")
    sample.validate_plan(config.window)
    multisegment = asdict(sample)
    shared_fields = {
        "trajectory_id",
        "document_id",
        "dedup_cluster",
        "split",
        "source",
        "window_char_span",
        "text",
        "segments",
        "text_char_length",
        "estimated_tokens",
        "estimated_tokens_rule",
    }
    assert set(multisegment) == shared_fields | {"continuation"}
    common_document = {key: multisegment[key] for key in shared_fields}

    candidates, decisions = [], []
    candidate_counts, task_counts, gate_counts = qa_quotas(len(sample.segments))
    sentence = re.compile(r"Fact (?P<fact>\d{6}) has (?P<answer>value-\d{6})\.\n")
    for segment, required in zip(sample.segments, candidate_counts, strict=True):
        start, end = segment["char_span"]
        matches = list(sentence.finditer(sample.text[start:end]))
        assert len(matches) >= max(12, required)
        for match in matches[:required]:
            fact_id = match.group("fact")
            qa_id = f"{segment['segment_id']}-fact-{fact_id}"
            candidates.append(
                {
                    "qa_id": qa_id,
                    "segment_id": segment["segment_id"],
                    "fact_statement": match.group().strip(),
                    "question": f"What value does fact {fact_id} have?",
                    "answer": match.group("answer"),
                    "evidence_char_span": [start + match.start(), start + match.end()],
                    "answer_char_span": [
                        start + match.start("answer"),
                        start + match.end("answer"),
                    ],
                }
            )
            decisions.append(
                {
                    "qa_id": qa_id,
                    "accepted": True,
                    "reason": "",
                    "fact_group_id": f"fact-{fact_id}",
                }
            )

    result = assemble_document(
        common_document, candidates, decisions, {"role_seed": 20260928, "max_answer_chars": 128}
    )
    assert result["ok"] and result["shortfalls"] == []
    factqa = result["trajectory"]
    assert set(factqa) == shared_fields | {"qas", "usage"}
    assert {key: factqa[key] for key in shared_fields} == common_document
    assert len(factqa["qas"]) == sum(task_counts) + sum(gate_counts)
    assert len(factqa["usage"]) == len(sample.segments)

    start, end = sample.window_char_span
    candidate_end = end + config.window.continuation_chars
    assert sample.text == source_text[start:end]
    assert sample.continuation == source_text[end:candidate_end]
    assert sample.text + sample.continuation == source_text[start:candidate_end]
