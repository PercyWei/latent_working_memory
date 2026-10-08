import copy
import json
import string

import pytest
from tokenizers import Tokenizer, models, processors
from transformers import PreTrainedTokenizerFast

from latent_working_memory.data_preparation.fineweb_factqa.assembly import (
    assemble_document,
    qa_quotas,
)
from latent_working_memory.v3.data import load_factqa, tokenize_trajectory


ROLE_SEED = 17
QA_CONFIG = {"role_seed": ROLE_SEED, "max_answer_chars": 128}
SOURCE_SEED = 37


@pytest.fixture
def tokenizer():
    vocabulary = {token: index for index, token in enumerate(("<unk>", "<bos>", "<eos>", "<pad>"))}
    for character in string.printable + "标题🌍":
        if character not in vocabulary:
            vocabulary[character] = len(vocabulary)
    vocabulary["ab"] = len(vocabulary)
    backend = Tokenizer(models.BPE(vocabulary, [("a", "b")], unk_token="<unk>"))
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", vocabulary["<bos>"]), ("<eos>", vocabulary["<eos>"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=32,
    )


def make_record(
    segment_count=8,
    split="train",
    index=0,
    qa_namespace=None,
    segment_chars=None,
    leading_padding=0,
    answer_suffix="",
    qa_config=None,
    document_id=None,
):
    document_id = document_id or f"document-{split}-{index}"
    parts, segments, candidates = [], [], []
    offset = 0
    _, tasks, gates = qa_quotas(segment_count)
    for segment_index in range(segment_count):
        segment_id = f"seg{segment_index}"
        lines = [f"b\n标题 🌍 {document_id} {segment_id}\n", " " * leading_padding]
        for item in range(tasks[segment_index] + gates[segment_index]):
            name = f"{index}_{segment_index}_{item}"
            answer = f"value_{name}{answer_suffix}"
            evidence = f"Fact {name} has {answer}.\n"
            start = offset + sum(map(len, lines))
            answer_start = start + evidence.index(answer)
            candidates.append(
                {
                    "qa_id": f"{qa_namespace or document_id}:{segment_id}:qa{item}",
                    "segment_id": segment_id,
                    "fact_statement": f"Fact {name} has {answer}",
                    "question": f"What value belongs to fact {name}?",
                    "answer": answer,
                    "evidence_char_span": [start, start + len(evidence)],
                    "answer_char_span": [answer_start, answer_start + len(answer)],
                }
            )
            lines.append(evidence)
        # Consecutive segments meet at 'ab', which merges only in full-prefix encoding.
        if segment_chars is not None:
            lines.append(" " * (segment_chars - sum(map(len, lines)) - 1))
        lines.append("a")
        text = "".join(lines)
        parts.append(text)
        segments.append({"segment_id": segment_id, "char_span": [offset, offset + len(text)]})
        offset += len(text)
    document = {
        "trajectory_id": f"trajectory-{split}-{index}",
        "document_id": document_id,
        "dedup_cluster": f"cluster-{split}-{index}",
        "split": split,
        "source": {"file": f"sample/{split}.parquet", "row_group": 0, "row_index": index},
        "window_char_span": [100, 100 + offset],
        "text": "".join(parts),
        "segments": segments,
    }
    decisions = [
        {"qa_id": q["qa_id"], "accepted": True, "reason": "", "fact_group_id": q["qa_id"]}
        for q in candidates
    ]
    return assemble_document(document, candidates, decisions, qa_config or QA_CONFIG)["trajectory"]


def write_dataset(directory, records, qa_config=None):
    preparation = {
        "qa": dict(qa_config or QA_CONFIG),
        "split_counts": {
            split: sum(record["split"] == split for record in records)
            for split in ("train", "dev", "test")
        },
        "source_pool_config": {
            "source_seed": SOURCE_SEED,
            "selection_seed": 17,
            "split_counts": {"train": 1000, "dev": 100, "test": 100},
            "window": {
                "capacity": 512,
                "min_segment_ratio": 1,
                "max_segment_ratio": 3,
                "min_segments": 6,
                "max_segments": 10,
                "content_reserve_ratio": 1.5,
            },
        },
    }
    (directory / "preparation.json").write_text(json.dumps(preparation), encoding="utf-8")
    for split in ("train", "dev", "test"):
        (directory / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in records if r["split"] == split
            ),
            encoding="utf-8",
        )
    return preparation


@pytest.mark.parametrize("segment_count", range(6, 11))
@pytest.mark.parametrize("split", ("train", "dev", "test"))
def test_tokenization_preserves_variable_segments_roles_and_actual_usage(
    tokenizer, segment_count, split
):
    record = make_record(segment_count, split)
    trajectory = tokenize_trajectory(record, tokenizer, QA_CONFIG, split)
    assert trajectory.text == record["text"]
    assert trajectory.source == record["source"]
    assert trajectory.window_char_span == tuple(record["window_char_span"])
    assert len(trajectory.segments) == len(trajectory.usage) == segment_count
    assert len(trajectory.qas) == 8 * segment_count
    task_role = "train" if split == "train" else "evaluation"
    assert sum(q.role == task_role for q in trajectory.qas.values()) == 4 * segment_count
    assert sum(q.role == "gate" for q in trajectory.qas.values()) == 4 * segment_count
    for index, (segment, step) in enumerate(
        zip(trajectory.segments, trajectory.usage, strict=True)
    ):
        source = record["text"][slice(*segment.char_span)]
        assert segment.input_ids == tuple(tokenizer.encode(source, add_special_tokens=False))
        assert step.new_qa_ids == tuple(record["usage"][index]["task_new_qa_ids"])
        assert step.old_qa_ids == tuple(record["usage"][index]["task_old_qa_ids"])
        assert step.gate_qa_ids == tuple(record["usage"][index]["gate_qa_ids"])
        assert len(step.old_qa_ids) == ((4 if index else 0) if split == "train" else 4 * index)
        assert len(step.gate_qa_ids) == (8 if index else 0)
        for qa_id in step.new_qa_ids + step.old_qa_ids:
            assert trajectory.qas[qa_id].role == task_role
        for qa_id in step.gate_qa_ids:
            assert trajectory.qas[qa_id].role == "gate"
        for qa_id in step.old_qa_ids + step.gate_qa_ids:
            assert trajectory.qas[qa_id].evidence_char_span[1] <= segment.char_span[0]
    for raw in record["qas"]:
        qa = trajectory.qas[raw["qa_id"]]
        assert (qa.question, qa.answer) == (raw["question"], raw["answer"])
        assert qa.question_ids == tuple(tokenizer.encode(qa.question, add_special_tokens=False))
        assert qa.answer_ids == tuple(tokenizer.encode(qa.answer, add_special_tokens=False))
        assert trajectory.text[slice(*qa.answer_char_span)] == qa.answer


def test_prefix_encoding_uses_original_text_without_special_tokens_or_truncation(tokenizer):
    record = make_record(6)
    trajectory = tokenize_trajectory(record, tokenizer, QA_CONFIG, "train")
    flattened = sum((segment.input_ids for segment in trajectory.segments), ())
    assert trajectory.full_input_ids != flattened
    assert len(trajectory.full_input_ids) > tokenizer.model_max_length
    assert not set(tokenizer.all_special_ids).intersection(trajectory.full_input_ids)
    for step in range(6):
        end = trajectory.segments[step].char_span[1]
        assert trajectory.prefix_ids(step, tokenizer) == tuple(
            tokenizer.encode(record["text"][:end], add_special_tokens=False)
        )
    assert trajectory.prefix_ids(1, tokenizer) != sum(
        (s.input_ids for s in trajectory.segments[:2]), ()
    )
    assert trajectory.prefix_ids(5, tokenizer) is trajectory.full_input_ids
    for step in (-1, 6, True):
        with pytest.raises(IndexError, match="zero-based"):
            trajectory.prefix_ids(step, tokenizer)


def test_load_preserves_all_splits_and_usage(tmp_path, tokenizer):
    records = [make_record(6, "train"), make_record(9, "dev"), make_record(10, "test")]
    write_dataset(tmp_path, records)
    dataset = load_factqa(tmp_path, tokenizer)
    assert set(dataset) == {"train", "dev", "test"}
    assert all(isinstance(rows, tuple) for rows in dataset.values())
    assert [len(dataset[s][0].segments) for s in dataset] == [6, 9, 10]
    for split in ("dev", "test"):
        trajectory = dataset[split][0]
        final = trajectory.usage[-1]
        assert len(final.new_qa_ids + final.old_qa_ids) == 4 * len(trajectory.segments)


@pytest.mark.parametrize("answer_suffix,limit", [("", 32), ("x" * 140, 192)])
def test_loader_uses_metadata_answer_limit(tmp_path, tokenizer, answer_suffix, limit):
    config = dict(QA_CONFIG, max_answer_chars=limit)
    record = make_record(6, answer_suffix=answer_suffix, qa_config=config)
    metadata = write_dataset(tmp_path, [record], qa_config=config)
    trajectory = load_factqa(tmp_path, tokenizer)["train"][0]
    assert set(trajectory.qas) == {qa["qa_id"] for qa in record["qas"]}
    maximum = max(len(qa.answer) for qa in trajectory.qas.values())
    if answer_suffix:
        assert maximum > 128
    metadata["qa"]["max_answer_chars"] = maximum - 1
    (tmp_path / "preparation.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=f"answer exceeds {maximum - 1} characters"):
        load_factqa(tmp_path, tokenizer)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "128"])
def test_loader_rejects_invalid_metadata_answer_limit(tmp_path, tokenizer, limit):
    write_dataset(tmp_path, [make_record(6)], qa_config=dict(QA_CONFIG, max_answer_chars=limit))
    with pytest.raises(ValueError, match="qa.max_answer_chars must be a positive integer"):
        load_factqa(tmp_path, tokenizer)


@pytest.mark.parametrize("segment_chars", (3073, 3392, 3393, 4607, 4608, 4610, 6144, 6145))
def test_loader_uses_saved_character_spans_without_resegmenting(tmp_path, tokenizer, segment_chars):
    record = make_record(6, segment_chars=segment_chars)
    write_dataset(tmp_path, [record])
    trajectory = load_factqa(tmp_path, tokenizer)["train"][0]
    for segment, raw in zip(trajectory.segments, record["segments"], strict=True):
        assert segment.char_span == tuple(raw["char_span"])
        assert segment.input_ids == tuple(
            tokenizer.encode(record["text"][slice(*segment.char_span)], add_special_tokens=False)
        )


def test_loader_preserves_evidence_in_expanded_part_of_each_segment(tmp_path, tokenizer):
    nominal_tokens = 769
    record = make_record(6, segment_chars=4614, leading_padding=3300)
    write_dataset(tmp_path, [record])
    trajectory = load_factqa(tmp_path, tokenizer)["train"][0]
    assert trajectory.text == record["text"]
    assert record["estimated_tokens"] == 6 * 4614 / 4
    for qa in trajectory.qas.values():
        segment = trajectory.segments[int(qa.segment_id[3:])]
        assert qa.evidence_char_span[0] - segment.char_span[0] > 4 * nominal_tokens
        assert trajectory.text[slice(*qa.answer_char_span)] == qa.answer
        assert segment.input_ids == tuple(
            tokenizer.encode(trajectory.text[slice(*segment.char_span)], add_special_tokens=False)
        )


@pytest.mark.parametrize("segment_count", (2, 11))
def test_loader_uses_saved_segment_count(tmp_path, tokenizer, segment_count):
    record = make_record(segment_count)
    write_dataset(tmp_path, [record])
    trajectory = load_factqa(tmp_path, tokenizer)["train"][0]
    assert len(trajectory.segments) == segment_count


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("future_task", "usage schedule"),
        ("gate_as_task", "usage schedule"),
        ("duplicate_usage", "usage schedule"),
        ("duplicate_fact", "fact group appears"),
        ("future_evidence", "evidence is outside"),
        ("answer_span", "answer span does not match"),
        ("segment_gap", "consecutively cover"),
        ("quote_field", "QA fields are not canonical"),
        ("legacy_usage", "usage schedule"),
        ("extra_trajectory", "requires exactly"),
    ],
)
def test_invalid_evidence_fact_pools_and_schedules_are_rejected(tokenizer, change, error):
    record = make_record(6)
    if change == "future_task":
        record["usage"][1]["task_old_qa_ids"][0] = record["usage"][2]["task_new_qa_ids"][0]
    elif change == "gate_as_task":
        record["usage"][1]["task_old_qa_ids"][0] = record["usage"][1]["gate_qa_ids"][0]
    elif change == "duplicate_usage":
        record["usage"][1]["gate_qa_ids"][1] = record["usage"][1]["gate_qa_ids"][0]
    elif change == "duplicate_fact":
        record["qas"][4]["fact_group_id"] = record["qas"][0]["fact_group_id"]
    elif change == "future_evidence":
        record["qas"][0]["evidence_char_span"][1] = record["segments"][1]["char_span"][1]
    elif change == "answer_span":
        record["qas"][0]["answer_char_span"][0] -= 1
    elif change == "segment_gap":
        record["segments"][1]["char_span"][0] += 1
    elif change == "quote_field":
        record["qas"][0]["evidence_quote"] = "old schema"
    elif change == "legacy_usage":
        step = record["usage"][0]
        step["train_new_qa_ids"] = step.pop("task_new_qa_ids")
    else:
        record["unexpected"] = True
    with pytest.raises(ValueError, match=error):
        tokenize_trajectory(record, tokenizer, QA_CONFIG, "train")


@pytest.mark.parametrize("mismatch", ("file_split", "cluster_split"))
def test_loader_rejects_file_and_cluster_split_leakage(tmp_path, tokenizer, mismatch):
    train = make_record(6)
    dev = make_record(6, "dev")
    if mismatch == "cluster_split":
        dev["dedup_cluster"] = train["dedup_cluster"]
    write_dataset(tmp_path, [train, dev])
    if mismatch == "file_split":
        (tmp_path / "train.jsonl").write_text(json.dumps(dev) + "\n")
    message = {
        "file_split": "requested dataset split",
        "cluster_split": "multiple dataset splits",
    }[mismatch]
    with pytest.raises(ValueError, match=message):
        load_factqa(tmp_path, tokenizer)


@pytest.mark.parametrize("duplicate", ("document", "trajectory", "source_row", "qa_id"))
def test_loader_rejects_duplicate_samples_and_ids(tmp_path, tokenizer, duplicate):
    first, second = make_record(6), make_record(6, index=1)
    if duplicate == "document":
        second = copy.deepcopy(first)
        second["trajectory_id"] = "another-trajectory"
        second["source"]["row_index"] = 1
    elif duplicate == "trajectory":
        second["trajectory_id"] = first["trajectory_id"]
    elif duplicate == "source_row":
        second["source"] = dict(first["source"])
    else:
        first = make_record(6, qa_namespace="shared")
        second = make_record(6, index=1, qa_namespace="shared")
    write_dataset(tmp_path, [first, second])
    message = {
        "document": "source document changes",
        "trajectory": "duplicate trajectory",
        "source_row": "source row belongs",
        "qa_id": "duplicate QA",
    }[duplicate]
    with pytest.raises(ValueError, match=message):
        load_factqa(tmp_path, tokenizer)


@pytest.mark.parametrize("overlap", [False, True])
def test_loader_accepts_same_source_disjoint_trajectories_and_rejects_overlap(
    tmp_path, tokenizer, overlap
):
    first = make_record(6)
    second = make_record(6, index=1, document_id=first["document_id"], qa_namespace="second")
    second["dedup_cluster"] = first["dedup_cluster"]
    second["source"] = dict(first["source"])
    start = first["window_char_span"][1] - int(overlap)
    second["window_char_span"] = [start, start + len(second["text"])]
    # Read the later window first to exercise order-independent interval checks.
    write_dataset(tmp_path, [second, first])
    if overlap:
        with pytest.raises(ValueError, match="overlapping source windows"):
            load_factqa(tmp_path, tokenizer)
    else:
        rows = load_factqa(tmp_path, tokenizer)["train"]
        assert [row.trajectory_id for row in rows] == [
            second["trajectory_id"],
            first["trajectory_id"],
        ]
        assert len({row.document_id for row in rows}) == 1


def test_loader_validates_frozen_qa_role_seed(tmp_path, tokenizer):
    record = make_record(6)
    preparation = write_dataset(tmp_path, [record])
    preparation["qa"]["role_seed"] = 123
    (tmp_path / "preparation.json").write_text(json.dumps(preparation))
    with pytest.raises(ValueError, match="role order"):
        load_factqa(tmp_path, tokenizer)


def test_loader_does_not_replay_source_selection_from_construction_snapshot(tmp_path, tokenizer):
    records = [make_record(split=split, segment_chars=3073) for split in ("train", "dev", "test")]
    preparation = write_dataset(tmp_path, records)
    preparation["source_pool_config"]["source_seed"] = 123
    preparation["source_pool_config"]["split_counts"] = {"train": 1000, "dev": 0, "test": 0}
    (tmp_path / "preparation.json").write_text(json.dumps(preparation), encoding="utf-8")
    result = load_factqa(tmp_path, tokenizer)
    assert {split: len(rows) for split, rows in result.items()} == {"train": 1, "dev": 1, "test": 1}
    for split, raw in zip(("train", "dev", "test"), records, strict=True):
        assert result[split][0].split == raw["split"]
        assert result[split][0].text == raw["text"]
