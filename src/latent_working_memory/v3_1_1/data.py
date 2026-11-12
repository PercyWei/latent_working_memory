"""单次 AE／LM 样本与按文档分组的 QA；选样由稳定 ID 决定。"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

from latent_working_memory.v4_1.data import EpisodeEncoder, document_id

AE_PROMPT = "Reconstruct the original text:\n"
LM_PROMPT = "Continue the text:\n"


@dataclass(frozen=True)
class PretrainExample:
    id: str
    document_id: str
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    task: str
    prompt_ids: tuple[int, ...]


@dataclass(frozen=True)
class QA:
    question: str
    answer: str
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    segment_id: str | None = None
    segment_char_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class Document:
    id: str
    document_id: str
    input_ids: tuple[int, ...]
    qas: tuple[QA, ...]


def _digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(encoded, digest_size=20).hexdigest()


def data_identity(examples):
    """绑定 loader 返回的有序 ID 字典及其实际输入、目标和提示 tokens。"""
    digest = hashlib.blake2b(digest_size=20)
    for identifier, example in examples.items():
        digest.update(json.dumps(
            [identifier, asdict(example)], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode())
    return {
        "samples": len(examples),
        "documents": len({row.document_id for row in examples.values()}),
        "tokenized_content": digest.hexdigest(),
    }


def tokenization_identity(tokenizer):
    """词表、特殊符号和实际聊天拼接协议均属于实验身份。"""
    return {
        "tokenizer": tokenizer.name_or_path,
        "class": type(tokenizer).__name__,
        "vocabulary": _digest(tokenizer.get_vocab()),
        "special_tokens": tokenizer.special_tokens_map,
        "chat_template": tokenizer.chat_template,
        "qa_encoding": EpisodeEncoder(tokenizer).identity(),
        "ae_prompt": AE_PROMPT,
        "lm_prompt": LM_PROMPT,
    }


def _select(examples, max_samples, seed, split, allow_empty=False):
    if max_samples is not None and len(examples) > max_samples:
        examples.sort(key=lambda item: hashlib.blake2b(
            f"{seed}:{split}:{item.id}".encode(), digest_size=16,
        ).digest())
        examples = examples[:max_samples]
    if not examples and not allow_empty:
        raise ValueError(f"{split}: no documents remain after filtering")
    return {example.id: example for example in examples}


def _path(directory, split):
    return Path(directory).expanduser().resolve() / f"{split}.jsonl"


def _sources(directory, split, length_curriculum=False):
    root = Path(directory).expanduser().resolve()
    if (root / split).is_dir():
        preparation = json.loads((root / f"preparation-{split}.json").read_text(encoding="utf-8"))
        bins = preparation["bins"]
        return root / split, [(root / item["file"], index) for index, item in enumerate(bins)], bins
    if split == "train" and length_curriculum:
        raise ValueError("length curriculum requires offline length bins; run prepare-length first")
    path = _path(root, split)
    return path, [(path, None)], None


def _lines(sources):
    sample_number = 0
    for path, bin_index in sources:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                sample_number += 1
                yield path, line_number, sample_number, bin_index, line


def _ordered_bins(examples, bin_by_id, bins):
    if bins is None:
        return examples, None
    groups = [[] for _ in bins]
    for example in examples.values():
        groups[bin_by_id[example.id]].append(example)
    ordered = {example.id: example for group in groups for example in group}
    statistics = [{
        "file": item["file"], "min_tokens": item["min_tokens"], "max_tokens": item["max_tokens"],
        "samples": len(group),
    } for item, group in zip(bins, groups, strict=True)]
    return ordered, statistics


def _limit(settings, split):
    if split not in {"train", "dev"}:
        raise ValueError("training data split must be train or dev")
    return getattr(settings.training.data, f"max_{split}_samples")


def _metadata(path, split, examples, raw_documents, raw_texts, filtered, maximum, seed):
    return {
        "file": str(path), "split": split,
        "raw_document_ids": sorted(raw_documents),
        "raw_text_ids": sorted(raw_texts),
        "raw_documents": len(raw_documents),
        "selected_samples": len(examples), "episodes": len(examples),
        "selected_documents": len({row.document_id for row in examples.values()}),
        "filtered_input_length": filtered,
        "selection_seed": seed, "max_samples": maximum,
        "identity": data_identity(examples),
    }


def load_pretraining(settings, tokenizer, split):
    """完整文本只选 AE 或 LM；LM 预测末 N tokens，压缩前缀不足时转 AE。"""
    maximum = _limit(settings, split)
    data, pretrain, seed = settings.training.data, settings.method.pretrain, settings.seed
    data_path, sources, bins = _sources(data.dataset_dir, split, data.length_curriculum)
    if tokenizer.eos_token_id is None:
        raise ValueError("AE/LM pretraining requires an EOS token")
    prompts = {
        "ae": tuple(tokenizer.encode(AE_PROMPT, add_special_tokens=False)),
        "lm": tuple(tokenizer.encode(LM_PROMPT, add_special_tokens=False)),
    }
    examples, raw_documents, bin_by_id = [], set(), {}
    raw_samples, filtered, short_lm_inputs = 0, 0, 0
    for path, line_number, sample_number, bin_index, line in _lines(sources):
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"text"}:
                raise ValueError("expected exactly text")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("text must be a nonempty string")
            doc_id = document_id(row["text"])
            raw_documents.add(doc_id)
            raw_samples += 1
            full_ids = tuple(tokenizer.encode(row["text"], add_special_tokens=False))
            if len(full_ids) <= settings.model.structure.memory_slots or (
                split == "train" and data.max_train_data_tokens is not None
                and len(full_ids) > data.max_train_data_tokens
            ):
                filtered += 1
                continue
            rng = random.Random(f"{seed}:pretraining:{doc_id}")
            task = "lm" if rng.random() < pretrain.lm_ratio else "ae"
            inputs = target = full_ids
            if task == "lm":
                if len(full_ids) - pretrain.lm_target_tokens <= settings.model.structure.memory_slots:
                    task = "ae"
                    short_lm_inputs += 1
                else:
                    inputs = full_ids[:-pretrain.lm_target_tokens]
                    target = full_ids[-pretrain.lm_target_tokens:]
            example = PretrainExample(
                f"{doc_id}:{sample_number}:{task}", doc_id, inputs,
                (*target, tokenizer.eos_token_id), task, prompts[task],
            )
            examples.append(example)
            if bin_index is not None:
                bin_by_id[example.id] = bin_index
        except (ValueError, KeyError, TypeError) as error:
            error.add_note(f"{path}:{line_number}")
            raise
    examples = _select(examples, maximum, seed, split, allow_empty=split == "dev")
    examples, length_bins = _ordered_bins(examples, bin_by_id, bins)
    metadata = _metadata(
        data_path, split, examples, raw_documents, raw_documents, filtered, maximum, seed,
    )
    metadata.update({
        "raw_samples": raw_samples,
        "lm_to_ae_sources": short_lm_inputs,
        "ae_samples": sum(row.task == "ae" for row in examples.values()),
        "lm_samples": sum(row.task == "lm" for row in examples.values()),
        "lm_ratio": pretrain.lm_ratio, "lm_target_tokens": pretrain.lm_target_tokens,
    })
    if length_bins is not None:
        metadata["length_bins"] = length_bins
    return examples, metadata


def _load_documents(directory, tokenizer, split, minimum, upper, maximum, seed,
                    allow_empty=False, length_curriculum=False):
    data_path, sources, bins = _sources(directory, split, length_curriculum)
    encoder = EpisodeEncoder(tokenizer)
    examples, raw_documents, bin_by_id, filtered, raw_qas = [], set(), {}, 0, 0
    for path, line_number, _, bin_index, line in _lines(sources):
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"text", "qas"}:
                raise ValueError("expected exactly text and qas")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("text must be a nonempty string")
            if not isinstance(row["qas"], list) or not row["qas"]:
                raise ValueError("qas must be a nonempty list")
            identifier = document_id(row["text"])
            if identifier in raw_documents:
                raise ValueError("duplicate document; put all QA for one document in one row")
            raw_documents.add(identifier)
            qas = []
            for qa in row["qas"]:
                if not isinstance(qa, dict) or set(qa) not in (
                    {"question", "answer"},
                    {"question", "answer", "segment_id", "segment_char_span"},
                ):
                    raise ValueError(
                        "QA requires question and answer; segment_id and segment_char_span "
                        "must be supplied together, with no other fields"
                    )
                if any(not isinstance(qa[name], str) or not qa[name].strip()
                       for name in ("question", "answer")):
                    raise ValueError("question and answer must be nonempty strings")
                segment_id = qa.get("segment_id")
                segment_span = qa.get("segment_char_span")
                if "segment_id" in qa:
                    if not isinstance(segment_id, str) or not segment_id.strip():
                        raise ValueError("segment_id must be a nonempty string")
                    if (not isinstance(segment_span, list) or len(segment_span) != 2
                        or any(type(position) is not int for position in segment_span)
                        or not 0 <= segment_span[0] < segment_span[1] <= len(row["text"])):
                        raise ValueError(
                            "segment_char_span must contain two integer character positions "
                            "within the complete text"
                        )
                    segment_span = tuple(segment_span)
                qas.append(QA(
                    qa["question"], qa["answer"], encoder.question(qa["question"]),
                    encoder.target(qa["answer"]), segment_id, segment_span,
                ))
            raw_qas += len(qas)
            body_length = len(tokenizer.encode(row["text"], add_special_tokens=False))
            if body_length < minimum or (upper is not None and body_length > upper):
                filtered += 1
                continue
            inputs = encoder.context(row["text"])
            examples.append(Document(identifier, identifier, inputs, tuple(qas)))
            if bin_index is not None:
                bin_by_id[identifier] = bin_index
        except (ValueError, KeyError, TypeError) as error:
            error.add_note(f"{path}:{line_number}")
            raise
    examples = _select(examples, maximum, seed, split, allow_empty=allow_empty)
    examples, length_bins = _ordered_bins(examples, bin_by_id, bins)
    metadata = _metadata(
        data_path, split, examples, raw_documents, raw_documents, filtered, maximum, seed,
    )
    metadata.update({"raw_qas": raw_qas, "selected_qas": sum(len(x.qas) for x in examples.values())})
    if length_bins is not None:
        metadata["length_bins"] = length_bins
    return examples, metadata


def load_documents(settings, tokenizer, split):
    data = settings.training.data
    return _load_documents(
        data.dataset_dir, tokenizer, split, settings.model.structure.memory_slots + 1,
        data.max_train_data_tokens if split == "train" else None,
        _limit(settings, split), settings.seed, allow_empty=split == "dev",
        length_curriculum=data.length_curriculum,
    )


def load_evaluation_documents(directory, tokenizer, max_samples=None, seed=42):
    """读取 test.jsonl，按文档限量，不应用训练长度筛选。"""
    return _load_documents(
        directory, tokenizer, "test", 1, None, max_samples, seed,
    )
