"""读取 text/qas 文档，展开独立问答样本并按训练约束过滤。"""

import hashlib
import json
from pathlib import Path

from latent_working_memory.v4.data_types import TokenizedEpisode as TokenizedEpisode


QA_SYSTEM_PROMPT = (
    "Answer questions using the provided document. Use exact details whenever possible. "
    "Answer with a short span, yes, no, or a concise phrase. "
    "If the document does not contain the answer, reply exactly: No information available."
)


def _episode_statistics(episodes, min_length):
    lengths = [len(episode.prompt_ids) + len(episode.target_ids) for episode in episodes]
    groups = {episode.group_id for episode in episodes if episode.group_id is not None}
    return {
        "episodes": len(episodes),
        "groups": len(groups),
        "tokens": sum(lengths),
        "target_tokens": sum(
            len(episode.prompt_ids)
            + len(episode.target_ids)
            - max(len(episode.prompt_ids), min_length - 1)
            for episode in episodes
        ),
        "length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
    }


def _validate_lengths(min_length, max_seq_length, model_window):
    if type(min_length) is not int or min_length < 2:
        raise ValueError("min_length must be an integer >= 2")
    if max_seq_length is not None and (type(max_seq_length) is not int or max_seq_length < 1):
        raise ValueError("max_seq_length must be null or a positive integer")
    if type(model_window) is not int or model_window < min_length:
        raise ValueError("model_window must be an integer >= min_length")


def _tokenize_qa(text, qa, tokenizer, group_id, qa_index, location):
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"Document:\n{text}\n\nQuestion:\n{qa['question']}"},
    ]
    prompt_ids = tuple(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    complete_ids = tuple(
        tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": qa["answer"]}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if complete_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"{location}: chat-template generation prompt is not a token prefix of the "
            "complete answer; the template cannot define an exact answer supervision boundary"
        )
    target_ids = complete_ids[len(prompt_ids) :]
    if not target_ids:
        raise ValueError(f"{location}: chat template produced no assistant target tokens")
    return TokenizedEpisode(f"{group_id}:{qa_index}", prompt_ids, target_ids, group_id)


def _load_raw_episodes(path, tokenizer):
    """验证统一格式并展开所有 QA；保留过滤前的完整文档集合。"""
    path = Path(path).resolve()
    episodes, document_ids = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON") from error
            if not isinstance(row, dict) or set(row) != {"text", "qas"}:
                raise ValueError(f"{location}: expected an object containing only text and qas")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError(f"{location}: text must be a nonempty string")
            if not isinstance(row["qas"], list) or not row["qas"]:
                raise ValueError(f"{location}: qas must be a nonempty list")
            group_id = hashlib.blake2b(row["text"].encode("utf-8"), digest_size=16).hexdigest()
            if group_id in document_ids:
                raise ValueError(f"{location}: duplicate text; combine its QA pairs in one row")
            document_ids.add(group_id)
            for qa_index, qa in enumerate(row["qas"]):
                qa_location = f"{location}:qas[{qa_index}]"
                if not isinstance(qa, dict) or set(qa) != {"question", "answer"}:
                    raise ValueError(
                        f"{qa_location}: expected an object containing only question and answer"
                    )
                for name in ("question", "answer"):
                    if not isinstance(qa[name], str) or not qa[name].strip():
                        raise ValueError(f"{qa_location}: {name} must be a nonempty string")
                episodes.append(
                    _tokenize_qa(row["text"], qa, tokenizer, group_id, qa_index, qa_location)
                )
    if not episodes:
        raise ValueError(f"{path}: dataset must contain at least one document")
    return tuple(episodes), {
        "file": str(path),
        "documents": len(document_ids),
        "qas": len(episodes),
    }


def _filter_split(episodes, min_length, max_seq_length, model_window, split):
    raw_groups = sorted({episode.group_id for episode in episodes if episode.group_id is not None})
    raw = {
        "documents": len(raw_groups),
        "qas": len(episodes),
        "group_ids": raw_groups,
        "episode_ids": [episode.id for episode in episodes],
    }
    train_limit = max_seq_length if split == "train" else None
    after_train_limit = (
        tuple(
            episode
            for episode in episodes
            if len(episode.prompt_ids) + len(episode.target_ids) <= train_limit
        )
        if train_limit is not None
        else episodes
    )
    training_limit = {
        "applied": train_limit is not None,
        "max_seq_length": train_limit,
        "read": len(episodes),
        "kept": len(after_train_limit),
        "filtered_too_long": len(episodes) - len(after_train_limit),
    }
    after_model_window = tuple(
        episode
        for episode in after_train_limit
        if len(episode.prompt_ids) + len(episode.target_ids) <= model_window
    )
    model_limit = {
        "max_position_embeddings": model_window,
        "read": len(after_train_limit),
        "kept": len(after_model_window),
        "filtered_too_long": len(after_train_limit) - len(after_model_window),
    }
    retained = []
    supervision = {
        "min_seq_length": min_length,
        "read": len(after_model_window),
        "filtered_too_short": 0,
        "filtered_prompt_before_first_write": 0,
    }
    for episode in after_model_window:
        if len(episode.prompt_ids) + len(episode.target_ids) < min_length:
            supervision["filtered_too_short"] += 1
        elif len(episode.prompt_ids) < min_length - 1:
            supervision["filtered_prompt_before_first_write"] += 1
        else:
            retained.append(episode)
    supervision["kept"] = len(retained)
    if not retained:
        raise ValueError(f"{split}: no episodes remain after length and supervision filtering")
    result = tuple(retained)
    statistics = {
        **_episode_statistics(result, min_length),
        "raw": raw,
        "filtering": {
            "training_limit": training_limit,
            "model_window": model_limit,
            "supervision": supervision,
        },
    }
    return result, statistics


def load_tokenized_episodes(path, tokenizer, min_length, max_seq_length, model_window):
    """加载单个训练文件；显式模型窗口并应用与 train split 一致的过滤。"""
    _validate_lengths(min_length, max_seq_length, model_window)
    episodes, _ = _load_raw_episodes(path, tokenizer)
    retained, _ = _filter_split(episodes, min_length, max_seq_length, model_window, split="train")
    return retained


def load_episode_splits(
    train_file,
    dev_file,
    tokenizer,
    min_length,
    max_seq_length,
    model_window,
):
    _validate_lengths(min_length, max_seq_length, model_window)
    train_path, dev_path = Path(train_file).resolve(), Path(dev_file).resolve()
    train, train_source = _load_raw_episodes(train_path, tokenizer)
    dev, dev_source = _load_raw_episodes(dev_path, tokenizer)
    train_groups = {episode.group_id for episode in train if episode.group_id is not None}
    dev_groups = {episode.group_id for episode in dev if episode.group_id is not None}
    if train_groups & dev_groups:
        raise ValueError("raw train and dev files contain overlapping text documents")
    train, train_statistics = _filter_split(
        train, min_length, max_seq_length, model_window, split="train"
    )
    dev, dev_statistics = _filter_split(dev, min_length, max_seq_length, model_window, split="dev")
    statistics = {
        "min_seq_length": min_length,
        "max_seq_length": max_seq_length,
        "model_window": model_window,
        "training_limit_split": "train",
        "partition": {"mode": "explicit_files"},
        "sources": [train_source, dev_source],
        "train": train_statistics,
        "dev": dev_statistics,
    }
    return train, dev, statistics


def dataset_identity(episodes):
    """记录提示、监督目标与文档分组，避免续训时替换训练目标。"""
    digest = hashlib.blake2b(digest_size=32)
    for episode in episodes:
        digest.update(
            json.dumps(
                [episode.id, episode.prompt_ids, episode.target_ids, episode.group_id],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "episodes": len(episodes),
        "tokens": sum(len(episode.prompt_ids) + len(episode.target_ids) for episode in episodes),
        "fingerprint": digest.hexdigest(),
    }
