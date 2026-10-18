"""读取 text/qas 文档，展开独立问答样本并按长度过滤、限量选样。"""

import hashlib
import json
from pathlib import Path

from latent_working_memory.v4.data_types import TokenizedEpisode as TokenizedEpisode


QA_SYSTEM_PROMPT = (
    "Answer questions using the provided document. Use exact details whenever possible. "
    "Answer with a short span, yes, no, or a concise phrase. "
    "If the document does not contain the answer, reply exactly: No information available."
)


def _episode_statistics(episodes):
    lengths = [len(episode.prompt_ids) + len(episode.target_ids) for episode in episodes]
    groups = {episode.group_id for episode in episodes if episode.group_id is not None}
    return {
        "episodes": len(episodes),
        "groups": len(groups),
        "tokens": sum(lengths),
        "target_tokens": sum(len(episode.target_ids) for episode in episodes),
        "length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
    }


def _validate_lengths(min_text_tokens, max_train_data_tokens, model_window):
    if type(min_text_tokens) is not int or min_text_tokens < 1:
        raise ValueError("min_text_tokens must be a positive integer")
    if max_train_data_tokens is not None and (
        type(max_train_data_tokens) is not int or max_train_data_tokens < 1
    ):
        raise ValueError("max_train_data_tokens must be null or a positive integer")
    if type(model_window) is not int or model_window <= min_text_tokens:
        raise ValueError("model_window must be an integer > min_text_tokens")


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
    """展开所有 QA，保留完整文档集合并按文档记录原始正文 token 数。"""
    path = Path(path).resolve()
    episodes, document_ids, references, text_lengths = [], set(), {}, {}
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
            text_lengths[group_id] = len(tokenizer.encode(row["text"], add_special_tokens=False))
            for qa_index, qa in enumerate(row["qas"]):
                qa_location = f"{location}:qas[{qa_index}]"
                if not isinstance(qa, dict) or set(qa) != {"question", "answer"}:
                    raise ValueError(
                        f"{qa_location}: expected an object containing only question and answer"
                    )
                for name in ("question", "answer"):
                    if not isinstance(qa[name], str) or not qa[name].strip():
                        raise ValueError(f"{qa_location}: {name} must be a nonempty string")
                episode = _tokenize_qa(row["text"], qa, tokenizer, group_id, qa_index, qa_location)
                episodes.append(episode)
                references[episode.id] = qa
    if not episodes:
        raise ValueError(f"{path}: dataset must contain at least one document")
    return (
        tuple(episodes),
        {"file": str(path), "documents": len(document_ids), "qas": len(episodes)},
        references,
        text_lengths,
    )


def _filter_split(
    episodes, text_lengths, min_text_tokens, max_train_data_tokens, model_window, split
):
    raw_groups = sorted({episode.group_id for episode in episodes if episode.group_id is not None})
    raw = {
        "documents": len(raw_groups),
        "qas": len(episodes),
        "group_ids": raw_groups,
        "episode_ids": [episode.id for episode in episodes],
    }
    after_text_minimum = tuple(
        episode for episode in episodes if text_lengths[episode.group_id] >= min_text_tokens
    )
    text_minimum = {
        "scope": "text",
        "min_text_tokens": min_text_tokens,
        "read": len(episodes),
        "kept": len(after_text_minimum),
        "filtered_too_short": len(episodes) - len(after_text_minimum),
    }
    train_limit = max_train_data_tokens if split == "train" else None
    after_train_limit = (
        tuple(
            episode
            for episode in after_text_minimum
            if text_lengths[episode.group_id] <= train_limit
        )
        if train_limit is not None
        else after_text_minimum
    )
    training_limit = {
        "applied": train_limit is not None,
        "scope": "text",
        "max_train_data_tokens": train_limit,
        "read": len(after_text_minimum),
        "kept": len(after_train_limit),
        "filtered_too_long": len(after_text_minimum) - len(after_train_limit),
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
    if not after_model_window:
        raise ValueError(f"{split}: no episodes remain after length filtering")
    statistics = {
        **_episode_statistics(after_model_window),
        "raw": raw,
        "filtering": {
            "text_minimum": text_minimum,
            "training_limit": training_limit,
            "model_window": model_limit,
        },
    }
    return after_model_window, statistics


def load_tokenized_episodes(
    path, tokenizer, min_text_tokens, max_train_data_tokens, model_window
):
    """加载训练文件，依次应用正文长度下限、上限和完整序列模型窗口上限。"""
    _validate_lengths(min_text_tokens, max_train_data_tokens, model_window)
    episodes, _, _, text_lengths = _load_raw_episodes(path, tokenizer)
    retained, _ = _filter_split(
        episodes, text_lengths, min_text_tokens, max_train_data_tokens, model_window, split="train"
    )
    return retained


def load_evaluation_episodes(
    path, tokenizer, min_text_tokens, model_window, seed, max_test_samples=None
):
    """测试集先检查正文下限与模型窗口，再固定抽取 QA；保留原始划分统计。"""
    _validate_lengths(min_text_tokens, None, model_window)
    if max_test_samples is not None and (type(max_test_samples) is not int or max_test_samples < 1):
        raise ValueError("max_test_samples must be a positive integer or null")
    episodes, source, references, text_lengths = _load_raw_episodes(path, tokenizer)
    retained, statistics = _filter_split(
        episodes, text_lengths, min_text_tokens, None, model_window, split="test"
    )
    selected = _select_episodes(retained, max_test_samples, seed=seed, split="test")
    statistics.update(
        **_episode_statistics(selected),
        selection={
            "seed": seed,
            "max_samples": max_test_samples,
            "read": len(retained),
            "kept": len(selected),
        },
    )
    return (
        selected,
        {episode.id: references[episode.id] for episode in selected},
        {"source": source, **statistics},
    )


def _select_episodes(episodes, limit, seed, split):
    if limit is None or limit >= len(episodes):
        return episodes

    def priority(episode):
        return hashlib.blake2b(f"{seed}:{split}:{episode.id}".encode(), digest_size=16).digest()

    return tuple(sorted(episodes, key=priority)[:limit])


def load_episode_splits(
    train_file,
    dev_file,
    tokenizer,
    min_text_tokens,
    max_train_data_tokens,
    model_window,
    seed,
    max_train_samples,
    max_dev_samples,
):
    _validate_lengths(min_text_tokens, max_train_data_tokens, model_window)
    train_path, dev_path = Path(train_file).resolve(), Path(dev_file).resolve()
    train, train_source, _, train_text_lengths = _load_raw_episodes(train_path, tokenizer)
    dev, dev_source, _, dev_text_lengths = _load_raw_episodes(dev_path, tokenizer)
    train_groups = {episode.group_id for episode in train if episode.group_id is not None}
    dev_groups = {episode.group_id for episode in dev if episode.group_id is not None}
    if train_groups & dev_groups:
        raise ValueError("raw train and dev files contain overlapping text documents")
    train, train_statistics = _filter_split(
        train,
        train_text_lengths,
        min_text_tokens,
        max_train_data_tokens,
        model_window,
        split="train",
    )
    dev, dev_statistics = _filter_split(
        dev, dev_text_lengths, min_text_tokens, max_train_data_tokens, model_window, split="dev"
    )
    splits = {"train": train, "dev": dev}
    split_statistics = {"train": train_statistics, "dev": dev_statistics}
    limits = {"train": max_train_samples, "dev": max_dev_samples}
    for split, episodes in splits.items():
        selected = _select_episodes(episodes, limits[split], seed, split)
        splits[split] = selected
        split_statistics[split].update(
            {
                **_episode_statistics(selected),
                "selection": {
                    "max_samples": limits[split],
                    "read": len(episodes),
                    "kept": len(selected),
                },
            }
        )
    statistics = {
        "min_text_tokens": min_text_tokens,
        "max_train_data_tokens": max_train_data_tokens,
        "model_window": model_window,
        "training_limit_split": "train",
        "partition": {"mode": "explicit_files"},
        "selection": {
            "seed": seed,
            "max_train_samples": max_train_samples,
            "max_dev_samples": max_dev_samples,
        },
        "sources": [train_source, dev_source],
        **split_statistics,
    }
    return splits["train"], splits["dev"], statistics


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
