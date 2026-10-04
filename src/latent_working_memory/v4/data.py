"""严格读取原文 JSONL；分词与 epoch 排序均只在内存中进行。"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random


@dataclass(frozen=True)
class TokenizedEpisode:
    id: str
    input_ids: tuple[int, ...]


def load_tokenized_episodes(path, tokenizer, min_length, max_seq_length):
    path = Path(path)
    episodes, identifiers = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON") from error
            if (
                not isinstance(row, dict)
                or set(row) != {"id", "text"}
                or not isinstance(row["id"], str)
                or not isinstance(row["text"], str)
            ):
                raise ValueError(f'{location}: expected exactly {{"id": str, "text": str}}')
            if row["id"] in identifiers:
                raise ValueError(f"{location}: duplicate episode id {row['id']!r}")
            identifiers.add(row["id"])
            ids = tuple(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
            if not min_length <= len(ids) <= max_seq_length:
                raise ValueError(
                    f"{location}: episode {row['id']!r} has {len(ids)} tokens; "
                    f"required range is [{min_length}, {max_seq_length}] (no truncation)"
                )
            episodes.append(TokenizedEpisode(row["id"], ids))
    if not episodes:
        raise ValueError(f"{path}: dataset must contain at least one episode")
    return tuple(episodes)


def episode_order(episodes, seed, epoch):
    indices = list(range(len(episodes)))
    random.Random(f"{seed}:v4:{epoch}").shuffle(indices)
    return tuple(episodes[index] for index in indices)


def dataset_identity(episodes):
    """记录实际训练序列的身份，避免续训时悄悄替换文本或 tokenizer。"""
    digest = hashlib.blake2b(digest_size=32)
    for episode in episodes:
        digest.update(
            json.dumps(
                [episode.id, episode.input_ids], ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "episodes": len(episodes),
        "tokens": sum(len(episode.input_ids) for episode in episodes),
        "fingerprint": digest.hexdigest(),
    }
