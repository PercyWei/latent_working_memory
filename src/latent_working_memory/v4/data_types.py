"""数据加载与训练 engine 共用的问答样本契约。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizedEpisode:
    id: str
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    group_id: str | None = None
