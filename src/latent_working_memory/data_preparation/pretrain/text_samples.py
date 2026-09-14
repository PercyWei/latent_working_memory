"""基础语料只保存原文；Episode 是训练时按当前 tokenizer 构造的视图。"""

import hashlib
import json
from dataclasses import asdict, dataclass

from latent_working_memory.v1.data import Episode, Read, Reference, Source


def tokenizer_identity(config):
    return {"name_or_path": config.model_name_or_path, "revision": config.model_revision}


def input_text_key(text):
    return hashlib.blake2b(" ".join(text.split()).encode()).hexdigest()


@dataclass(frozen=True)
class TextSample:
    sample_id: str
    document_id: str
    source_id: str
    dedup_cluster: str
    task: str
    text: str
    continuation: str | None
    x_char_span: list[int]
    y_char_span: list[int] | None
    boundary_method: str
    reference_input_tokens: int
    reference_target_tokens: int

    def __post_init__(self):
        if not all((self.sample_id, self.document_id, self.source_id, self.dedup_cluster)):
            raise ValueError("text sample requires source identifiers")
        if self.task not in {"ae", "continuation"} or not self.text.strip():
            raise ValueError("text sample requires non-empty text and an AE/continuation task")
        start, end = self.x_char_span
        if not 0 <= start < end or end - start != len(self.text):
            raise ValueError("input source text differs from character span")
        if self.reference_input_tokens <= 0 or self.reference_target_tokens <= 0:
            raise ValueError("reference token lengths must be positive")
        if self.task == "ae":
            if self.continuation is not None or self.y_char_span is not None:
                raise ValueError("AE stores its text once, without a continuation")
            if self.reference_target_tokens != self.reference_input_tokens:
                raise ValueError("AE reference lengths must agree")
        elif (
            not self.continuation
            or not self.continuation.strip()
            or self.y_char_span is None
            or self.y_char_span != [end, end + len(self.continuation)]
        ):
            raise ValueError("continuation must be adjacent to the input character span")
        if self.boundary_method not in {"pysbd_conservative", "random_token"}:
            raise ValueError("unknown construction boundary method")

    @classmethod
    def from_episode(cls, episode, document_text, tokenizer):
        source, read = episode.sources[0], episode.reads[0]
        p = source.provenance
        text = document_text[slice(*p["x_char_span"])]
        continuation = read.references[0].text if read.task == "continuation" else None
        target = document_text[slice(*p["y_char_span"])] if continuation is not None else text
        if target != read.references[0].text:
            raise ValueError("reference differs from original source text")
        if tuple(tokenizer.encode(text, add_special_tokens=False)) != episode.input_ids:
            raise ValueError("input tokens differ from original source text")
        return cls(
            episode.episode_id,
            source.document_id,
            source.source_id,
            p["dedup_cluster"],
            read.task,
            text,
            continuation,
            p["x_char_span"],
            p["y_char_span"],
            p["boundary_method"],
            len(episode.input_ids),
            len(tokenizer.encode(continuation, add_special_tokens=False))
            if continuation is not None
            else len(episode.input_ids),
        )

    def to_record(self):
        return asdict(self)

    def to_episode(self, tokenizer, config, variant):
        ids = tuple(tokenizer.encode(self.text, add_special_tokens=False))
        return self.to_episode_tokens(ids, config, variant)

    def to_episode_tokens(self, ids, config, variant):
        provenance = {
            "boundary_variant": variant,
            "boundary_method": self.boundary_method,
            "dedup_cluster": self.dedup_cluster,
            "input_text_key": input_text_key(self.text),
            "x_char_span": self.x_char_span,
            "y_char_span": self.y_char_span,
        }
        return Episode(
            self.sample_id,
            ids,
            (len(ids),),
            (Source(self.source_id, self.document_id, 0, len(ids), provenance),),
            (
                Read(
                    self.sample_id + ":read",
                    self.task,
                    len(ids),
                    config.ae_prompt if self.task == "ae" else config.lm_prompt,
                    (Reference(self.text if self.task == "ae" else self.continuation, ()),),
                ),
            ),
        )


def compact_metadata(metadata, audit):
    """Convert the completed construction record, including historical preparations."""
    result = {
        key: metadata[key]
        for key in (
            "preparation_id",
            "source_pool_id",
            "boundary_variant",
            "recipe",
            "input_histogram",
            "statistics",
        )
    }
    contract = metadata["contract"]
    result["tokenizer"] = {
        "name_or_path": contract["model_name_or_path"],
        "revision": contract["model_revision"],
    }
    result["source_pool"] = "../source-pool.json"
    result["checks"] = audit["checks"]
    result["lengths"] = audit["lengths"]
    if "reference_preparation_id" in metadata:
        result["reference_preparation_id"] = metadata["reference_preparation_id"]
    return json.loads(json.dumps(result))
