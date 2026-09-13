"""读取文本基础语料；已有实验的 tokenized Episode 数据仍由 EpisodeIndex 读取。"""

import json
from pathlib import Path

from latent_working_memory.data_preparation.fineweb import data_contract
from latent_working_memory.data_preparation.text_samples import TextSample, tokenizer_identity
from latent_working_memory.v1.data import EpisodeIndex


def validate_preparation(metadata, config):
    if "contract" in metadata:
        # Existing derived experiments and checkpoints use this exact training contract.
        if metadata["contract"] != data_contract(config):
            raise ValueError("data preparation contract differs from training config")


class TextSampleIndex(EpisodeIndex):
    """Index offsets and lengths only; tokenize selected text on demand, without disk copies."""

    def __init__(self, path, tokenizer, config, metadata):
        self.path = Path(path)
        self.tokenizer, self.config = tokenizer, config
        self.variant = metadata["boundary_variant"]
        self.groups, self.offsets, self.ids, self.input_lengths, self.tasks = {}, [], [], [], []
        self.source_ids, self.cluster_ids = set(), set()
        self.reference_lengths = metadata["tokenizer"] == tokenizer_identity(config)
        prompt_lengths = {
            "ae": len(tokenizer.encode(config.ae_prompt, add_special_tokens=False)),
            "continuation": len(tokenizer.encode(config.lm_prompt, add_special_tokens=False)),
        }
        seen = set()
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                sample = TextSample(**json.loads(line))
                expected_method = (
                    "pysbd_conservative" if self.variant == "semantic" else "random_token"
                )
                if sample.boundary_method != expected_method:
                    raise ValueError("sample boundary method differs from dataset metadata")
                if sample.sample_id in seen:
                    raise ValueError("duplicate sample_id")
                seen.add(sample.sample_id)
                size = eligible_input_length(
                    sample, tokenizer, config, self.reference_lengths, prompt_lengths
                )
                if size is None:
                    continue
                self.groups.setdefault(sample.document_id, []).append(len(self.offsets))
                self.offsets.append(offset)
                self.ids.append(sample.sample_id)
                self.input_lengths.append(size)
                self.tasks.append(sample.task)
                self.source_ids.add(sample.source_id)
                self.cluster_ids.add(sample.dedup_cluster)
        if not self.offsets:
            raise ValueError(
                f"no eligible text samples in {self.path}; length limits leave insufficient data"
            )

    def __getitem__(self, index):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            sample = TextSample(**json.loads(handle.readline()))
        return sample.to_episode(self.tokenizer, self.config, self.variant)


def pretraining_index(path, tokenizer, config):
    path = Path(path)
    metadata = json.loads((path.parent / "preparation.json").read_text())
    validate_preparation(metadata, config)
    if "contract" in metadata:
        return EpisodeIndex(path)
    return TextSampleIndex(path, tokenizer, config, metadata)


def eligible_input_length(sample, tokenizer, config, reference_lengths, prompt_lengths):
    size = (
        sample.reference_input_tokens
        if reference_lengths
        else len(tokenizer.encode(sample.text, add_special_tokens=False))
    )
    if size <= 0 or size > config.max_input_tokens or size + 1 > config.write_context_tokens:
        return None
    target_length = (
        sample.reference_target_tokens
        if reference_lengths
        else (
            size
            if sample.task == "ae"
            else len(tokenizer.encode(sample.continuation, add_special_tokens=False))
        )
    )
    if sample.task == "continuation" and target_length > config.max_continuation_tokens:
        return None
    # The full-context control must fit as well as compressed-memory reads.
    if (
        2 + max(size, config.pretrain_k_min) + prompt_lengths[sample.task] + target_length
        > config.read_context_tokens
    ):
        return None
    return size
