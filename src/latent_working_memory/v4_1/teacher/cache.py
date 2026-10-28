"""按文档与压缩边界保存响应 token、固定位置池及完整池的标准化尺度。"""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch
import transformers
from transformers import AutoConfig, GenerationConfig

from ..data import EpisodeEncoder
from .prompts import PROMPT_SOURCE, SOURCE_PROMPTS


def content_identity(value):
    return hashlib.blake2b(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), digest_size=20
    ).hexdigest()


def build_cache_identity(config, tokenizer):
    """准备与训练共用身份构造，绑定 teacher、tokenizer、模板及离线采样设置。"""
    backbone = AutoConfig.from_pretrained(config.model.model_name_or_path)
    eos = GenerationConfig.from_pretrained(config.model.model_name_or_path).eos_token_id
    encoder = EpisodeEncoder(tokenizer)
    prompts = list(SOURCE_PROMPTS[:config.teacher.num_prompts])
    return {
        "teacher_model": config.model.model_name_or_path,
        "teacher_revision": backbone._commit_hash,
        "backbone_dtype": config.model.backbone_dtype,
        "eos_token_ids": sorted(set([eos] if isinstance(eos, int) else eos)),
        "architecture": {
            "layers": backbone.num_hidden_layers,
            "query_heads": backbone.num_attention_heads,
            "kv_heads": backbone.num_key_value_heads,
            "head_dim": backbone.head_dim,
            "vocab_size": backbone.vocab_size,
            "max_position_embeddings": backbone.max_position_embeddings,
            "rope_theta": backbone.rope_theta,
            "rope_scaling": backbone.rope_scaling,
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "encoding": content_identity(json.loads(tokenizer.backend_tokenizer.to_str())),
            "special_tokens": tokenizer.special_tokens_map,
            "chat_template": tokenizer.chat_template,
        },
        "input_format": encoder.identity(),
        "prompt_source": PROMPT_SOURCE,
        "prompts": prompts,
        "prompt_token_ids": [list(encoder.question(prompt)) for prompt in prompts],
        "pending_size": config.model.pending_size,
        "recent_size": config.model.recent_size,
        "response_max_tokens": config.teacher.response_max_tokens,
        "pool_size": config.teacher.pool_size,
        "pool_selection": "uniform-reservoir-without-replacement",
        "am_std_floor": config.model.am_std_floor,
        "seed": config.seed,
        "transformers": transformers.__version__,
    }


def initialize_cache(directory, identity):
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "identity.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError(
                f"{path}: teacher cache format or settings differ; prepare into a new directory"
            )
    else:
        _atomic_json(path, identity)
    return TeacherCache(directory)


def _atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


class TeacherCache:
    def __init__(self, directory, expected_identity=None):
        self.directory = Path(directory).expanduser().resolve()
        self.identity = json.loads((self.directory / "identity.json").read_text(encoding="utf-8"))
        if "prompt_token_ids" not in self.identity:
            raise ValueError("expected a response-token teacher cache; prepare into a new directory")
        if expected_identity is not None and self.identity != expected_identity:
            raise ValueError("teacher cache identity differs from the current experiment")
        prompts = self.identity["prompt_token_ids"]
        if not isinstance(prompts, list) or len(prompts) != len(self.identity["prompts"]):
            raise ValueError("prompt_token_ids must contain one token sequence per fixed prompt")
        vocab_size = self.identity["architecture"]["vocab_size"]
        if any(
            not isinstance(prompt, list) or not prompt
            or any(type(token) is not int or not 0 <= token < vocab_size for token in prompt)
            for prompt in prompts
        ):
            raise ValueError("prompt_token_ids must contain nonempty sequences of valid token IDs")
        self.identity_id = content_identity(self.identity)

    def path(self, document_id, write_index):
        return self.directory / "documents" / document_id / f"write_{write_index}.safetensors"

    def _record(self, handle, document_id, write_index):
        record = json.loads(handle.metadata()["record"])
        if record["identity"] != self.identity_id:
            raise ValueError("boundary cache belongs to a different teacher configuration")
        if (record["document_id"], record["write_index"]) != (document_id, write_index):
            raise ValueError("boundary cache document or write index differs from its requested path")
        compressed_end = write_index * self.identity["pending_size"]
        if (record["processed_end"], record["compressed_end"]) != (
            compressed_end + self.identity["recent_size"], compressed_end,
        ):
            raise ValueError("boundary cache positions differ from the configured window sizes")
        return record

    def metadata(self, document_id, write_index):
        with safe_open(self.path(document_id, write_index), framework="pt", device="cpu") as handle:
            return self._record(handle, document_id, write_index)

    def read(self, document_id, write_index):
        """读取小型 CPU 缓存；抽样及 teacher forcing 由运行期 replay 负责。"""
        with safe_open(self.path(document_id, write_index), framework="pt", device="cpu") as handle:
            record = self._record(handle, document_id, write_index)
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        return record, tensors

    def _validate_tensors(self, tensors, candidate_count):
        names = {"response_token_ids", "response_offsets", "positions", "scales_o", "scales_z"}
        if set(tensors) != names:
            raise ValueError("expected only response tokens, offsets, positions and AM scales")
        architecture = self.identity["architecture"]
        prompt_lengths = torch.tensor([len(ids) for ids in self.identity["prompt_token_ids"]])
        num_prompts = len(prompt_lengths)
        pool_size = self.identity["pool_size"]
        expected_shapes = {
            "response_offsets": (num_prompts + 1,), "positions": (pool_size, 2),
            "scales_o": (architecture["layers"], architecture["query_heads"]),
            "scales_z": (architecture["layers"], architecture["query_heads"]),
        }
        for name, expected_shape in expected_shapes.items():
            if tuple(tensors[name].shape) != expected_shape:
                raise ValueError(f"{name}: expected {expected_shape}")
        for name, value in tensors.items():
            dtype = torch.float32 if name in {"scales_o", "scales_z"} else torch.int64
            if value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError(f"{name}: expected finite {dtype} values")
            if value.device.type != "cpu":
                raise ValueError(f"{name}: persisted teacher data must be on CPU")
        response_ids = tensors["response_token_ids"]
        offsets = tensors["response_offsets"]
        if response_ids.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(response_ids):
            raise ValueError("response offsets must span the complete flat response token sequence")
        lengths = offsets.diff()
        if (lengths < 1).any() or (lengths > self.identity["response_max_tokens"]).any():
            raise ValueError("each prompt must have a response within the configured token budget")
        if (response_ids < 0).any() or (response_ids >= architecture["vocab_size"]).any():
            raise ValueError("response_token_ids contains an out-of-vocabulary token")
        sequence_lengths = prompt_lengths + lengths
        if candidate_count != int(sequence_lengths.sum()):
            raise ValueError("candidate_count differs from the stored prompt and response lengths")
        positions = tensors["positions"]
        if (positions < 0).any() or (positions[:, 0] >= num_prompts).any():
            raise ValueError("positions contains an invalid prompt index or negative token position")
        if (positions[:, 1] >= sequence_lengths[positions[:, 0]]).any():
            raise ValueError("positions extends beyond a stored prompt and response sequence")
        if len(torch.unique(positions, dim=0)) != pool_size:
            raise ValueError("teacher pool must contain distinct prompt/token positions")
        for name in ("scales_o", "scales_z"):
            if (tensors[name] < self.identity["am_std_floor"]).any():
                raise ValueError(f"{name}: scales must respect the configured standard-deviation floor")

    def write(self, document_id, write_index, tensors, processed_end, compressed_end, context_ids,
              candidate_count):
        self._validate_tensors(tensors, candidate_count)
        if (
            write_index < 1 or compressed_end != write_index * self.identity["pending_size"]
            or processed_end != compressed_end + self.identity["recent_size"]
            or processed_end > len(context_ids)
        ):
            raise ValueError("teacher boundary must follow the configured windows inside the context")
        record = {
            "identity": self.identity_id, "document_id": document_id,
            "write_index": write_index, "processed_end": processed_end,
            "compressed_end": compressed_end, "context_tokens": content_identity(context_ids),
            "candidate_count": candidate_count,
        }
        path = self.path(document_id, write_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            save_file(
                {name: tensor.cpu().contiguous() for name, tensor in tensors.items()},
                str(temporary), metadata={"record": json.dumps(record, sort_keys=True)},
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
