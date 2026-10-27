"""按文档和压缩边界持久化固定大小的 FP32 teacher 监督池。"""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch
import transformers
from transformers import AutoConfig

from ..data import EpisodeEncoder
from .prompts import PROMPT_SOURCE, SOURCE_PROMPTS


def content_identity(value):
    return hashlib.blake2b(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), digest_size=20
    ).hexdigest()


def build_cache_identity(config, tokenizer):
    """准备与训练共用身份构造，绑定 teacher、tokenizer、模板及离线采样设置。"""
    backbone = AutoConfig.from_pretrained(config.model.model_name_or_path)
    return {
        "teacher_model": config.model.model_name_or_path,
        "teacher_revision": backbone._commit_hash,
        "backbone_dtype": config.model.backbone_dtype,
        "architecture": {
            "layers": backbone.num_hidden_layers,
            "query_heads": backbone.num_attention_heads,
            "kv_heads": backbone.num_key_value_heads,
            "head_dim": backbone.head_dim,
            "rope_theta": backbone.rope_theta,
            "rope_scaling": backbone.rope_scaling,
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "encoding": content_identity(json.loads(tokenizer.backend_tokenizer.to_str())),
            "special_tokens": tokenizer.special_tokens_map,
            "chat_template": tokenizer.chat_template,
        },
        "input_format": EpisodeEncoder(tokenizer).identity(),
        "prompt_source": PROMPT_SOURCE,
        "prompts": list(SOURCE_PROMPTS[:config.teacher.num_prompts]),
        "pending_size": config.model.pending_size,
        "recent_size": config.model.recent_size,
        "response_max_tokens": config.teacher.response_max_tokens,
        "pool_size": config.teacher.pool_size,
        "pool_selection": "uniform-reservoir-without-replacement",
        "am_std_floor": config.model.am_std_floor,
        "seed": config.seed,
        "transformers": transformers.__version__,
    }


def initialize_cache(directory, identity, sample_size=128):
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "identity.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"{path}: teacher cache identity differs from this configuration")
    else:
        _atomic_json(path, identity)
    return TeacherCache(directory, sample_size=sample_size)


def _atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


class TeacherCache:
    def __init__(self, directory, sample_size=128, expected_identity=None):
        self.directory = Path(directory).expanduser().resolve()
        self.identity = json.loads((self.directory / "identity.json").read_text(encoding="utf-8"))
        if expected_identity is not None and self.identity != expected_identity:
            raise ValueError("teacher cache identity differs from the current experiment")
        self.identity_id = content_identity(self.identity)
        self.sample_size = sample_size
        if not 0 < sample_size <= self.identity["pool_size"]:
            raise ValueError("AM sample size must be between one and the cached pool size")

    def path(self, document_id, write_index):
        return self.directory / "documents" / document_id / f"write_{write_index}.safetensors"

    def _record(self, handle):
        record = json.loads(handle.metadata()["record"])
        if record["identity"] != self.identity_id:
            raise ValueError("boundary cache belongs to a different teacher configuration")
        return record

    def metadata(self, document_id, write_index):
        with safe_open(self.path(document_id, write_index), framework="pt", device="cpu") as handle:
            return self._record(handle)

    def sample(self, document_id, write_index, device, generator=None):
        # 每次从完整候选池重新抽取，池不会随着调用被消耗。
        indices = torch.randperm(
            self.identity["pool_size"], generator=generator,
            device="cpu" if generator is None else generator.device,
        )[:self.sample_size].cpu()
        with safe_open(self.path(document_id, write_index), framework="pt", device="cpu") as handle:
            self._record(handle)
            result = {
                name: handle.get_tensor(name).index_select(1, indices).to(device)
                for name in ("queries", "outputs", "log_mass")
            }
            result.update({
                name: handle.get_tensor(name).to(device)
                for name in ("scales_o", "scales_z")
            })
            result["position_indices"] = handle.get_tensor("positions").index_select(0, indices)
        return result

    def write(self, document_id, write_index, tensors, processed_end, compressed_end, context_ids,
              candidate_count):
        shape = self.identity["architecture"]
        expected = (
            shape["layers"], self.identity["pool_size"], shape["query_heads"], shape["head_dim"]
        )
        for name in ("queries", "outputs"):
            if tuple(tensors[name].shape) != expected or tensors[name].dtype != torch.float32:
                raise ValueError(f"{name}: expected FP32 {expected}")
        expected_shapes = {
            "log_mass": expected[:-1], "scales_o": (expected[0], expected[2]),
            "scales_z": (expected[0], expected[2]), "positions": (expected[1], 2),
        }
        for name, expected_shape in expected_shapes.items():
            if tuple(tensors[name].shape) != expected_shape:
                raise ValueError(f"{name}: expected {expected_shape}")
        if len(torch.unique(tensors["positions"], dim=0)) != expected[1]:
            raise ValueError("teacher pool must contain distinct prompt/token positions")
        for name, value in tensors.items():
            dtype = torch.int64 if name == "positions" else torch.float32
            if value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError(f"{name}: expected finite {dtype} values")
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
