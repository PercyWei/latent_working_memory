"""本地评估加载与单样本生成，保留 OpenCompass 原始路径作为对照。"""

import importlib.metadata
import inspect
from pathlib import Path

from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM, TTTDynamicCache
from opencompass.models import HuggingFaceBaseModel
from opencompass.models.huggingface_above_v4_33 import (
    _get_stopping_criteria,
    _set_model_kwargs_torch_dtype,
)
from optimized.model import CompactTTTDynamicCache, OptimizedQwen3ForCausalLM
from transformers.cache_utils import DynamicCache


MODEL_CLASSES = {
    "original": Qwen3ForCausalLM,
    "optimized": OptimizedQwen3ForCausalLM,
}
MODEL_KWARGS = {
    "device_map": "cuda:0",
    "torch_dtype": "torch.bfloat16",
    "local_files_only": True,
}
TOKENIZER_KWARGS = {"local_files_only": True}
GENERATION_KWARGS = {
    "do_sample": False,
    "top_k": 20,
    "top_p": 0.95,
    "use_cache": True,
}


class EvaluationModel(HuggingFaceBaseModel):
    def __init__(self, checkpoint, inference_mode, mlp_chunk_size, max_seq_len):
        self.inference_mode = inference_mode
        self.mlp_chunk_size = mlp_chunk_size
        super().__init__(
            path=str(checkpoint),
            model_kwargs=MODEL_KWARGS.copy(),
            tokenizer_kwargs=TOKENIZER_KWARGS.copy(),
            generation_kwargs=GENERATION_KWARGS.copy(),
            max_seq_len=max_seq_len,
        )

    def _load_model(self, path, kwargs, peft_path=None, peft_kwargs=None):
        if peft_path is not None:
            raise ValueError("This evaluation loads the complete local Qwen3 checkpoint")
        model_kwargs = {"device_map": "auto", "trust_remote_code": True, **kwargs}
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs, path)
        model_class = MODEL_CLASSES[self.inference_mode]
        self.model, loading = model_class.from_pretrained(
            path, output_loading_info=True, **model_kwargs
        )
        failures = {
            key: loading[key]
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading[key]
        }
        if failures:
            raise ValueError(f"Checkpoint did not load strictly: {failures}")
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.generation_config.do_sample = False
        if self.inference_mode == "optimized":
            self.model.set_optimization(True, self.mlp_chunk_size)


def load_model(checkpoint, inference_mode="original", mlp_chunk_size=16384, max_seq_len=262144):
    if inference_mode not in MODEL_CLASSES:
        raise ValueError(f"Unknown inference mode: {inference_mode}")
    if mlp_chunk_size < 0:
        raise ValueError("mlp_chunk_size must be nonnegative")
    return EvaluationModel(
        Path(checkpoint).expanduser().resolve(), inference_mode, mlp_chunk_size, max_seq_len
    )


def generate_one(wrapper, prompt, max_out_len=64, max_total_tokens=262144):
    if wrapper.inference_mode == "original":
        input_tokens = len(
            wrapper.tokenizer.encode(prompt, add_special_tokens=True, truncation=False)
        )
        if input_tokens + max_out_len > max_total_tokens or input_tokens > wrapper.max_seq_len:
            raise ValueError(
                "Prompt plus generation budget exceeds the requested length; truncation is forbidden"
            )
        captured = {}
        original_generate = wrapper.model.generate
        instance_generate = wrapper.model.__dict__.get("generate")

        def capture_generate(*args, **kwargs):
            outputs = original_generate(*args, **kwargs)
            prefix_length = kwargs["input_ids"].shape[1]
            captured["token_ids"] = outputs[0, prefix_length:].detach().cpu().tolist()
            return outputs

        wrapper.model.generate = capture_generate
        try:
            # The native wrapper still performs its own tokenization and text decoding.
            raw_prediction = wrapper.generate(inputs=[prompt], max_out_len=max_out_len)[0]
        finally:
            if instance_generate is None:
                del wrapper.model.generate
            else:
                wrapper.model.generate = instance_generate
        token_ids = captured["token_ids"]
    elif wrapper.inference_mode == "optimized":
        # Valid inputs never reach native truncation; encode once and reuse exactly these IDs.
        tokens = wrapper.tokenizer.batch_encode_plus(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )
        input_tokens = tokens["input_ids"].shape[1]
        if input_tokens + max_out_len > max_total_tokens or input_tokens > wrapper.max_seq_len:
            raise ValueError(
                "Prompt plus generation budget exceeds the requested length; truncation is forbidden"
            )
        tokens = {name: value.to(wrapper.model.device) for name, value in tokens.items()}
        generation_kwargs = wrapper.generation_kwargs.copy()
        stop_words = list(set(wrapper.stop_words))
        if stop_words:
            generation_kwargs["stopping_criteria"] = _get_stopping_criteria(
                stop_words, wrapper.tokenizer, 1
            )
        generation_kwargs["max_new_tokens"] = max_out_len
        generation_kwargs["pad_token_id"] = wrapper.tokenizer.pad_token_id
        tokens.pop("token_type_ids", None)
        outputs = wrapper.model.generate(**tokens, **generation_kwargs)
        continuation = outputs[:, input_tokens:]
        token_ids = continuation[0].detach().cpu().tolist()
        raw_prediction = wrapper.tokenizer.batch_decode(continuation, skip_special_tokens=True)[0]
        for stop in stop_words:
            raw_prediction = raw_prediction.split(stop)[0]
    else:
        raise ValueError(f"Unknown inference mode: {wrapper.inference_mode}")
    return {
        "input_tokens": input_tokens,
        "raw_prediction": raw_prediction,
        "token_ids": token_ids,
        "generated_tokens": len(token_ids),
    }


def runtime_info(wrapper):
    config = wrapper.model.config
    if isinstance(wrapper.model, OptimizedQwen3ForCausalLM):
        compact_tails = wrapper.model.compact_tails
        mlp_chunk_size = wrapper.model.model.layers[0].mlp.mlp_chunk_size
    else:
        compact_tails = False
        mlp_chunk_size = 0
    cache_class = (
        (CompactTTTDynamicCache if compact_tails else TTTDynamicCache)
        if config.ttt_mode
        else DynamicCache
    )
    mlp_class = type(wrapper.model.model.layers[0].mlp)
    return {
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "opencompass")
        },
        "model_class": f"{type(wrapper.model).__module__}.{type(wrapper.model).__qualname__}",
        "inference_mode": wrapper.inference_mode,
        "compact_tails": compact_tails,
        "mlp_chunk_size": mlp_chunk_size,
        "cache_class": f"{cache_class.__module__}.{cache_class.__qualname__}",
        "cache_code": str(Path(inspect.getfile(cache_class)).resolve()),
        "mlp_class": f"{mlp_class.__module__}.{mlp_class.__qualname__}",
        "mlp_code": str(Path(inspect.getfile(mlp_class)).resolve()),
        "ttt_mode": config.ttt_mode,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_scaling": config.rope_scaling,
        "max_seq_len": wrapper.max_seq_len,
        "attention_implementation": config._attn_implementation,
        "parameter_dtype": str(next(wrapper.model.parameters()).dtype),
        "generation_kwargs": wrapper.generation_kwargs,
        "model_generation_config": wrapper.model.generation_config.to_dict(),
        "tokenizer": {
            "padding_side": wrapper.tokenizer.padding_side,
            "truncation_side": wrapper.tokenizer.truncation_side,
            "bos_token_id": wrapper.tokenizer.bos_token_id,
            "eos_token_id": wrapper.tokenizer.eos_token_id,
            "pad_token_id": wrapper.tokenizer.pad_token_id,
        },
        "postprocessor": "opencompass.utils.text_postprocessors.extract_non_reasoning_content",
    }
