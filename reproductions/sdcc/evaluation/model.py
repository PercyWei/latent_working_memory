"""从同一基座分别加载 SDCC 的编码器和解码器，以及各自的 LoRA。"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from reproductions.sdcc.source.sdcc.modeling_ctxcomp import CtxCompSemiDynamicModel
from .locomo_adapter import SDCCLoCoMo, sdcc_input_format


def lora_directory(component):
    """官方发布的 LoRA 分别保存在组件根目录或名为 resumed 的 adapter 目录。"""
    directories = [path for path in (component, component / "resumed")
                   if (path / "adapter_config.json").is_file()]
    if len(directories) != 1:
        raise ValueError(f"expected exactly one SDCC LoRA adapter in {component} or {component / 'resumed'}")
    return directories[0]


def load_adapter(settings):
    if not torch.cuda.is_available():
        raise RuntimeError("SDCC LoCoMo requires CUDA and FlashAttention 2")
    checkpoint = Path(settings["checkpoint"])
    base = settings["model_path"]
    config = json.loads((checkpoint / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    input_format = sdcc_input_format(tokenizer, config["placeholder_token_id"])
    if input_format != settings["input_format"]:
        raise ValueError("loaded SDCC input format differs from the evaluation plan")
    kwargs = dict(local_files_only=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
                  device_map={"": 0})
    # 上游会将 encoder 的注意力设为双向；独立加载才能保留 reader 的因果注意力。
    encoder = PeftModel.from_pretrained(AutoModel.from_pretrained(base, **kwargs),
                                        lora_directory(checkpoint / "encoder"))
    decoder = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, **kwargs),
                                        lora_directory(checkpoint / "decoder"))
    bridge = CtxCompSemiDynamicModel(
        encoder, decoder,
        **{k: config[k] for k in (
            "placeholder_token_id", "memory_token_begin_id", "comp_ratio_or_len",
            "mlp_converter_hidden_dim", "encoder_training", "decoder_training",
            "feature_extract_method", "discretize_ratio_mode", "discretize_compare_in_log",
        )},
    )
    weights = torch.load(checkpoint / "projector.pth", map_location="cpu", weights_only=True)
    bridge.projector.load_state_dict(weights["projector"])
    bridge.compress_ratio_head.load_state_dict(weights["compress_ratio_head"])
    return SDCCLoCoMo(bridge.requires_grad_(False).eval(), tokenizer,
                      settings["compress_ratio_scale"], input_format)
