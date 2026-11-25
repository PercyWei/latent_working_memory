"""Full-context 评估的模型加载与精度设置。"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed


WEIGHT_DTYPES = {"fp32": "float32", "bf16": "bfloat16"}
ATTENTION_IMPLEMENTATIONS = ("sdpa", "flash_attention_2")


def validate_compute(config):
    if config["backbone_dtype"] not in WEIGHT_DTYPES.values():
        raise ValueError("unsupported model.backbone_dtype")
    if config["attention_implementation"] not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError("attention implementation must be sdpa or flash_attention_2")
    amp = config["mixed_precision"]
    if type(amp) is not bool:
        raise ValueError("model.mixed_precision must be true or false")
    if amp and config["backbone_dtype"] != "float32":
        raise ValueError("--mixed-precision requires --dtype fp32")
    if (
        config["attention_implementation"] == "flash_attention_2"
        and config["backbone_dtype"] == "float32"
        and not amp
    ):
        raise ValueError("flash_attention_2 with --dtype fp32 requires --mixed-precision")


def load_model(config, seed, device):
    torch.cuda.set_device(device)
    set_seed(seed)
    model_path = config["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("benchmark generation requires tokenizer.eos_token_id")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=getattr(torch, config["backbone_dtype"]),
        attn_implementation=config["attention_implementation"],
        device_map=device,
    ).eval()
    generation = GenerationConfig(
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
    )
    return tokenizer, model, generation
