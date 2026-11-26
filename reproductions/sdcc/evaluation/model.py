"""从同一基座分别加载 SDCC 的编码器和解码器，以及各自的 LoRA。"""

import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from benchmarks.locomo.records import file_identity
from reproductions.sdcc.source.sdcc.modeling_ctxcomp import CtxCompSemiDynamicModel
from .locomo_adapter import SDCCLoCoMo, sdcc_input_format


class SDCCForEvaluation(CtxCompSemiDynamicModel):
    def _get_context_encoder_last_hidden(self, input_ids, attention_mask):
        # 本地部署独立加载编码器；倍率选择与池化只需要最后一层，长历史无需驻留所有层。
        positions = None if attention_mask is None else attention_mask.long().cumsum(1).sub_(1).clamp_min_(0)
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, position_ids=positions,
                            use_cache=False, output_hidden_states=False, return_dict=True).last_hidden_state


def lora_directory(component):
    """官方发布的 LoRA 分别保存在组件根目录或名为 resumed 的 adapter 目录。"""
    directories = [path for path in (component, component / "resumed")
                   if (path / "adapter_config.json").is_file()]
    if len(directories) != 1:
        raise ValueError(f"expected exactly one SDCC LoRA adapter in {component} or {component / 'resumed'}")
    return directories[0]


def add_arguments(parser):
    parser.add_argument("--model-path", type=Path, required=True, help="本地基座模型目录")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SDCC 半动态压缩权重目录")
    parser.add_argument("--compress-ratio-scale", type=float, default=0.0,
                        help="加在选择器预测值上的偏移量，默认 0，增大时倾向使用更少的记忆")


def configuration(parser, args):
    if not math.isfinite(args.compress_ratio_scale):
        parser.error("--compress-ratio-scale must be finite")
    base = args.model_path.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    if config["base_encoder_model_path"] != config["base_decoder_model_path"]:
        parser.error("--checkpoint must declare the same encoder and decoder base for --model-path")
    if (config["feature_extract_method"] != "mean_pooling"
            or not isinstance(config["comp_ratio_or_len"], list) or "discretize_ratio_mode" not in config):
        parser.error("--checkpoint must contain the semi-dynamic mean-pooling model")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    checkpoint_files = {}
    for path in (checkpoint / "config.json", checkpoint / "projector.pth",
                 lora_directory(checkpoint / "encoder"), lora_directory(checkpoint / "decoder")):
        checkpoint_files.update(file_identity(path))
    revision = (Path(__file__).resolve().parents[1] / "source/revision").read_text().strip()
    return {
        "model_path": str(base), "model_files": file_identity(base),
        "checkpoint": str(checkpoint), "checkpoint_files": checkpoint_files,
        "upstream_revision": revision, "compress_ratio_scale": args.compress_ratio_scale,
        "retention_options": config["comp_ratio_or_len"],
        "input_format": sdcc_input_format(tokenizer, config["placeholder_token_id"]),
        "dtype": "bfloat16", "attention_implementation": "flash_attention_2",
    }


def load_adapter(settings):
    if not torch.cuda.is_available():
        raise RuntimeError("SDCC evaluation requires CUDA and FlashAttention 2")
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
    bridge = SDCCForEvaluation(
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


def load_longmemeval_adapter(settings):
    return load_adapter(settings)
