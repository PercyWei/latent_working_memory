"""Full-context 评估的模型加载与精度设置。"""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

from benchmarks.locomo.records import file_identity
from latent_working_memory.prompts import qa_input_format


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


def parse_chat_evaluation(parser, argv):
    """问答评估共用的基座、精度及聊天模板配置。"""
    parser.add_argument("--model-path", type=Path, default=Path("~/models/Qwen3-4B-Instruct-2507"),
                        help="本地基座模型目录，默认 ~/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dtype", choices=tuple(WEIGHT_DTYPES), default="bf16",
                        help="基座权重精度，默认 bf16")
    parser.add_argument("--attention-implementation", choices=ATTENTION_IMPLEMENTATIONS,
                        default="flash_attention_2", help="注意力实现，默认 flash_attention_2")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="为 FP32 权重启用 PyTorch BF16 autocast，默认关闭")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    model_config = {
        "model_name_or_path": str(args.model_path.expanduser().resolve()),
        "backbone_dtype": WEIGHT_DTYPES[args.dtype],
        "attention_implementation": args.attention_implementation,
        "mixed_precision": args.mixed_precision,
    }
    validate_compute(model_config)
    model_path = Path(model_config["model_name_or_path"])
    settings = {
        "model": model_config,
        "model_files": file_identity(model_path),
        "model_window": AutoConfig.from_pretrained(model_path).max_position_embeddings,
        "input_format": qa_input_format(AutoTokenizer.from_pretrained(model_path)),
        "seed": args.seed,
    }
    return args, settings


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
