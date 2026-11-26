"""使用 ICAE v2 的预训练或指令微调权重压缩完整历史并回答问题。"""


import torch
from peft import LoraConfig
from safetensors.torch import load_file

from latent_working_memory.prompts import QA_SYSTEM_PROMPT
from reproductions.icae.formatting import icae_input_format
from reproductions.icae.source.icae.code.icae_v2.modeling_icae_multi_span import (
    ICAE,
    ModelArguments,
    TrainingArguments,
)


UPSTREAM_COMMIT = "469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd"


class ICAELoCoMo:
    def __init__(self, model, stage):
        self.model = model
        self.stage = stage
        self.decoder = model.icae.get_base_model()
        self.device = model.memory_token_embed.weight.device
        self.window = self.decoder.config.max_position_embeddings
        self.eos_id = model.tokenizer.eos_token_id
        self.input_format = icae_input_format(
            model.tokenizer, stage, model.ft_token_id, QA_SYSTEM_PROMPT,
        )
        self.prefix_ids = self.input_format["reader_prefix_ids"]

    def token_tensor(self, ids):
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def question_ids(self, question_prompt):
        ids = self.model.tokenizer.encode(question_prompt, add_special_tokens=False)
        return self.input_format["question_prefix_ids"] + ids + self.input_format["answer_prefix_ids"]

    @torch.inference_mode()
    def encode_history(self, text):
        body_ids = self.model.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        ids = self.model.tokenizer.build_inputs_with_special_tokens(body_ids)
        segments = self.model.compute_num_segments(len(ids))
        writer_positions = (len(ids) + segments - 1) // segments + self.model.mem_size
        if writer_positions > self.window:
            raise ValueError(f"ICAE writer needs {writer_positions} positions (window {self.window})")
        memory_tokens = segments * self.model.mem_size
        prefix_length = len(self.prefix_ids) + memory_tokens
        if prefix_length >= self.window:
            raise ValueError(f"ICAE reader prefix requires {prefix_length} tokens, window {self.window}")
        slots = self.model._compress(self.token_tensor(ids))
        prefix = self.model.tokens_to_embeddings(self.token_tensor(self.prefix_ids))
        embeddings = torch.cat((prefix, slots.to(prefix).unsqueeze(0)), dim=1)
        with self.model.icae.disable_adapter():
            # The reader reuses this tuple cache as a shared prefix for all questions.
            cache = self.decoder.model(inputs_embeds=embeddings, use_cache=True).past_key_values
        stats = {"context_tokens": len(body_ids), "memory_tokens": memory_tokens}
        return (cache, prefix_length), stats

    @torch.inference_mode()
    def answer(self, memory, question_prompt, max_new_tokens):
        cache, prefix_length = memory
        ids = self.question_ids(question_prompt)
        total_length = prefix_length + len(ids) + max_new_tokens
        if total_length > self.window:
            raise ValueError(f"ICAE QA requires {total_length} tokens, window {self.window}")
        inputs = self.model.tokens_to_embeddings(self.token_tensor(ids))
        generated = []
        with self.model.icae.disable_adapter():
            for _ in range(max_new_tokens):
                outputs = self.decoder(inputs_embeds=inputs, past_key_values=cache, use_cache=True)
                # Added padding/memory/operation rows are never answer tokens.
                token = outputs.logits[:, -1, : self.model.vocab_size - 1].argmax(dim=-1)
                token_id = token.item()
                if token_id == self.eos_id:
                    break
                generated.append(token_id)
                inputs = self.decoder.get_input_embeddings()(token.unsqueeze(1))
                cache = outputs.past_key_values
        return self.model.tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_adapter(model_path, checkpoint, stage, mean_compression_rate, fixed_mem_size):
    if not torch.cuda.is_available():
        raise RuntimeError("ICAE v2 evaluation requires CUDA and FlashAttention 2")
    model_args = ModelArguments(model_name_or_path=str(model_path), lora_r=512, train=False)
    inference_args = TrainingArguments(
        output_dir=str(checkpoint.parent), bf16=True, fixed_mem_size=fixed_mem_size,
        mean_compression_rate=mean_compression_rate, report_to=[],
    )
    lora = LoraConfig(r=512, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = ICAE(model_args, inference_args, lora)
    model.memory_token_embed.bfloat16()
    # Official safetensors contain the complete writer LoRA and slot embeddings,
    # while all frozen weights come from the base model directory.
    weights = load_file(str(checkpoint))
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    if set(weights) != expected:
        raise ValueError(
            f"ICAE adapter keys differ: missing={sorted(expected - weights.keys())}, "
            f"unexpected={sorted(weights.keys() - expected)}"
        )
    model.load_state_dict(weights, strict=False)
    return ICAELoCoMo(model.to("cuda").eval(), stage)

