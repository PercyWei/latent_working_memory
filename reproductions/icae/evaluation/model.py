"""使用 ICAE v2 的预训练或指令微调权重压缩完整历史并回答问题。"""


import torch

from benchmarks.locomo.reader import prefill_prefixes
from latent_working_memory.prompts import QA_SYSTEM_PROMPT
from reproductions.icae.formatting import icae_input_format
from reproductions.icae.source.icae.code.icae_v2.modeling_icae_multi_span import (
    ICAE,
    ModelArguments,
    TrainingArguments,
    adapter_config,
)


UPSTREAM_COMMIT = "469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd"


class ICAELoCoMo:
    def __init__(self, model, stage, system_prompt=QA_SYSTEM_PROMPT):
        self.model = model
        self.stage = stage
        self.decoder = model.icae.get_base_model()
        self.device = model.memory_token_embed.weight.device
        self.window = self.decoder.config.max_position_embeddings
        self.eos_id = model.tokenizer.eos_token_id
        self.input_format = icae_input_format(
            model.tokenizer, stage, model.ft_token_id, system_prompt,
        )
        self.prefix_ids = self.input_format["reader_prefix_ids"]

    def token_tensor(self, ids):
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def question_ids(self, question_prompt):
        ids = self.model.tokenizer.encode(question_prompt, add_special_tokens=False)
        return self.input_format["question_prefix_ids"] + ids + self.input_format["answer_prefix_ids"]

    @torch.inference_mode()
    def encode_histories(self, histories):
        # 沿用官方 _compress 的逐篇分段尺寸。每轮并行处理各正文的下一段。
        segments = []
        stats = []
        for text in histories:
            body = self.model.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            ids = self.model.tokenizer.build_inputs_with_special_tokens(body)
            count = self.model.compute_num_segments(len(ids))
            width = (len(ids) + count - 1) // count
            writer_length = width + self.model.mem_size
            memory_tokens = count * self.model.mem_size
            if writer_length > self.window:
                raise ValueError(f"ICAE writer needs {writer_length} positions (window {self.window})")
            if len(self.prefix_ids) + memory_tokens >= self.window:
                raise ValueError(f"ICAE reader prefix exceeds position window {self.window}")
            segments.append([ids[i:i + width] for i in range(0, len(ids), width)])
            stats.append({"context_tokens": len(body), "memory_tokens": memory_tokens})

        prefix = self.model.tokens_to_embeddings(self.token_tensor(self.prefix_ids))[0]
        prefixes = [prefix.new_empty((len(self.prefix_ids) + row["memory_tokens"], prefix.size(-1)))
                    for row in stats]
        for embeds in prefixes:
            embeds[:len(self.prefix_ids)] = prefix
        for index in range(max(map(len, segments))):
            active = [i for i, spans in enumerate(segments) if index < len(spans)]
            spans = [segments[i][index] for i in active]
            span_lengths = [len(span) for span in spans]
            max_length = max(span_lengths)
            # 左侧填充让每行的最终 mem_size 个位置始终是该行的记忆槽。
            padded = torch.tensor(
                [[self.model.pad_token_id] * (max_length - len(span)) + span for span in spans],
                dtype=torch.long, device=self.device,
            )
            ids = torch.cat((padded, self.model.append_sequence.expand(len(active), -1)), dim=1)
            lengths = torch.tensor(span_lengths, device=self.device)
            mask = torch.arange(ids.size(1), device=self.device) >= max_length - lengths[:, None]
            positions = mask.long().cumsum(1).sub_(1).clamp_min_(0)
            hidden = self.decoder.model(inputs_embeds=self.model.tokens_to_embeddings(ids),
                                        attention_mask=mask, position_ids=positions,
                                        use_cache=False, return_dict=True).last_hidden_state
            start = len(self.prefix_ids) + index * self.model.mem_size
            for row, document in enumerate(active):
                prefixes[document][start:start + self.model.mem_size] = hidden[row, -self.model.mem_size:]
            del hidden

        with self.model.icae.disable_adapter():
            caches = prefill_prefixes(self.decoder, prefixes)
        return [((cache, len(embeds)), row) for cache, embeds, row in zip(caches, prefixes, stats, strict=True)]


def load_adapter(model_path, checkpoint, stage, mean_compression_rate, fixed_mem_size,
                 system_prompt=QA_SYSTEM_PROMPT):
    if not torch.cuda.is_available():
        raise RuntimeError("ICAE v2 evaluation requires CUDA and FlashAttention 2")
    lora = adapter_config()
    model_args = ModelArguments(model_name_or_path=str(model_path), lora_r=lora.r, train=False)
    inference_args = TrainingArguments(
        output_dir=str(checkpoint.parent), bf16=True, fixed_mem_size=fixed_mem_size,
        mean_compression_rate=mean_compression_rate, report_to=[],
    )
    # 特殊 token 读取 checkpoint 嵌入，PAD 有掩码，生成只在原词表内选词。
    model = ICAE(model_args, inference_args, lora, mean_resizing=False)
    model.memory_token_embed.bfloat16()
    model.load_adapter_weights(checkpoint)
    return ICAELoCoMo(model.to("cuda").eval(), stage, system_prompt)

