"""将每题的完整会话历史和问题放入独立的聊天上下文，按批贪心生成。"""

import torch

from benchmarks.longmemeval.data import format_history, format_question
from latent_working_memory.prompts import QAInputBuilder, qa_input_format
from .generation import left_pad
from .model import load_model


PREFILL_CHUNK_SIZE = 4096


class FullContextLongMemEval:
    def __init__(self, model, tokenizer, generation, settings):
        self.model = model
        self.tokenizer = tokenizer
        self.generation = generation
        self.prompt = QAInputBuilder(tokenizer, settings["input_format"])
        self.window = settings["model_window"]
        self.budget = settings["max_new_tokens"]
        self.chunk_size = settings["prefill_chunk_size"]
        self.mixed_precision = settings["model"]["mixed_precision"]

    @torch.inference_mode()
    def answer_batch(self, inputs):
        prompts = [self.prompt.encode(format_history(item) + format_question(item)) for item in inputs]
        width = max(map(len, prompts))
        if width + self.budget > self.window:
            raise ValueError(f"full-context input ({width}) + answer budget ({self.budget}) "
                             f"exceeds model window ({self.window})")
        ids, mask = left_pad(prompts, self.generation.pad_token_id, self.model.device)
        with torch.autocast(self.model.device.type, dtype=torch.bfloat16, enabled=self.mixed_precision):
            # 预填期间保留完整 KV，只分块计算中间激活。左填充行按各自真实 token 编位置。
            positions = (mask.cumsum(dim=1) - 1).masked_fill_(mask == 0, 1)
            cache = None
            for start in range(0, width - self.chunk_size, self.chunk_size):
                end = start + self.chunk_size
                cache = self.model.base_model(
                    input_ids=ids[:, start:end], attention_mask=mask[:, :end],
                    position_ids=positions[:, start:end], past_key_values=cache,
                    use_cache=True, return_dict=True,
                ).past_key_values
            tokens = self.model.generate(
                input_ids=ids, attention_mask=mask, past_key_values=cache,
                generation_config=self.generation, max_new_tokens=self.budget, use_model_defaults=False,
            )
        return self.tokenizer.batch_decode(tokens[:, width:].tolist(), skip_special_tokens=True)


def load_adapter(settings):
    tokenizer, model, generation = load_model(settings["model"], settings["seed"], torch.device("cuda", 0))
    if qa_input_format(tokenizer) != settings["input_format"]:
        raise ValueError("loaded chat input format differs from the evaluation plan")
    return FullContextLongMemEval(model, tokenizer, generation, settings)
