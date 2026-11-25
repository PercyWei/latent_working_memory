"""小模型关键检查：正文分支复用、逐题原生对照、分块边界和问题顺序。"""

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from benchmarks.data import BenchmarkSample, locomo_question_prompt
from benchmarks.locomo.engine import evaluate_conversations
from latent_working_memory.prompts import QA_SYSTEM_PROMPT


def tokenizer_for_validation():
    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, **{f"t{i}": i for i in range(3, 64)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                  eos_token="[EOS]", unk_token="[UNK]")


@torch.no_grad()
def independent_answer(adapter, reference, sample, prefix_length, recompute=False):
    ids = torch.tensor([adapter.tokenizer.encode(
        QA_SYSTEM_PROMPT + "\n\n" + sample.prompt + "\nAnswer:", add_special_tokens=True,
    )], device=adapter.device)
    tokens = []
    with torch.autocast(adapter.device.type, dtype=torch.bfloat16, enabled=adapter.mixed_precision):
        if not recompute:
            output = reference(input_ids=ids[:, :prefix_length], use_cache=True, logits_to_keep=1)
            for position in range(prefix_length, ids.shape[1]):
                output = reference(input_ids=ids[:, position:position + 1],
                                   past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
        for _ in range(sample.max_new_tokens):
            if recompute:
                output = reference(input_ids=ids, use_cache=False, logits_to_keep=1)
            token = output.logits[0, -1].argmax().item()
            tokens.append(token)
            if token in adapter.eos_ids:
                break
            continuation = ids.new_tensor([[token]])
            if recompute:
                ids = torch.cat((ids, continuation), dim=1)
            else:
                output = reference(input_ids=continuation, past_key_values=output.past_key_values,
                                   use_cache=True, logits_to_keep=1)
    return tokens


def validate_context(adapter, reference, chunk_size, state_tensors, recompute=False):
    prefix_words = len(adapter.tokenizer.encode(QA_SYSTEM_PROMPT, add_special_tokens=True))
    for remainder in (chunk_size - 1, 0, 1):
        count = 2 * chunk_size + (remainder - prefix_words) % chunk_size
        history = "t3 t4 " * (count // 2) + ("t3 " if count % 2 else "")
        samples = [BenchmarkSample(
            f"c1/{index}", history + locomo_question_prompt(question, 4), ("t3",), 5,
            {"conversation_id": "c1", "question": question, "category": 4},
        ) for index, question in enumerate(("t5", "t6", "t7 t8"))]
        prepared = adapter.prepare_context(history, samples)
        assert prepared.state[1] % chunk_size == remainder
        assert prepared.statistics == {}
        values = list(state_tensors(prepared.state[0]))
        frozen = [None if value is None else value.clone() for value in values]
        expected = {s.id: independent_answer(adapter, reference, s, prepared.state[1], recompute)
                    for s in samples}
        actual = {}
        def completed(question, answer):
            actual[question.sample.id] = answer["generated_token_ids"]
        adapter.answer_batch(prepared.state, prepared.questions[:2], completed)
        adapter.answer_batch(prepared.state, prepared.questions[2:], completed)
        assert actual == expected, (remainder, actual, expected)
        after = list(state_tensors(prepared.state[0]))
        for before, value in zip(frozen, after, strict=True):
            if before is not None:
                torch.testing.assert_close(before, value, rtol=0, atol=0, equal_nan=True)
        reversed_answers = {}
        evaluate_conversations(list(reversed(samples)), adapter, 1,
                               lambda sample, answer, stats: reversed_answers.update(
                                   {sample.id: answer["generated_token_ids"]}))
        assert reversed_answers == expected
        resumed = adapter.prepare_context(history, samples[1:])
        assert resumed.statistics == prepared.statistics
    print("LoCoMo context checks passed: chunk boundaries, B1/B2, question order, immutable state", flush=True)
