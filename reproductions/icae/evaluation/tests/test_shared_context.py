"""在 ICAE 独立环境检查两阶段的正文缓存与批量问题分支。"""

from contextlib import nullcontext
from pathlib import Path
import os
import tempfile
from unittest.mock import patch

import torch
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

from benchmarks.data import BenchmarkSample, locomo_question_prompt
from benchmarks.locomo.engine import evaluate_conversations
from latent_working_memory.prompts import QA_SYSTEM_PROMPT
from reproductions.icae.evaluation.model import ICAE, ICAELoCoMo, ModelArguments, TrainingArguments
from reproductions.icae.evaluation.locomo_adapter import ICAEAdapter


def test_both_stages_preserve_independent_answers():
    torch.manual_seed(12)
    torch.set_num_threads(1)
    device = torch.device(os.environ.get("LOCOMO_TEST_DEVICE", "cpu"))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        MistralForCausalLM(MistralConfig(
            vocab_size=32000, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
            sliding_window=None, eos_token_id=2,
        )).save_pretrained(root)
        vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                      **{f"t{i}": i for i in range(4, 64)}}
        tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]",
            eos_token="[EOS]", unk_token="[UNK]",
            chat_template=("{{ bos_token }}[INST] {{ messages[0]['content'] }}"
                           "\n\n{{ messages[1]['content'] }} [/INST]"),
        ).save_pretrained(root)
        load = AutoModelForCausalLM.from_pretrained
        def cpu_model(path, **kwargs):
            return load(path, torch_dtype=torch.float32, attn_implementation="sdpa")
        override = patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=cpu_model)
        with override if device.type == "cpu" else nullcontext():
            model = ICAE(
                ModelArguments(model_name_or_path=str(root), train=False),
                TrainingArguments(output_dir=str(root), use_cpu=device.type == "cpu", report_to=[]),
                LoraConfig(r=4, lora_alpha=8, bias="none", task_type="CAUSAL_LM"),
            ).eval().to(device)
        with torch.no_grad():
            for name, weight in model.named_parameters():
                if "lora_B." in name:
                    weight.normal_(std=0.05)
        histories = ["t4 t5 t6 t7 " * 140, "t17 t22 " * 47]
        samples = [BenchmarkSample(
            f"{doc}/{i}", histories[doc] + locomo_question_prompt(question, 4), ("t4",), budget,
            {"conversation_id": str(doc), "question": question, "category": 4, "category_name": "single-hop"},
        ) for i, (doc, question, budget) in enumerate(((0, "t8", 3), (0, "t9 t11", 2),
                                                      (0, "t10", 2), (1, "t45 t46 t47", 4)))]
        for stage in ("pretrained", "finetuned"):
            original = ICAELoCoMo(model, stage)
            assert original.prefix_ids == (
                model.tokenizer.encode("[BOS][INST]", add_special_tokens=False)
                + model.tokenizer.encode(QA_SYSTEM_PROMPT + "\n\n", add_special_tokens=False)
            )
            adapter = ICAEAdapter(original)
            documents = [(histories[0], samples[:3]), (histories[1], samples[3:])]
            shapes = []
            def record_inputs(module, args, kwargs):
                shapes.append(tuple(kwargs["inputs_embeds"].shape[:2]))
            hook = original.decoder.model.register_forward_pre_hook(record_inputs, with_kwargs=True)
            contexts = adapter.prepare_contexts(documents)
            hook.remove()
            assert shapes[0][0] == 2
            expected_tokens = {}
            # 官方逐篇 _compress 与不复用 KV 的单题解码作为独立基准。
            with torch.inference_mode():
                for context, (history, _) in zip(contexts, documents, strict=True):
                    ids = model.tokenizer.build_inputs_with_special_tokens(
                        model.tokenizer.encode(history, add_special_tokens=False))
                    slots = model._compress(original.token_tensor(ids))
                    prefix = model.tokens_to_embeddings(original.token_tensor(original.prefix_ids))
                    memory = torch.cat((prefix, slots.to(prefix).unsqueeze(0)), dim=1)
                    assert context.statistics["memory_tokens"] == slots.size(0)
                    with model.icae.disable_adapter():
                        expected_cache = original.decoder.model(inputs_embeds=memory, use_cache=True).past_key_values
                        for actual_layer, layer in zip(context.state[0], expected_cache, strict=True):
                            for actual, expected in zip(actual_layer, layer, strict=True):
                                tolerance = 1e-4 if actual.dtype == torch.float32 else 1e-2
                                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                        for question in context.questions:
                            sample = question.sample
                            prompt = locomo_question_prompt(sample.metadata["question"], 4)
                            tokens = model.tokenizer.encode(prompt, add_special_tokens=False)
                            if stage == "finetuned":
                                tokens = [model.ft_token_id] + tokens
                            tokens += model.tokenizer.encode(" [/INST]", add_special_tokens=False)
                            assert question.inputs == tokens
                            embeds = torch.cat((memory, model.tokens_to_embeddings(original.token_tensor(tokens))), dim=1)
                            generated = []
                            for _ in range(sample.max_new_tokens):
                                output = original.decoder(inputs_embeds=embeds, use_cache=False)
                                token = output.logits[:, -1, :model.vocab_size - 1].argmax(-1)
                                generated.append(token.item())
                                if token.item() == original.eos_id:
                                    break
                                embeds = torch.cat((embeds, original.decoder.get_input_embeddings()(token[:, None])), dim=1)
                            expected_tokens[sample.id] = generated

            for ordered, batch in ((samples, 2), (list(reversed(samples)), 4), (samples, 1)):
                outputs = {}
                evaluate_conversations(ordered, adapter, batch, lambda s, a, _:
                                       outputs.update({s.id: (a["generated_token_ids"], a["decode_batch_size"])}))
                assert {i: tokens for i, (tokens, _) in outputs.items()} == expected_tokens
                assert all(size == batch for _, size in outputs.values())
            before = [layer[0].clone() for layer in contexts[0].state[0]]
            adapter.answer_batch([contexts[0].state] * 2, contexts[0].questions[:2], lambda *args: None)
            assert all(torch.equal(keys, layer[0]) for keys, layer in zip(before, contexts[0].state[0]))
