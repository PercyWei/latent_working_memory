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

from latent_working_memory.benchmarks.data import BenchmarkSample, locomo_question_prompt
from latent_working_memory.benchmarks.locomo.engine import evaluate_conversations
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
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]",
                               eos_token="[EOS]", unk_token="[UNK]").save_pretrained(root)
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
        history = "t4 t5 t6 t7 " * 140
        samples = [BenchmarkSample(
            f"c/{i}", history + locomo_question_prompt(question, 4), ("t4",), 4,
            {"conversation_id": "c", "question": question, "category": 4, "category_name": "single-hop"},
        ) for i, question in enumerate(("t8", "t9", "t10 t11"))]
        for stage in ("pretrained", "finetuned"):
            original = ICAELoCoMo(model, stage)
            adapter = ICAEAdapter(original)
            state, _ = original.encode_history(history)
            expected = {
                sample.id: original.answer(state, locomo_question_prompt(sample.metadata["question"], 4), 4)
                for sample in samples
            }
            for ordered, batch in ((samples, 2), (list(reversed(samples)), 1)):
                answers = {}
                evaluate_conversations(ordered, adapter, batch,
                                       lambda s, a, stats: answers.update({s.id: a["prediction"]}))
                assert answers == expected
            prepared = adapter.prepare_context(history, samples)
            before = [layer[0].clone() for layer in prepared.state[0]]
            adapter.answer_batch(prepared.state, prepared.questions[:2], lambda *args: None)
            assert all(torch.equal(keys, layer[0]) for keys, layer in zip(before, prepared.state[0]))
