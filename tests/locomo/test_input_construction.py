"""输入包装重构回归：对照原始拼接，保留 token、记忆边界及答案起始。"""

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from benchmarks.data import BenchmarkSample, locomo_question_prompt
from benchmarks.locomo.causal import CausalLoCoMo
from benchmarks.protocol import encode_prompt
from latent_working_memory.prompts import QA_SYSTEM_PROMPT, QAInputBuilder, qa_input_format
from latent_working_memory.v3_1_1.data import QAEncoder
from reproductions.autocompressors.evaluation.locomo_adapter import AutoCompressorLoCoMo
from reproductions.autocompressors.evaluation.locomo_adapter import INPUT_FORMAT as AC_FORMAT
from reproductions.icae.formatting import icae_input_format
from reproductions.ttt_locomo import INPUT_FORMAT as TTT_FORMAT


@pytest.fixture(params=("mistral", "qwen"))
def tokenizer(request):
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    special = ["[UNK]", "<bos>", "<eos>", "<|im_start|>", "<|im_end|>"]
    backend.train_from_iterator([
        QA_SYSTEM_PROMPT, locomo_question_prompt("Where does Alice live?", 4),
        "Alice moved to Paris.\n\nQuestion: When?\nAnswer: Paris",
        "[INST] [/INST] system user assistant 你好，今天是周二。",
    ], trainers.BpeTrainer(vocab_size=400, special_tokens=special, show_progress=False))
    if request.param == "mistral":
        backend.post_processor = processors.TemplateProcessing(
            single="<bos> $A", special_tokens=[("<bos>", backend.token_to_id("<bos>"))],
        )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       bos_token="<bos>", eos_token="<eos>")
    tokenizer.chat_template = (
        "{{ bos_token }}[INST] {{ messages[0]['content'] }}\n\n{{ messages[1]['content'] }} [/INST]"
        "{% if messages|length > 2 %} {{ messages[2]['content'] }}{{ eos_token }}{% endif %}"
        if request.param == "mistral" else
        "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    return tokenizer


def sample(history, question="Where does Alice live?", category=4):
    return BenchmarkSample("c/0", history + locomo_question_prompt(question, category), ("Paris",), 5,
                           {"conversation_id": "c", "question": question, "category": category})


def chat(tokenizer, system, content, answer=None):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=answer is None,
                                         enable_thinking=False)


@pytest.mark.parametrize("history", ["Alice moved to Paris.\n\n", " leading and trailing space ", "你好。\n"])
def test_whole_chat_and_memory_boundaries_match_original(tokenizer, history):
    row = sample(history)
    builder = QAInputBuilder(tokenizer, qa_input_format(tokenizer))
    original = chat(tokenizer, QA_SYSTEM_PROMPT, row.prompt)
    assert builder.render(row.prompt) == original
    assert encode_prompt(row, tokenizer, 10000, "locomo", True) == tokenizer.encode(original, add_special_tokens=False)
    parts = builder.parts()
    wrapped = chat(tokenizer, QA_SYSTEM_PROMPT, "__LWM_DOCUMENT_PLACEHOLDER__")
    before, after = wrapped.split("__LWM_DOCUMENT_PLACEHOLDER__")
    assert (parts.prefix, parts.suffix) == (before, after)
    if "<|im_start|>" in tokenizer.chat_template:
        encoder = QAEncoder(tokenizer)
        assert encoder.reader_prefix_ids == tuple(tokenizer.encode(before, add_special_tokens=False))
        assert encoder.context(history) == tuple(tokenizer.encode(history, add_special_tokens=False))
        question = locomo_question_prompt(row.metadata["question"], 4)
        assert tokenizer.encode("\n\n" + question + encoder.suffix, add_special_tokens=False) == tokenizer.encode(
            "\n\n" + question + after, add_special_tokens=False,
        )
        ending = chat(tokenizer, QA_SYSTEM_PROMPT, "__LWM_DOCUMENT_PLACEHOLDER__", "__ANSWER__").split("__ANSWER__")[1]
        assert encoder.target("Paris") == tuple(tokenizer.encode("Paris" + ending, add_special_tokens=False))


@pytest.mark.parametrize("stage", ["pretrained", "finetuned"])
def test_icae_preserves_separate_tokenization_and_ft_position(tokenizer, stage):
    # 冻结原适配的双占位解析和分段分词，避免新实现与自身比较。
    original = chat(tokenizer, "<|icae_system_placeholder|>", "<|icae_memory_placeholder|>")
    instruction, rest = original.split("<|icae_system_placeholder|>")
    separator, suffix = rest.split("<|icae_memory_placeholder|>")
    question = locomo_question_prompt("When did Alice move?", 2)
    for slots in (64, 128):
        ft_id = len(tokenizer) + 1 + slots + 2
        settings = icae_input_format(tokenizer, stage, ft_id, QA_SYSTEM_PROMPT)
        assert settings["use_chat_template"] is True
        assert "type" not in settings
        assert settings["reader_prefix_ids"] == (
            tokenizer.encode(instruction.rstrip(" "), add_special_tokens=False)
            + tokenizer.encode(QA_SYSTEM_PROMPT + separator, add_special_tokens=False)
        )
        expected_question = ([ft_id] if stage == "finetuned" else []) + tokenizer.encode(question, add_special_tokens=False)
        expected_question += tokenizer.encode(suffix, add_special_tokens=False)
        assert settings["question_prefix_ids"] + tokenizer.encode(question, add_special_tokens=False) + settings["answer_prefix_ids"] == expected_question


@pytest.mark.parametrize("answer_prefix", ["", "Answer:", "\nAnswer:"])
def test_unwrapped_input_is_explicit_even_with_chat_template(tokenizer, answer_prefix):
    builder = QAInputBuilder(tokenizer, qa_input_format(use_chat_template=False, answer_prefix=answer_prefix))
    content = sample("Alice moved to Paris.\n\n").prompt
    original = QA_SYSTEM_PROMPT + "\n\n" + content + answer_prefix
    assert builder.render(content) == original
    assert builder.encode(content) == tokenizer.encode(original, add_special_tokens=True)
    assert builder.parts().prefix + content + builder.parts().suffix == original


def test_missing_chat_template_never_changes_selected_wrapping(tokenizer):
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="one explicit tokenizer chat template"):
        qa_input_format(tokenizer, use_chat_template=True)
    builder = QAInputBuilder(tokenizer, qa_input_format(use_chat_template=False))
    assert builder.render("question") == QA_SYSTEM_PROMPT + "\n\nquestionAnswer:"


class RecordingModel(torch.nn.Module):
    """只记录适配器实际送入的 ID／embedding；不模拟模型生成。"""

    def __init__(self, tokenizer):
        super().__init__()
        self.embedding = torch.nn.Embedding(len(tokenizer), 4)
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(summary_length=2, max_position_embeddings=10000)
        self.generation_config = SimpleNamespace(eos_token_id=tokenizer.eos_token_id)
        self.writes = []
        self.prefix_embeddings = None

    def get_input_embeddings(self):
        return self.embedding

    def model(self, inputs_embeds, use_cache):
        self.prefix_embeddings = inputs_embeds.clone()
        length = inputs_embeds.shape[1]
        return SimpleNamespace(past_key_values=((torch.zeros(1, length, 2, 1, 4), length),))

    def forward(self, input_ids, softprompt=None, output_softprompt=False, **kwargs):
        if output_softprompt:
            self.writes.append(input_ids.clone())
            slots = torch.full((1, 2, 4), -1.0)
            return SimpleNamespace(softprompt=slots if softprompt is None else torch.cat((softprompt, slots), dim=1))
        return SimpleNamespace(past_key_values=input_ids.clone())


def test_autocompressor_actual_prefix_memory_and_questions(tokenizer):
    model = RecordingModel(tokenizer)
    history = "Alice moved to Paris.\n\n" * 2
    row = sample(history)
    adapter = AutoCompressorLoCoMo(model, tokenizer, 8, AC_FORMAT)
    prepared = adapter.prepare_contexts([(history, [row])])[0]
    body = tokenizer.encode(history, add_special_tokens=False)
    assert torch.cat(model.writes, dim=1).tolist() == [body]
    prefix_ids = tokenizer.encode(QA_SYSTEM_PROMPT + "\n\n", add_special_tokens=True)
    prefix = model.embedding(torch.tensor([prefix_ids]))
    torch.testing.assert_close(model.prefix_embeddings[:, :len(prefix_ids)], prefix, rtol=0, atol=0)
    slots = prepared.statistics["memory_tokens"]
    assert model.prefix_embeddings.shape[1] == len(prefix_ids) + slots
    assert torch.all(model.prefix_embeddings[:, len(prefix_ids):] == -1)
    assert prepared.questions[0].inputs == tokenizer.encode(
        locomo_question_prompt(row.metadata["question"], 4) + "Answer:", add_special_tokens=False,
    )


def test_shared_ttt_actual_prefix_and_question_tokens(tokenizer):
    history = "Alice moved to Paris.\n\n"
    rows = [sample(history), sample(history, "When?", 2)]
    adapter = CausalLoCoMo(RecordingModel(tokenizer), tokenizer, None, TTT_FORMAT)
    prepared = adapter.prepare_contexts([(history, rows)])[0]
    original_prefix = tokenizer.encode(QA_SYSTEM_PROMPT + "\n\n" + history, add_special_tokens=True)
    probe = tokenizer.encode(QA_SYSTEM_PROMPT + "\n\n" + history + locomo_question_prompt("", 4) + "\nAnswer:", add_special_tokens=True)
    shared = next((i for i, (a, b) in enumerate(zip(original_prefix, probe)) if a != b), min(len(original_prefix), len(probe)))
    prefix = prepared.state[0][0].tolist()
    assert prefix == original_prefix[:shared]
    for question in prepared.questions:
        original = tokenizer.encode(QA_SYSTEM_PROMPT + "\n\n" + question.sample.prompt + "\nAnswer:", add_special_tokens=True)
        assert prefix + question.inputs == original
