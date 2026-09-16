import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, set_seed

from latent_working_memory.v2.gmsa_config import GMSAConfig
from latent_working_memory.v2.gmsa import GMSA


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gmsa-base")
    set_seed(42)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model.save_pretrained(directory)
    tokens = [
        "[PAD]",
        "[BOS]",
        "[EOS]",
        "[UNK]",
        "red",
        "blue",
        "sky",
        "water",
        "What",
        "color",
        "?",
        "Restate",
        "the",
        "aforementioned",
        "Text",
        ".",
    ]
    tokenizer = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    fast.save_pretrained(directory)
    return directory


@pytest.fixture
def model(tiny_base):
    return GMSA(
        GMSAConfig(
            str(tiny_base),
            encoder_layers=2,
            alignment_layers=1,
            compression_ratios=(2, 4),
            lora_rank=2,
            lora_alpha=4,
            lora_dropout=0,
            attention_implementation="eager",
        )
    )
