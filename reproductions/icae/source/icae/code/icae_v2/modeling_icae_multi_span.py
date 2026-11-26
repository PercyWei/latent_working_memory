# ICAE that supports multi span concat

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from peft import (
    LoraConfig, get_peft_model,
)
import math
from safetensors.torch import load_file

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-v0.1")
    lora_r: int = field(
        default=128,
        metadata={"help": "lora rank"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "lora dropout"}
    )
    train: bool = field(
        default=True,
        metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    debug_data: bool = field(default=False, metadata={"help": "Enable debug dataset to quickly verify the training process"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    fixed_mem_size: int = field(
        default=128,
        metadata={"help": "Enalbing the fixed mem size."},
    )
    mean_compression_rate: int = field(
        default=4,
        metadata={"help": "Mean compression rate; default=4"},
    )
    min_tokens_for_lm: int = field(
        default=64,
        metadata={"help": "Minimum tokens for lm objective learning"},
    )
    leave_tokens_for_lm: int = field(
        default=8,
        metadata={"help": "Leave some tokens without loss for lm objective"},
    )
    lm_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio for LM training."},
    )
    add_special_token_for_lm: bool = field(
        default=False,
        metadata={"help": "Add a special token for the prompt of language modeling; default: False"},
    )
    restore_from: str = field(
        default="",
        metadata={"help": "The checkpoint that should be restored from for fine-tuning"}
    )

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


def adapter_config():
    return LoraConfig(r=512, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")


def segment_count(length, mem_size, compression_rate):
    if length <= 0:
        raise ValueError("ICAE requires at least one document token")
    return math.ceil(length / (mem_size * compression_rate))


class ICAE(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config, mean_resizing=None):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        load_options = {
            "dtype": torch.bfloat16 if training_args.bf16 else torch.float16,
            "attn_implementation": "flash_attention_2",
        }
        self.icae = AutoModelForCausalLM.from_pretrained(self.model_name, **load_options)
        self.training = self.model_args.train
        if self.training:
            self.decoder = AutoModelForCausalLM.from_pretrained(self.model_name, **load_options)

        self.vocab_size = self.icae.config.vocab_size + 1    # [PAD] token
        self.pad_token_id = self.vocab_size - 1
        self.mean_compression_rate = training_args.mean_compression_rate

        # tunable
        self.mem_size = self.training_args.fixed_mem_size
        self.vocab_size_with_mem = self.vocab_size + self.mem_size # so, the mem tokens are in the range [self.vocab_size, self.vocab_size + self.mem_size)

        # special tokens in addition to mem and length tokens
        self.ae_token_id = self.vocab_size_with_mem + 0
        self.lm_token_id = self.vocab_size_with_mem + 1
        self.ft_token_id = self.vocab_size_with_mem + 2        

        resize_options = {} if mean_resizing is None else {"mean_resizing": mean_resizing}
        self.icae.resize_token_embeddings(self.vocab_size_with_mem + 3, **resize_options)
        
        self.dim = self.icae.config.hidden_size
        self.icae = get_peft_model(self.icae, lora_config)

        self.memory_token_embed = nn.Embedding(self.mem_size + 3, self.dim, padding_idx=None)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.register_buffer("append_sequence", torch.arange(self.vocab_size, self.vocab_size_with_mem).unsqueeze(0),
                             persistent=False)
        
        if self.training:
            self.init()


    def init(self):
        print("Freezing the decoder...")
        freeze_model(self.decoder)
        self.decoder.eval()
        print_trainable_parameters(self)
        if self.training_args.restore_from is not None and self.training_args.restore_from != "":
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            self.load_adapter_weights(self.training_args.restore_from)
            print(f"Finished loading from {self.training_args.restore_from}")
        print("Enabling gradient checkpointing...")
        # self.icae.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                
        
    def adapter_parameters(self):
        return {name: param for name, param in self.named_parameters() if param.requires_grad}

    def load_adapter_weights(self, path):
        weights = load_file(str(path))
        expected = self.adapter_parameters().keys()
        if weights.keys() != expected:
            raise ValueError(
                f"ICAE adapter keys differ: missing={sorted(expected - weights.keys())}, "
                f"unexpected={sorted(weights.keys() - expected)}"
            )
        self.load_state_dict(weights, strict=False)

    def compute_num_segments(self, total_length):
        return segment_count(total_length, self.mem_size, self.mean_compression_rate)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        prompt_answer_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        if input_ids.size(0) != 1:
            raise ValueError("ICAE training requires one document per device per microbatch")
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)

        memories = []
        writer = self.icae.get_base_model().model
        for segment in input_ids.split(segment_length, dim=1):
            ids = torch.cat((segment, self.append_sequence), dim=1)
            hidden = writer(inputs_embeds=self.tokens_to_embeddings(ids), use_cache=False,
                            return_dict=True).last_hidden_state
            memories.append(hidden[0, -self.mem_size:])
        prompt_answer_embs = self.tokens_to_embeddings(prompt_answer_ids)
        slots = (prompt_answer_ids >= self.vocab_size) & (prompt_answer_ids < self.vocab_size_with_mem)
        prompt_answer_embs[slots] = torch.cat(memories).to(prompt_answer_embs)

        # Trainer 切到 eval() 计算验证损失时，仍使用训练时的冻结 Reader 和原词表。
        if self.model_args.train:
            decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, use_cache=False)
        else:
            with self.icae.disable_adapter():
                decoder_outputs = self.icae(inputs_embeds=prompt_answer_embs, use_cache=False)


        logits = decoder_outputs.logits
        effective_logits = logits[:,:-1,:].reshape(-1, logits.size(-1))
        target_ids = labels[:,1:].reshape(-1)
        loss = self.loss_fct(effective_logits, target_ids)
        return {"loss": loss, "logits": logits}
    
    
    def tokens_to_embeddings(self, token_ids):   # input_tokens can be either normal tokens and special tokens
        embeddings = self.icae.get_base_model().model.embed_tokens(token_ids)
        special_flags = token_ids >= self.vocab_size
        embeddings[special_flags] = self.memory_token_embed(token_ids[special_flags] - self.vocab_size).to(embeddings)    # replace special token's embedding from self.memory_token_embed
        return embeddings
        
    
    def _compress(
        self,
        input_ids: torch.LongTensor = None
    ):  # for inference; compress a fixed length of input into memory slots

        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)
        
        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((max_compressed_length, self.dim))
        
        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_input_ids = torch.cat([segment_input_ids, self.append_sequence], dim=1)
            mem_flag = segment_input_ids >= self.vocab_size

            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            segment_input_embedding[mem_flag] = self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding)

            # compress the current segment
            segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True)
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]

            # collect memory tokens
            compress_outputs[segment_idx*self.mem_size: self.mem_size*(segment_idx+1)] = segment_compress_outputs[mem_flag]
            
            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()
        
        return compress_outputs
