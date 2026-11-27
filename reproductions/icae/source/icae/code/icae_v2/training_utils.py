import math
import os
import random
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import SAFE_WEIGHTS_NAME

from reproductions.icae.formatting import icae_input_format
from .modeling_icae_multi_span import segment_count


class ICAETrainer(Trainer):
    """沿用 Trainer 的优化器及随机数状态断点，模型文件仅保存可训练参数。"""

    def _save(self, output_dir=None, state_dict=None):
        # Trainer 只在负责保存的进程调用 _save，参数名取自原始 ICAE 模型。
        parameters = self.model.adapter_parameters()
        if state_dict is None:
            state_dict = parameters
        weights = {name: state_dict[name].detach().cpu().contiguous() for name in parameters}
        directory = Path(output_dir or self.args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / SAFE_WEIGHTS_NAME
        temporary = target.with_suffix(".safetensors.tmp")
        save_file(weights, str(temporary), metadata={"format": "pt"})
        temporary.replace(target)
        torch.save(self.args, directory / TRAINING_ARGS_NAME)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # 基座与冻结的解码器已经从 model_name_or_path 初始化。
        target = self.model if model is None else self.accelerator.unwrap_model(model)
        target.load_adapter_weights(Path(resume_from_checkpoint) / SAFE_WEIGHTS_NAME)

    def _load_best_model(self):
        self._load_from_checkpoint(self.state.best_model_checkpoint)


def train_model(model, train_dataset, eval_dataset, training_args, data_collator=None):

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    if max(training_args.per_device_train_batch_size, training_args.per_device_eval_batch_size) == 1:
        data_collator = None
        
    # print training_args at local_rank 0
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(training_args)

    trainer = ICAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    checkpoint = None
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    print(f"Loaded from the checkpoint: {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def text_extraction(input_ids, length, lm_ratio=0.0):
    
    input_len = len(input_ids)
    assert input_len >= 1, f"Error: invalid input length ({input_len})"
    
    # ae
    if random.random() >= lm_ratio: 
        if input_len <= length: # if shorter, keep the complete text
            return input_ids, []
        else:
            last_start = input_len - length
            random_start = random.randint(0, last_start)
            return input_ids[random_start: random_start+length], []
    
    # lm    
    if input_len <= length:
        r = random.randint(0, input_len-1)
        return input_ids[:r+1], input_ids[r+1:]
    else:
        last_start = input_len - length
        random_start = random.randint(0, last_start)
        return input_ids[random_start: random_start+length], input_ids[random_start+length:]


class ICAETrainingInputs:
    """逐样本构建 Writer 输入和只对答案计算损失的 Reader 输入。仅保留分词器及配置。"""

    def __init__(self, model, stage):
        args = model.training_args
        self.tokenizer = model.tokenizer
        self.window = model.icae.config.max_position_embeddings
        self.max_length = args.model_max_length
        self.mem_size = model.mem_size
        self.compression_rate = model.mean_compression_rate
        self.first_mem_id = model.vocab_size
        self.ae_id = model.ae_token_id
        self.lm_id = model.lm_token_id
        self.eos_id = model.eos_id
        if self.eos_id is None:
            raise ValueError("ICAE training requires a tokenizer EOS token")
        self.min_tokens_for_lm = args.min_tokens_for_lm
        self.leave_tokens_for_lm = args.leave_tokens_for_lm
        self.add_lm_token = args.add_special_token_for_lm
        if stage == "finetuned":
            self.template = icae_input_format(self.tokenizer, stage, model.ft_token_id, system_prompt=None)

    def memory_slots(self, document):
        count = segment_count(len(document), self.mem_size, self.compression_rate)
        writer_length = math.ceil(len(document) / count) + self.mem_size
        if writer_length > self.window:
            raise ValueError(f"ICAE writer needs {writer_length} positions (window {self.window})")
        return [self.first_mem_id] * (count * self.mem_size)

    def append_example(self, output, document, prompt, answer, skip=0):
        reader_ids = prompt + answer
        if len(reader_ids) > self.window:
            raise ValueError(f"ICAE reader needs {len(reader_ids)} positions (window {self.window})")
        output["input_ids"].append(document)
        output["prompt_answer_ids"].append(reader_ids)
        output["labels"].append([-100] * (len(prompt) + skip) + answer[skip:])

    def pretrain(self, examples, lm_ratio=0.0, validation=False):
        output = {"input_ids": [], "prompt_answer_ids": [], "labels": []}
        texts = self.tokenizer(examples["text"], truncation=False, return_attention_mask=False)["input_ids"]
        for tokens in texts:
            if validation:
                document, continuation = tokens[:self.max_length], []
            else:
                document, continuation = text_extraction(tokens, self.max_length, lm_ratio)
            prompt = self.memory_slots(document)
            if len(continuation) > self.min_tokens_for_lm and self.max_length > self.leave_tokens_for_lm:
                if self.add_lm_token:
                    prompt.append(self.lm_id)
                self.append_example(output, document, prompt, continuation[:self.max_length], self.leave_tokens_for_lm)
            else:
                self.append_example(output, document, prompt + [self.ae_id], document + [self.eos_id])
        return output

    def finetune(self, examples):
        output = {"input_ids": [], "prompt_answer_ids": [], "labels": []}
        documents = self.tokenizer(examples["input"], truncation=True, max_length=self.max_length,
                                   return_attention_mask=False)["input_ids"]
        prompts = self.tokenizer(examples["prompt"], add_special_tokens=False, return_attention_mask=False)["input_ids"]
        answers = self.tokenizer(examples["answer"], add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for document, prompt, answer in zip(documents, prompts, answers, strict=True):
            reader_prompt = (self.template["reader_prefix_ids"] + self.memory_slots(document)
                             + self.template["question_prefix_ids"] + prompt + self.template["answer_prefix_ids"])
            self.append_example(output, document, reader_prompt, answer + [self.eos_id])
        return output
