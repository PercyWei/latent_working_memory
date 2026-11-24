"""ICAE v2 接入公共 LoCoMo 会话评估。"""

from pathlib import Path
import torch

from latent_working_memory.benchmarks.data import locomo_question_prompt
from latent_working_memory.benchmarks.locomo.engine import PreparedConversation, PreparedQuestion
from latent_working_memory.benchmarks.locomo.reader import greedy_read
from latent_working_memory.benchmarks.locomo.records import file_identity
from .model import CHECKPOINT_FILES, UPSTREAM_COMMIT, load_adapter as load_official


class ICAEAdapter:
    def __init__(self, adapter):
        self.adapter = adapter

    def prepare_context(self, history, questions):
        state, stats = self.adapter.encode_history(history)
        model = self.adapter.model
        prepared = []
        for sample in questions:
            prompt = locomo_question_prompt(sample.metadata["question"], sample.metadata["category"])
            ids = model.tokenizer.encode(prompt, add_special_tokens=False)
            if self.adapter.stage == "finetuned":
                ids = [model.ft_token_id] + ids
            ids += [733, 28748, 16289, 28793]
            if state[1] + len(ids) + sample.max_new_tokens > self.adapter.window:
                raise ValueError(f"ICAE reader exceeds position window: {sample.id}")
            prepared.append(PreparedQuestion(sample, ids, (len(ids), sample.max_new_tokens)))
        return PreparedConversation(state, prepared, stats)

    @torch.no_grad()
    def answer_batch(self, state, questions, complete):
        prefix, _ = state
        size = len(questions)
        # 官方 Transformers 4.40 的 tuple cache 追加时分配新张量。
        cache = tuple(tuple(tensor.expand(size, -1, -1, -1) for tensor in layer) for layer in prefix)
        model = self.adapter.model
        ids = torch.tensor([question.inputs for question in questions], device=self.adapter.device)
        embeddings = model.tokens_to_embeddings(ids)
        def completed(index, result):
            tokens = result["generated_token_ids"]
            result.update(prediction=model.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                          generated_tokens=len(tokens), decode_batch_size=size)
            complete(questions[index], result)
        with model.icae.disable_adapter():
            greedy_read(self.adapter.decoder, cache, embeddings,
                        [q.sample.max_new_tokens for q in questions], model.eos_id,
                        completed, vocab_size=model.vocab_size - 1)


def add_arguments(parser):
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", choices=CHECKPOINT_FILES, required=True)


def configuration(args):
    checkpoint = args.checkpoint.expanduser().absolute()
    if checkpoint.name != CHECKPOINT_FILES[args.stage]:
        raise ValueError(f"--stage {args.stage} requires {CHECKPOINT_FILES[args.stage]}")
    return {"model_path": str(args.model_path.expanduser().resolve()), "checkpoint": str(checkpoint),
            "stage": args.stage, "upstream_commit": UPSTREAM_COMMIT,
            "model_files": file_identity(args.model_path), "checkpoint_files": file_identity(checkpoint)}


def load_adapter(settings):
    return ICAEAdapter(load_official(Path(settings["model_path"]), Path(settings["checkpoint"]), settings["stage"]))
