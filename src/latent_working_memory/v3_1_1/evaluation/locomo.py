"""执行独立 LoCoMo 评估，逐 checkpoint 保存本地结果。"""

import argparse
from collections import defaultdict
import sys

from latent_working_memory.benchmarks.data import load_samples
from latent_working_memory.benchmarks.protocol import data_identity, protocol
from latent_working_memory.benchmarks.scoring import score_sample, summarize_scores
from latent_working_memory.v4_1.data import EpisodeEncoder
from ..cli import ignore_resume_options, positive_int
from . import DEFAULT_EVAL_BATCH_SIZE
from .inference import EvaluationInput, EvaluationQuestion, compute_batch as compute_model_batch
from .run import run_evaluations


NAME = "locomo"
QA_INSTRUCTION = (
    "\nBased on the above conversations, write a short answer for the following "
    "question in a few words. Do not write complete and lengthy sentences. "
    "Answer with exact words from the conversations whenever possible.\n\n"
)


def plan_options(args, saved):
    return {
        "data": getattr(args, "data", None),
        "max_samples": None,
        "max_new_tokens": None,
        "lengths": None,
    }


def execution_settings(args, settings):
    return ({"eval_batch_size_per_gpu": args.batch_size_per_gpu},
            ["--batch-size-per-gpu", str(args.batch_size_per_gpu)])


def load_data(config, tokenizer, model_window, data, max_samples, max_new_tokens, lengths):
    if max_new_tokens is not None:
        raise ValueError("max_new_tokens applies only to test evaluation")
    if lengths is not None:
        raise ValueError("lengths applies only to RULER")
    samples = load_samples(NAME, data, ())
    if max_samples is not None:
        samples = samples[:max_samples]
    identity = data_identity(samples, data, NAME)
    evaluation_protocol = protocol(
        NAME, (), "immutable compressed embedding blocks; each block has fixed K slots",
    )
    evaluation_protocol.update({
        "input": "complete body; separately encoded task instructions and question; no truncation or training length filtering",
        "write_boundary": (
            "complete body is compressed; task instructions and query are passed to the reader; "
            "forced final pending flush; no raw tail or writes during generation"
        ),
        "selection": "first max_samples overall",
        "max_samples": max_samples,
        "memory_sharing": "compress each conversation once per attempt; answer and commit each pending question independently",
    })
    return samples, identity, evaluation_protocol


def sample_batches(samples, execution):
    conversations = defaultdict(list)
    for sample in samples:
        conversations[sample.metadata["conversation_id"]].append(sample)
    ordered = sorted(conversations.values(),
                     key=lambda questions: len(questions[0].prompt.rsplit(QA_INSTRUCTION, 1)[0]))
    batch_size = execution["eval_batch_size_per_gpu"]
    for start in range(0, len(ordered), batch_size):
        yield ordered[start:start + batch_size]


def compute_batch(conversations, request, model, config, max_new_tokens, commit):
    by_first_id = {questions[0].id: questions for questions in conversations}
    encoder = EpisodeEncoder(model.tokenizer)

    def encode(sample):
        questions = by_first_id[sample.id]
        context = sample.prompt.rsplit(QA_INSTRUCTION, 1)[0]
        return EvaluationInput(
            encoder.context(context),
            tuple(EvaluationQuestion(
                tuple(model.tokenizer.encode(
                    f"\n\n{QA_INSTRUCTION}{question.prompt.rsplit(QA_INSTRUCTION, 1)[1]}{encoder.suffix}",
                    add_special_tokens=False,
                )), question.max_new_tokens,
            ) for question in questions),
        )

    return compute_model_batch(
        [questions[0] for questions in conversations], model, config, commit, encode,
        lambda sample, answers: make_record(by_first_id[sample.id][len(answers) - 1], [answers[-1]]),
        commit_per_question=True,
    )


def make_record(sample, answers):
    answer = answers[0]
    return {
        "id": sample.id,
        "question": sample.metadata["question"],
        "references": list(sample.references), "metadata": sample.metadata,
        **answer, "score": score_sample(NAME, sample, answer["prediction"]),
    }


def summarize(records):
    return summarize_scores(NAME, records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--eval-run", required=True,
                        help="运行根目录，包含 train/、plan/ 与 eval/")
    parser.add_argument("--eval-id", default=None, help="评估计划标识，默认 YYYYMMDD-HHMMSS")
    parser.add_argument("--eval-data", help="LoCoMo 数据目录，新建或覆盖计划时必填")
    parser.add_argument("--checkpoint", nargs="+", default=None,
                        help="global_step_<step> 名称；新建时省略选全部，续跑时省略沿用计划")
    parser.add_argument("--batch-size-per-gpu", type=positive_int,
                        default=DEFAULT_EVAL_BATCH_SIZE, help="每卡评估 batch，默认 1")
    parser.add_argument("--output-root", default=None,
                        help="产物父目录；默认在原 run 下保存评估结果")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true", default=False,
                         help="恢复指定 eval-id 下已有的 LoCoMo 计划")
    restart.add_argument("--overwrite", action="store_true", default=False,
                         help="清理当前 LoCoMo 的评估结果和计划后重新评估")
    args = parser.parse_args(argv)
    if args.resume:
        ignore_resume_options(args, {
            "resume", "overwrite", "checkpoint", "eval_run", "eval_id", "output_root",
            "batch_size_per_gpu",
        })
        if not args.eval_id:
            parser.error("--resume requires an explicit --eval-id")
    if hasattr(args, "eval_data"):
        args.data = vars(args).pop("eval_data")
    return run_evaluations(args, sys.modules[__name__])


if __name__ == "__main__":
    main()
