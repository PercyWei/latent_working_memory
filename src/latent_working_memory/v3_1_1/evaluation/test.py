"""执行独立 test 评估，逐 checkpoint 保存本地结果。"""

import argparse
import math
import sys

from latent_working_memory.data_preparation.personamem.audit import scores
from ..cli import ignore_resume_options, optional_positive_int, positive_int
from ..data import load_evaluation_documents
from . import DEFAULT_EVAL_BATCH_SIZE
from .inference import EvaluationInput, EvaluationQuestion, compute_batch as compute_model_batch
from .run import run_evaluations


NAME = "test"


def plan_options(args, saved):
    return {
        "data": getattr(args, "data", None),
        "max_samples": getattr(args, "max_samples", None),
        "max_new_tokens": getattr(args, "max_new_tokens", 128),
        "lengths": None,
    }


def execution_settings(args, settings):
    options = {"eval_batch_size_per_gpu": args.batch_size_per_gpu}
    arguments = ["--batch-size-per-gpu", str(args.batch_size_per_gpu)]
    for name in ("max_samples", "max_new_tokens"):
        if settings[name] is not None:
            arguments.extend(("--" + name.replace("_", "-"), str(settings[name])))
    return options, arguments


def load_data(config, tokenizer, model_window, data, max_samples, max_new_tokens, lengths):
    if lengths is not None:
        raise ValueError("lengths applies only to RULER")
    documents, statistics = load_evaluation_documents(
        data, tokenizer, max_samples=max_samples, seed=config.seed,
    )
    evaluation_protocol = {
        "input": "complete document; separately encoded question; no writes during QA",
        "tail": "fixed recent is reserved before policy decisions; final pending below pending_size stays raw; pending then recent precede the reader question",
        "generation": "greedy; fresh generation config; all QA share one document memory",
        "generation_budget": max_new_tokens or 128,
        "max_samples": max_samples,
        "selection": "stable document hash sampling; no training length filtering",
        "nll": "QA-token mean, then QA mean per document, then document mean; assistant ending included",
        "scoring": "single-reference QA EM/token-overlap F1; normalized case/articles/punctuation",
    }
    return tuple(documents.values()), statistics, evaluation_protocol


def sample_batches(samples, execution):
    ordered = sorted(samples, key=lambda sample: len(sample.input_ids))
    batch_size = execution["eval_batch_size_per_gpu"]
    for start in range(0, len(ordered), batch_size):
        yield ordered[start:start + batch_size]


def make_record(sample, answers):
    for qa_index, (qa, answer) in enumerate(zip(sample.qas, answers, strict=True)):
        answer.update(qa_index=qa_index, question=qa.question, reference=qa.answer,
                      **scores(answer["prediction"], qa.answer))
        if qa.segment_id is not None:
            answer.update(segment_id=qa.segment_id, segment_char_span=list(qa.segment_char_span))
    return {"document_id": sample.document_id, "qas": answers}


def compute_batch(samples, request, model, config, max_new_tokens, commit):
    return compute_model_batch(
        samples, model, config, commit,
        lambda sample: EvaluationInput(sample.input_ids, tuple(
            EvaluationQuestion(qa.prompt_ids, max_new_tokens or 128, qa.target_ids, truncate_to_window=True)
            for qa in sample.qas
        )),
        make_record, samples[0].reader_prefix_ids,
    )


def summarize(records):
    answers = [answer for row in records for answer in row["qas"]]
    target_tokens = sum(row["target_tokens"] for row in answers)
    nll = sum(row["nll"] * row["target_tokens"] for row in answers) / target_tokens
    return {
        "documents": len(records), "questions": len(answers),
        "qa_nll": sum(sum(answer["nll"] for answer in row["qas"]) / len(row["qas"])
                      for row in records) / len(records),
        "token_nll": nll, "perplexity": math.exp(nll),
        "em": sum(row["em"] for row in answers) / len(answers),
        "f1": sum(row["f1"] for row in answers) / len(answers),
        "target_tokens": target_tokens,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--eval-run", required=True,
                        help="运行根目录，包含 train/、plan/ 与 eval/")
    parser.add_argument("--eval-id", default=None, help="评估计划标识，默认 YYYYMMDD-HHMMSS")
    parser.add_argument("--eval-data", help="test 数据目录，新建或覆盖计划时必填")
    parser.add_argument("--checkpoint", nargs="+", default=None,
                        help="global_step_<step> 名称；新建时省略选全部，续跑时省略沿用计划")
    parser.add_argument("--batch-size-per-gpu", type=positive_int,
                        default=DEFAULT_EVAL_BATCH_SIZE, help="每卡评估 batch，默认 1")
    parser.add_argument("--output-root", default=None,
                        help="产物父目录；默认在原 run 下保存评估结果")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true", default=False,
                         help="恢复指定 eval-id 下已有的 test 计划")
    restart.add_argument("--overwrite", action="store_true", default=False,
                         help="清理当前 test 的评估结果和计划后重新评估")
    parser.add_argument("--max-test-samples", type=optional_positive_int, help="null 取消样本数限制")
    parser.add_argument("--max-new-tokens", type=positive_int, help="生成 token 上限，新计划默认 128")
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
    if hasattr(args, "max_test_samples"):
        args.max_samples = vars(args).pop("max_test_samples")
    return run_evaluations(args, sys.modules[__name__])


if __name__ == "__main__":
    main()
