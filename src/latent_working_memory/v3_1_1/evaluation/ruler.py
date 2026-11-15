"""执行独立 RULER 评估，支持按长度设置每卡 batch。"""

import argparse
from collections import Counter
from itertools import groupby
import sys

from latent_working_memory.benchmarks.data import load_samples
from latent_working_memory.benchmarks.protocol import protocol
from latent_working_memory.benchmarks.scoring import score_sample, summarize_scores
from ..cli import ignore_resume_options, positive_int
from . import DEFAULT_EVAL_BATCH_SIZE
from .inference import compute_batch as compute_model_batch, encode_context_question
from .records import normalize_lengths, ruler_data_identity
from .run import run_evaluations


NAME = "ruler"


# NVIDIA/RULER c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
# scripts/data/synthetic/constants.py；任务模板沿用上游 Apache-2.0 许可。
_VT_PREFIX = "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
_CWE_PREFIX = (
    "Below is a numbered list of words. In these words, some appear more often than others. "
    "Memorize the ones that appear most often.\n"
)
_FWE_PREFIX = (
    "Read the following coded text and track the frequency of each coded word. "
    "Find the three most frequently appeared coded words. "
)
_QA_INSTRUCTION = (
    "Answer the question based on the given documents. Only give me the answer "
    "and do not output any other words.\n\n"
)


def split_ruler_prompt(sample):
    """保留原始任务指令、查询及 few-shot 示例，只将主正文交给 Writer。"""
    task, prompt = sample.metadata["task"], sample.prompt
    few_shot = task in {"vt", "cwe"}
    if task.startswith("niah_"):
        plural = task in {"niah_multivalue", "niah_multiquery"}
        needle = "uuid" if task in {"niah_single_3", "niah_multikey_3"} else "number"
        if plural:
            needle += "s"
        prefix = (
            f"{'Some' if plural else 'A'} special magic {needle} "
            f"{'are' if plural else 'is'} hidden within the following text. "
            f"Make sure to memorize it. I will quiz you about the {needle} afterwards.\n"
        )
        separator = f"\nWhat {'are all' if plural else 'is'} the special magic {needle} for "
    elif task == "vt":
        prefix, separator = _VT_PREFIX, "\nQuestion: Find all variables that are assigned the value "
    elif task == "cwe":
        prefix, separator = _CWE_PREFIX, "\nQuestion: What are the 10 most common words in the above list?"
    elif task == "fwe":
        prefix, separator = _FWE_PREFIX, "\nQuestion: Do not provide any explanation. Please ignore the dots '....'. "
    elif task in {"qa_1", "qa_2"}:
        prefix = _QA_INSTRUCTION + "The following are given documents.\n\n"
        separator = "\n\n" + _QA_INSTRUCTION + "Question: "
    else:
        raise ValueError(f"unsupported RULER task: {task}")

    # VT/CWE 的主样本位于示例之后；其余任务从固定前缀开始。
    start = prompt.rfind(prefix) if few_shot else 0
    if start < 0 or not prompt.startswith(prefix, start):
        raise ValueError(f"{sample.id}: input does not match the RULER {task} template")
    start += len(prefix)
    context, marker, question = prompt[start:].rpartition(separator)
    if not marker or not context:
        raise ValueError(f"{sample.id}: missing body/query boundary in the RULER {task} template")
    return context, prompt[:start] + separator + question


def length_batch_size(value):
    """解析 RULER 的 LENGTH=BATCH 每卡批量覆盖。"""
    try:
        length, batch_size = map(int, value.split("="))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use LENGTH=BATCH with positive integers") from error
    if length < 1 or batch_size < 1:
        raise argparse.ArgumentTypeError("use LENGTH=BATCH with positive integers")
    return length, batch_size


def plan_options(args, saved):
    lengths = (saved["lengths"] if saved is not None
               else normalize_lengths(getattr(args, "lengths", None)))
    unplanned = set(args.batch_size_by_length) - set(lengths)
    if unplanned:
        raise ValueError(
            "--batch-size-by-length contains lengths outside the evaluation plan: "
            + ", ".join(map(str, sorted(unplanned)))
        )
    return {
        "data": getattr(args, "data", None),
        "max_samples": None,
        "max_new_tokens": None,
        "lengths": list(lengths),
    }


def execution_settings(args, settings):
    mapping = args.batch_size_by_length
    options = {"eval_batch_size_per_gpu": args.batch_size_per_gpu,
               "batch_size_by_length": mapping}
    arguments = ["--batch-size-per-gpu", str(args.batch_size_per_gpu),
                 "--lengths", *map(str, settings["lengths"])]
    if mapping:
        arguments.extend((
            "--batch-size-by-length",
            *(f"{length}={batch}" for length, batch in sorted(mapping.items())),
        ))
    return options, arguments


def load_data(config, tokenizer, model_window, data, max_samples, max_new_tokens, lengths):
    if max_new_tokens is not None:
        raise ValueError("max_new_tokens applies only to test evaluation")
    lengths = normalize_lengths(lengths)
    samples = load_samples(NAME, data, tuple(lengths))
    if max_samples is not None:
        counts, selected = Counter(), []
        for sample in samples:
            group = (sample.metadata["length"], sample.metadata["task"])
            if counts[group] < max_samples:
                selected.append(sample)
                counts[group] += 1
        samples = tuple(selected)
    identity = ruler_data_identity(samples, data, lengths)
    evaluation_protocol = protocol(
        NAME, lengths, "immutable compressed embedding blocks; each block has fixed K slots",
    )
    evaluation_protocol.update({
        "input": "complete body; separately encoded task instructions and question; no truncation or training length filtering",
        "write_boundary": (
            "complete body is compressed; task instructions and query are passed to the reader; "
            "forced final pending flush; no raw tail or writes during generation"
        ),
        "selection": "first max_samples per task/length",
        "max_samples": max_samples,
    })
    return samples, identity, evaluation_protocol


def sample_batches(samples, execution):
    ordered = sorted(samples, key=lambda sample: (sample.metadata["length"], len(sample.prompt)))
    groups = groupby(ordered, key=lambda sample: sample.metadata["length"])
    overrides = execution["batch_size_by_length"]
    for length, group in groups:
        limit = overrides.get(length, execution["eval_batch_size_per_gpu"])
        batch = []
        for sample in group:
            batch.append(sample)
            if len(batch) == limit:
                yield batch
                batch = []
        if batch:
            yield batch


def compute_batch(samples, request, model, config, max_new_tokens, commit):
    return compute_model_batch(
        samples, model, config, commit,
        lambda sample: encode_context_question(
            *split_ruler_prompt(sample), model.tokenizer, sample.max_new_tokens,
        ),
        make_record,
    )


def make_record(sample, answers):
    answer = answers[0]
    return {
        "question": sample.metadata.get("question", ""),
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
    parser.add_argument("--eval-data", help="RULER 数据目录，新建或覆盖计划时必填")
    parser.add_argument("--checkpoint", nargs="+", default=None,
                        help="global_step_<step> 名称；新建时省略选全部，续跑时省略沿用计划")
    parser.add_argument("--batch-size-per-gpu", type=positive_int,
                        default=DEFAULT_EVAL_BATCH_SIZE, help="未单独指定长度的每卡 batch，默认 1")
    parser.add_argument("--batch-size-by-length", nargs="+", type=length_batch_size, default=[],
                        metavar="LENGTH=BATCH", help="按 RULER 长度覆盖每卡 batch；续跑时可修改")
    parser.add_argument("--lengths", nargs="+", type=positive_int,
                        help="评估的 RULER 长度档位；续跑沿用计划")
    parser.add_argument("--output-root", default=None,
                        help="产物父目录；默认在原 run 下保存评估结果")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true", default=False,
                         help="恢复指定 eval-id 下已有的 RULER 计划")
    restart.add_argument("--overwrite", action="store_true", default=False,
                         help="清理当前 RULER 的评估结果和计划后重新评估")
    args = parser.parse_args(argv)
    pairs = args.batch_size_by_length
    if len({length for length, _ in pairs}) != len(pairs):
        parser.error("--batch-size-by-length must not repeat a length")
    args.batch_size_by_length = dict(pairs)
    if args.resume:
        ignore_resume_options(args, {
            "resume", "overwrite", "checkpoint", "eval_run", "eval_id", "output_root",
            "batch_size_per_gpu", "batch_size_by_length",
        })
        if not args.eval_id:
            parser.error("--resume requires an explicit --eval-id")
    if hasattr(args, "eval_data"):
        args.data = vars(args).pop("eval_data")
    return run_evaluations(args, sys.modules[__name__])


if __name__ == "__main__":
    main()
