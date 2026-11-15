"""内部单卡 worker 入口；任务模块提供数据和计算规则。"""

import argparse

from ..cli import positive_int
from . import DEFAULT_EVAL_BATCH_SIZE, locomo, ruler, test
from .evaluate import evaluate_shard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", choices=("test", "locomo", "ruler"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=positive_int)
    parser.add_argument("--max-samples", type=positive_int)
    parser.add_argument("--lengths", nargs="+", type=positive_int)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--worker-rank", type=int, required=True)
    parser.add_argument("--batch-size-per-gpu", type=positive_int,
                        default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--batch-size-by-length", nargs="+", type=ruler.length_batch_size, default=[])
    args = parser.parse_args(argv)
    if args.task == "ruler":
        task_module = ruler
    elif args.task == "locomo":
        task_module = locomo
    else:
        task_module = test
    execution = {"eval_batch_size_per_gpu": args.batch_size_per_gpu}
    if args.task == "ruler":
        execution["batch_size_by_length"] = dict(args.batch_size_by_length)
    evaluate_shard(
        args.checkpoint, args.data, args.output_dir, args.worker_rank, task_module, execution,
        max_new_tokens=args.max_new_tokens, max_samples=args.max_samples,
        lengths=args.lengths, eval_id=args.eval_id,
    )


if __name__ == "__main__":
    main()
