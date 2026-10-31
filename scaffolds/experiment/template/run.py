"""四种模式共用的控制入口；计算进程只接受已保存计划。"""

from functools import partial
import json
from pathlib import Path
import signal
import sys

from .cli import config_overrides, parse_run_args
from .config import load_config
from .evaluation.run import evaluate_checkpoint
from .execution.lifecycle import evaluate_checkpoints, resume_run, start_run
from .execution.processes import run_workers
from .execution.runtime import worker_count
from .method import method_name


def launch_training(plan_dir, checkpoint, device, workers):
    command = [sys.executable]
    if workers > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={workers}",
            "--module",
        ]
    else:
        command += ["-m"]
    command += [f"{__package__}.training.train", "--plan-dir", str(plan_dir), "--device", device]
    if checkpoint is not None:
        command += ["--resume", str(checkpoint)]
    run_workers([command], Path(plan_dir).parent / "train/workers.log")


def main(argv=None):
    # 与 Ctrl-C 一样，SIGTERM 先经过生命周期记录和子进程清理。
    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    args = parse_run_args(argv)
    workers = worker_count(args.device, args.cpu_processes)
    train = partial(launch_training, device=args.device, workers=workers)
    evaluate = partial(evaluate_checkpoint, device=args.device, workers=workers)
    if args.resume:
        result = resume_run(args.checkpoint, workers, train, evaluate)
    elif args.evaluate_only:
        checkpoints = args.checkpoint
        if args.eval_run is not None:
            checkpoints = list(
                Path(args.eval_run)
                .expanduser()
                .resolve()
                .glob("train/checkpoints/global_step_*/state.pt")
            )
        names = {
            "test_data": "data",
            "max_test_samples": "max_samples",
            "max_new_tokens": "max_new_tokens",
        }
        options = {
            field: getattr(args, name) for name, field in names.items() if name in args._supplied
        }
        result = evaluate_checkpoints(
            checkpoints, getattr(args, "eval_id", None), "test", options, evaluate
        )
    else:
        config = load_config(getattr(args, "config", None), config_overrides(args))
        result = start_run(config, method_name(config), workers, train, evaluate)
    print(
        json.dumps(
            result if isinstance(result, dict) else {"evaluation_dir": str(result)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
