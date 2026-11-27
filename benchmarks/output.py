"""评估结果目录与对应的命令行参数。"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def add_output_arguments(parser, output_root, checkpoint_layout=False):
    parser.add_argument("--output-root", type=Path, default=Path(output_root))
    if checkpoint_layout:
        parser.add_argument("--output-id", help="模型或 checkpoint 下的结果目录名，默认由评估设置和任务名称确定")
    else:
        parser.add_argument("--output-id", help="运行 ID，默认使用方法名称及启动时间")


def evaluation_directory(args, task, output_name, checkpoint_name=None):
    root = args.output_root.expanduser().resolve() / "eval"
    if args.output_id is None:
        if checkpoint_name is not None:
            task_name = task.replace("/", "_")
            args.output_id = f"{output_name}_{task_name}" if output_name else task_name
        else:
            if args.resume:
                raise ValueError("--resume requires the original --output-id")
            timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M")
            args.output_id = f"{output_name}_{timestamp}"
    if not args.output_id.strip() or args.output_id in {".", ".."} or any(
        character in args.output_id for character in ("/", "\\")
    ):
        raise ValueError("--output-id must be a single nonempty directory name")
    if checkpoint_name is not None:
        return root / checkpoint_name / args.output_id
    return root / args.output_id / task
