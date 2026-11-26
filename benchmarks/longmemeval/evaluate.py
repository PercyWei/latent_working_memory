"""七种基线共用的 LongMemEval 评估入口，生成回答后调用项目根环境逐题评分。"""

from pathlib import Path
import subprocess

from benchmarks.parallel import resolve_generation_run
from latent_working_memory.prompts import QA_SYSTEM_PROMPT
from .data import LONGMEMEVAL_FILES, QUESTION_FORMAT, SESSION_HEADER, TURN_FORMAT
from .generate import run_generation


def add_evaluation_arguments(parser, output_root):
    root = Path(__file__).resolve().parents[2]
    # 要求完整参数名，避免把 --benchmark 解析为 --benchmark-data。
    parser.allow_abbrev = False
    parser.add_argument("--split", choices=LONGMEMEVAL_FILES, default="longmemeval_s_cleaned",
                        help="选择 LongMemEval 的 S 或 Oracle 设置，默认 longmemeval_s_cleaned")
    parser.add_argument("--benchmark-data", type=Path, default=root / "data/benchmarks/longmemeval")
    parser.add_argument("--output-root", type=Path, default=Path(output_root))
    parser.add_argument("--output-id", help="默认使用模型或 checkpoint 名称及启动时间")
    parser.add_argument("--batch-size-per-gpu", type=int, default=1, help="每卡批处理数，默认 1")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="每题的回答 token 上限")
    parser.add_argument("--resume", action="store_true",
                        help="沿用原输出 ID 和配置，补齐未完成的回答和评分")


def run_evaluation(args, method, settings, load_adapter, output_name):
    output_id, devices = resolve_generation_run(args, output_name)
    settings = {
        **settings, "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
        "system_prompt": QA_SYSTEM_PROMPT, "session_header": SESSION_HEADER,
        "turn_format": TURN_FORMAT, "question_format": QUESTION_FORMAT,
    }
    predictions = run_generation(
        args.split, args.benchmark_data, args.output_root, output_id, method, settings,
        load_adapter, args.batch_size_per_gpu, args.resume, devices=devices,
    )
    command = ["uv", "run", "--frozen", "python", "-m", "benchmarks.longmemeval.score",
               "--run-dir", str(predictions.parent)]
    if args.resume:
        command.append("--resume")
    print(f"{args.split} | scoring | {predictions.parent}", flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
    return predictions
