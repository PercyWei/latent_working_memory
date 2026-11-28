"""为评估入口配置裁判开关，并在项目根环境启动评分。"""

from pathlib import Path
import subprocess


def add_judge_arguments(parser):
    parser.add_argument("--judge", action="store_true", default=True,
                        help="在保存官方指标后计算 Judge Accuracy，默认开启")
    parser.add_argument("--no-llm-as-judge", dest="judge", action="store_false",
                        help="只计算官方指标")


def run_answer_judge(directory, resume):
    root = Path(__file__).resolve().parents[1]
    command = ["uv", "run", "--frozen", "python", "-m", "benchmarks.judge_score",
               "--run-dir", str(directory)]
    if resume:
        command.append("--resume")
    print(f"Judge Accuracy | scoring | {directory}", flush=True)
    subprocess.run(command, cwd=root, check=True)
