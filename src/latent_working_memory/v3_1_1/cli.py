"""公共参数转换与恢复选项处理。"""

import argparse
import math
import sys

from latent_working_memory.cli import positive_int


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def optional_positive_int(value):
    return None if value == "null" else positive_int(value)


def boolean(value):
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("must be true or false")
    return value == "true"


def unit_interval(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be finite and between 0 and 1")
    return number


def save_frequency(value):
    return value if value == "after_each_epoch" else positive_int(value)


def ignore_resume_options(args, retained):
    """删除恢复模式不使用的显式选项，并集中提示。"""
    ignored = sorted(set(vars(args)) - retained)
    for name in ignored:
        delattr(args, name)
    if ignored:
        options = ", ".join("--" + name.replace("_", "-") for name in ignored)
        print(f"[resume] ignored options; using the saved plan: {options}", file=sys.stderr)
