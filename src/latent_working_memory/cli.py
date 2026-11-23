"""训练与独立评估共用的命令行参数类型。"""

import argparse


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def length_batch_size(value):
    """解析 RULER 的 LENGTH=BATCH 每卡批量覆盖。"""
    try:
        length, batch_size = map(int, value.split("="))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use LENGTH=BATCH with positive integers") from error
    if length < 1 or batch_size < 1:
        raise argparse.ArgumentTypeError("use LENGTH=BATCH with positive integers")
    return length, batch_size
