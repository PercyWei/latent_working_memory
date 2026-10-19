# Adapted from snap-research/locomo by Adyasha Maharana, Dong-Ho Lee,
# Sergey Tulyakov, Mohit Bansal, Francesco Barbieri and Yuwei Fang.
# Source: https://github.com/snap-research/locomo/blob/
# 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py
# Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
# The upstream license and disclaimer are retained in LOCOMO-LICENSE.txt.

"""LoCoMo 官方 QA F1 函数；仅去除未使用的检索、BERTScore 和文件评估代码。"""

from collections import Counter
import string

import numpy as np
from nltk.stem import PorterStemmer
import regex


ps = PorterStemmer()


def normalize_answer(s):
    s = s.replace(",", "")

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return np.mean(
        [max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths]
    )
