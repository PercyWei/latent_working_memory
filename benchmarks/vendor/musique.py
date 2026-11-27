# Adapted from Stony Brook NLP's MuSiQue v1.0 answer evaluator.
# https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/answer.py
# CC BY 4.0. See MUSIQUE-LICENSE.txt.
"""保留官方归一化、空答案及多参考答案规则，供逐题 F1 / EM 评估调用。"""

from collections import Counter
import re
import string


EVALUATOR_REVISION = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"


def normalize_answer(text):
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if not prediction_tokens or not ground_truth_tokens:
        return int(prediction_tokens == ground_truth_tokens)
    if num_same == 0:
        return 0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_answer(prediction, references):
    """F1 和 EM 分别在主答案及全部 aliases 中取最大值。"""
    f1 = max(token_f1_score(prediction, reference) for reference in references)
    normalized_prediction = normalize_answer(prediction)
    em = max(int(normalized_prediction == normalize_answer(reference)) for reference in references)
    return f1, em
