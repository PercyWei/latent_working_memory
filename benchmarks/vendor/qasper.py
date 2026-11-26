# Adapted from the Allen Institute for AI QASPER v0.3 evaluator.
# https://github.com/allenai/qasper-led-baseline/blob/afd0fb96bf78ce8cd8157639c6f6a6995e4f9089/scripts/evaluator.py
# Apache License 2.0. See QASPER-LICENSE.txt.
"""保留官方答案转换、SQuAD token F1 和多标注选取规则，供逐题评估调用。"""

from collections import Counter
import re
import string


EVALUATOR_REVISION = "afd0fb96bf78ce8cd8157639c6f6a6995e4f9089"
ANSWER_TYPES = ("extractive", "abstractive", "boolean", "none")


def answer_reference(answer_info):
    if answer_info["unanswerable"]:
        return {"answer": "Unanswerable", "type": "none"}
    if answer_info["extractive_spans"]:
        answer = ", ".join(answer_info["extractive_spans"])
        answer_type = "extractive"
    elif answer_info["free_form_answer"]:
        answer = answer_info["free_form_answer"]
        answer_type = "abstractive"
    elif answer_info["yes_no"] is not None:
        answer = "Yes" if answer_info["yes_no"] else "No"
        answer_type = "boolean"
    else:
        raise ValueError("QASPER annotation does not contain an answer")
    return {"answer": answer, "type": answer_type}


def normalize_answer(text):
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_answer(prediction, references):
    """以最高 F1 对应的人工标注确定题型；并列时沿用原标注顺序。"""
    scores = ((token_f1_score(prediction, ref["answer"]), ref["type"]) for ref in references)
    return max(scores, key=lambda item: item[0])
