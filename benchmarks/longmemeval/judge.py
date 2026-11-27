"""使用 LongMemEval 官方提示词判分，并按官方口径汇总准确率。"""

from statistics import mean

from benchmarks.chat_api import chat_completion
from benchmarks.longmemeval.data import QUESTION_TYPES
from benchmarks.vendor.longmemeval import get_anscheck_prompt


JUDGE_MODEL = "gpt-6-luna"
JUDGE_OPTIONS = {"reasoning_effort": "none", "temperature": 0, "max_completion_tokens": 10}
JUDGE_REVISION = "9e0b455f4ef0e2ab8f2e582289761153549043fc"


def judge_answer(sample, hypothesis):
    """返回作者 evaluate_qa.py 的 autoeval_label，仅发送题目、参考答案和回答。"""
    prompt = get_anscheck_prompt(
        sample.question_type, sample.input.question, sample.answer, hypothesis,
        abstention="_abs" in sample.id,
    )
    response = chat_completion(JUDGE_MODEL, [{"role": "user", "content": prompt}], **JUDGE_OPTIONS)
    choice = response["choices"][0]
    content = (choice["message"]["content"] or "").strip()
    if choice["finish_reason"] != "stop" or not content:
        raise ValueError(f"{sample.id}: incomplete judge reply (finish_reason={choice['finish_reason']})")
    return {"model": JUDGE_MODEL, "label": "yes" in content.lower()}


def summarize_scores(records):
    """按官方 print_qa_metrics.py 汇总；没有样本的统计项表示为 null。"""
    accuracies = {question_type: [] for question_type in QUESTION_TYPES}
    abstention = []
    for record in records:
        judgment = record["autoeval_label"]
        if judgment["model"] != JUDGE_MODEL or not isinstance(judgment["label"], bool):
            raise ValueError(f"{record['question_id']}: expected a boolean {JUDGE_MODEL} judgment")
        score = int(judgment["label"])
        accuracies[record["metadata"]["question_type"]].append(score)
        if "_abs" in record["question_id"]:
            abstention.append(score)
    groups = [
        {"question_type": task, "samples": len(values), "score": mean(values) if values else None}
        for task, values in accuracies.items()
    ]
    return {
        "metric": "accuracy",
        "score": sum(sum(values) for values in accuracies.values()) / len(records) if records else None,
        "samples": len(records),
        "groups": groups,
        "task_averaged_accuracy": mean(g["score"] for g in groups) if all(accuracies.values()) else None,
        "abstention": {"samples": len(abstention), "score": mean(abstention) if abstention else None},
    }
