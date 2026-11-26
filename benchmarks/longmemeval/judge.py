"""LongMemEval 官方问答裁判及准确率汇总。"""

from functools import cache
import json
import os
from pathlib import Path
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import dotenv_values
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from benchmarks.longmemeval.data import QUESTION_TYPES
from benchmarks.vendor.longmemeval import get_anscheck_prompt


JUDGE_MODEL = "gpt-4o-2024-08-06"
JUDGE_REVISION = "9e0b455f4ef0e2ab8f2e582289761153549043fc"


@cache
def _project_credentials():
    """首次判题时读取项目根目录的 .env，同一进程的后续题目复用配置。"""
    values = dotenv_values(Path(__file__).resolve().parents[2] / ".env", interpolate=False)
    return values.get("OPENAI_API_KEY"), values.get("OPENAI_BASE_URL")


def _transient_error(error):
    if isinstance(error, HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(error, (URLError, TimeoutError, ConnectionError))


@retry(retry=retry_if_exception(_transient_error), wait=wait_random_exponential(min=1, max=30),
       stop=stop_after_attempt(5), reraise=True)
def judge_answer(sample, hypothesis):
    """返回作者 evaluate_qa.py 的 autoeval_label，仅发送题目、参考答案和回答。"""
    file_key, file_base_url = _project_credentials()
    key = os.environ.get("OPENAI_API_KEY", file_key)
    if not key:
        raise ValueError("Set OPENAI_API_KEY in the environment or the project root .env")
    prompt = get_anscheck_prompt(
        sample.question_type, sample.input.question, sample.answer, hypothesis,
        abstention="_abs" in sample.id,
    )
    body = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "n": 1,
        "temperature": 0,
        "max_tokens": 10,
    }
    base = os.environ.get("OPENAI_BASE_URL", file_base_url) or "https://api.openai.com/v1"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if organization := os.environ.get("OPENAI_ORGANIZATION"):
        headers["OpenAI-Organization"] = organization
    url = f"{base.rstrip('/')}/chat/completions"
    request = Request(url, data=json.dumps(body).encode(), headers=headers)
    with urlopen(request, timeout=90) as response:
        content = json.load(response)["choices"][0]["message"]["content"]
    return {"model": JUDGE_MODEL, "label": "yes" in content.strip().lower()}


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
