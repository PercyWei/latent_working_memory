"""并发提交裁判请求，并按原题序保存已完成的判定。"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
import json

from benchmarks.io import write_jsonl
from benchmarks.progress import QuestionTimer


DEFAULT_CONCURRENCY = 8


def load_judgments(path, question_ids):
    """读取原题序中的已评分子集，按题目位置索引。"""
    rows = {}
    if path.exists():
        positions = {question_id: index for index, question_id in enumerate(question_ids)}
        previous_index = -1
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                index = positions.get(row["question_id"], -1)
                if index <= previous_index:
                    raise ValueError(f"{path}:{line_number}: unknown, duplicate or out-of-order question ID")
                rows[index] = row
                previous_index = index
    return rows


def score_pending(total, rows, path, benchmark, concurrency, judge_row):
    """填补缺失判定。有请求失败时，保存其余在途请求的成功结果后报错。"""
    timer = QuestionTimer(len(rows))
    pending = (index for index in range(total) if index not in rows)
    failure = None
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        in_flight = {}
        while True:
            if failure is None:
                for index in islice(pending, concurrency - len(in_flight)):
                    in_flight[executor.submit(judge_row, index)] = index
            if not in_flight:
                break
            completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                index = in_flight.pop(future)
                try:
                    row = future.result()
                except Exception as error:
                    if failure is None:
                        failure = error
                    continue
                rows[index] = row
                write_jsonl(path, (rows[i] for i in sorted(rows)))
                print(f"{benchmark} | judged {len(rows)}/{total} | {timer.format_rate(len(rows))}",
                      flush=True)
    if failure is not None:
        raise failure
