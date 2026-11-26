"""读取 LongMemEval 的原始 JSON，仅将日期、对话角色、正文和问题暴露给方法。"""

from dataclasses import dataclass
import json
from operator import attrgetter
from pathlib import Path


DATASET_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
LONGMEMEVAL_FILES = {
    "longmemeval_s_cleaned": "longmemeval_s_cleaned.json",
    "longmemeval_oracle": "longmemeval_oracle.json",
}
QUESTION_TYPES = (
    "single-session-user", "single-session-preference", "single-session-assistant",
    "multi-session", "temporal-reasoning", "knowledge-update",
)
SESSION_HEADER = "Session date: {date}"
TURN_FORMAT = "{role}: {content}"
QUESTION_FORMAT = "\n\nQuestion date: {question_date}\n\nQuestion: {question}"


@dataclass(frozen=True, slots=True)
class Turn:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class Session:
    date: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True, slots=True)
class LongMemEvalInput:
    sessions: tuple[Session, ...]
    question: str
    question_date: str


@dataclass(frozen=True, slots=True)
class LongMemEvalSample:
    id: str
    input: LongMemEvalInput
    question_type: str
    answer: str | int


def format_history(item: LongMemEvalInput) -> str:
    """按数据读取顺序渲染完整会话，仅包含日期、角色和原始正文。"""
    sessions = []
    for session in item.sessions:
        lines = [SESSION_HEADER.format(date=session.date)]
        lines.extend(TURN_FORMAT.format(role=turn.role, content=turn.content) for turn in session.turns)
        sessions.append("\n".join(lines))
    return "\n\n".join(sessions)


def format_question(item: LongMemEvalInput) -> str:
    """问题与提问日期只在读出时追加，开头的空行分隔最后一场会话。"""
    return QUESTION_FORMAT.format(question_date=item.question_date, question=item.question)


def _required_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: expected a nonempty string")
    return value


def _read_session(date, turns, location):
    _required_text(date, f"{location}.date")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{location}: expected a nonempty list of turns")
    dialogue = []
    for turn in turns:
        role, content = turn["role"], turn["content"]
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError(f"{location}: expected user or assistant role and text content")
        dialogue.append(Turn(role, content))
    return Session(date, tuple(dialogue))


def load_longmemeval(benchmark, data_dir):
    path = Path(data_dir) / LONGMEMEVAL_FILES[benchmark]
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty list of questions")
    samples, seen = [], set()
    for row in rows:
        qid = _required_text(row["question_id"], "question_id")
        if qid in seen:
            raise ValueError(f"{path}: duplicate question_id {qid}")
        seen.add(qid)
        kind, answer = row["question_type"], row["answer"]
        if kind not in QUESTION_TYPES or type(answer) not in (str, int):
            raise ValueError(f"{qid}: unsupported question_type or reference answer")
        dates, history = row["haystack_dates"], row["haystack_sessions"]
        if not isinstance(dates, list) or not isinstance(history, list) or not history:
            raise ValueError(f"{qid}: expected lists of dates and sessions")
        sessions = [
            _read_session(date, turns, f"{qid}.haystack_sessions[{i}]")
            for i, (date, turns) in enumerate(zip(dates, history, strict=True))
        ]
        # 官方时间戳 YYYY/MM/DD (Day) HH:MM 可按字典序排列；同一时刻保留原顺序。
        sessions.sort(key=attrgetter("date"))
        inputs = LongMemEvalInput(
            tuple(sessions), _required_text(row["question"], f"{qid}.question"),
            _required_text(row["question_date"], f"{qid}.question_date"),
        )
        samples.append(LongMemEvalSample(qid, inputs, kind, answer))
    return tuple(samples)
