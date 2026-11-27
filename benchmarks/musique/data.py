"""读取 MuSiQue-Ans v1.0 dev，按原顺序保留全部候选段落。"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


SPLIT_FILES = {"dev": "musique_ans_v1.0_dev.jsonl"}
QUESTION_FORMAT = "\n\nQuestion: {question}"
SYSTEM_PROMPT = (
    "Answer the question using the provided paragraphs. Give only a short answer."
)


@dataclass(frozen=True, slots=True)
class MusiqueSample:
    id: str
    history: str
    question: str
    references: tuple[str, ...]
    hops: int


def load_musique(split, data_dir):
    """保留官方题目、候选段落及答案别名顺序；标注不进入模型上下文。"""
    path = Path(data_dir).expanduser() / SPLIT_FILES[split]
    samples, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            qid, question = raw["id"], raw["question"]
            if not isinstance(qid, str) or not qid or qid in seen:
                raise ValueError(f"{path}: empty or duplicate question ID {qid!r}")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{qid}: missing question")
            if raw["answerable"] is not True:
                raise ValueError(f"{qid}: MuSiQue-Ans requires an answerable question")
            answer, aliases = raw["answer"], raw["answer_aliases"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"{qid}: missing answer")
            if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
                raise ValueError(f"{qid}: answer_aliases must be a list of strings")
            decomposition = raw["question_decomposition"]
            if not isinstance(decomposition, list) or len(decomposition) not in (2, 3, 4):
                raise ValueError(f"{qid}: expected a 2-, 3-, or 4-hop question")
            paragraphs = raw["paragraphs"]
            if not isinstance(paragraphs, list) or not paragraphs:
                raise ValueError(f"{qid}: expected candidate paragraphs")
            blocks = []
            for paragraph in paragraphs:
                title, text = paragraph["title"], paragraph["paragraph_text"]
                if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{qid}: invalid paragraph title or text")
                blocks.append(f"Title: {title}\n{text}")
            samples.append(MusiqueSample(
                qid, "\n\n".join(blocks), question, (answer, *aliases), len(decomposition),
            ))
            seen.add(qid)
    if not samples:
        raise ValueError(f"{path}: empty dataset")
    return tuple(samples)


def musique_data_identity(samples, data_dir, split):
    fingerprint = hashlib.blake2b(digest_size=32)
    for sample in samples:
        fingerprint.update(json.dumps(asdict(sample), ensure_ascii=False, sort_keys=True).encode("utf-8"))
        fingerprint.update(b"\n")
    return {
        "directory": str(Path(data_dir).expanduser().resolve()),
        "filename": SPLIT_FILES[split], "split": split, "source_release": "v1.0",
        "samples": len(samples), "fingerprint": fingerprint.hexdigest(),
    }
