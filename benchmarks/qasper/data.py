"""按论文读取 QASPER v0.3；正文使用官方 LED 的 full_text 章节顺序。"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from benchmarks.vendor.qasper import answer_reference


SPLIT_FILES = {"test": "qasper-test-v0.3.json"}
QUESTION_FORMAT = "\n\nQuestion: {question}"
SYSTEM_PROMPT = (
    "Answer the question using the provided research paper. Give a concise answer. "
    "For a yes/no question, answer Yes or No. If multiple spans are needed, "
    "separate them with commas. If the paper does not answer the question, reply exactly: Unanswerable."
)


@dataclass(frozen=True, slots=True)
class AnnotatedQuestion:
    id: str
    question: str
    references: tuple[dict, ...]


@dataclass(frozen=True, slots=True)
class QasperPaper:
    id: str
    history: str
    questions: tuple[AnnotatedQuestion, ...]


def load_qasper(split, data_dir):
    """保留官方论文、题目和标注顺序。每个题号对应一组参考答案。"""
    path = Path(data_dir).expanduser() / SPLIT_FILES[split]
    with path.open(encoding="utf-8") as stream:
        corpus = json.load(stream)
    papers, seen = [], set()
    for paper_id, raw in corpus.items():
        paragraphs = []
        for section in raw["full_text"]:
            if section["section_name"] is not None:
                paragraphs.append(section["section_name"])
            for paragraph in section["paragraphs"]:
                text = paragraph.replace("\n", " ").strip()
                if text:
                    paragraphs.append(text)
        history = "\n\n".join(paragraphs)
        if not history.strip():
            raise ValueError(f"{paper_id}: empty full text")
        questions = []
        for qa in raw["qas"]:
            qid, question = qa["question_id"], qa["question"]
            if not isinstance(qid, str) or not qid or qid in seen:
                raise ValueError(f"{paper_id}: empty or duplicate question ID {qid!r}")
            if not isinstance(question, str) or not question.strip() or not qa["answers"]:
                raise ValueError(f"{qid}: missing question or annotations")
            seen.add(qid)
            refs = tuple(answer_reference(annotation["answer"]) for annotation in qa["answers"])
            questions.append(AnnotatedQuestion(qid, question, refs))
        if not questions:
            raise ValueError(f"{paper_id}: no questions")
        papers.append(QasperPaper(paper_id, history, tuple(questions)))
    if not papers:
        raise ValueError(f"{path}: empty dataset")
    return tuple(papers)


def qasper_data_identity(papers, data_dir, split):
    fingerprint = hashlib.blake2b(digest_size=32)
    for paper in papers:
        fingerprint.update(json.dumps(asdict(paper), ensure_ascii=False, sort_keys=True).encode("utf-8"))
        fingerprint.update(b"\n")
    return {
        "directory": str(Path(data_dir).expanduser().resolve()),
        "filename": SPLIT_FILES[split], "split": split, "source_release": "v0.3",
        "papers": len(papers), "samples": sum(len(paper.questions) for paper in papers),
        "fingerprint": fingerprint.hexdigest(),
    }
