"""Read shared PersonaMem text/QA and tokenize in memory for the current experiment."""

from collections import defaultdict
import json
from pathlib import Path

from latent_working_memory.data_preparation.personamem.blocks import evidence_blocks
from latent_working_memory.v1.data import Episode, Read, Reference, Source


class PersonaMemDataset:
    def __init__(self, dataset_dir, tokenizer):
        self.dataset_dir = Path(dataset_dir)
        self.tokenizer = tokenizer
        sources = json.loads((self.dataset_dir / "sources.json").read_text())
        qas = [
            json.loads(line) for line in (self.dataset_dir / "qas.jsonl").read_text().splitlines()
        ]
        if len({q["qa_id"] for q in qas}) != len(qas):
            raise ValueError("duplicate PersonaMem QA IDs")
        users = {h["persona_id"]: h for h in sources["histories"]}
        if len(users) != len(sources["histories"]) or len(
            {h["history_id"] for h in users.values()}
        ) != len(users):
            raise ValueError("duplicate PersonaMem history identities")
        by_user = defaultdict(list)
        for qa in qas:
            user = users.get(qa["persona_id"])
            if (
                user is None
                or qa["split"] != user["split"]
                or qa["history_id"] != user["history_id"]
            ):
                raise ValueError("PersonaMem QA source or split mismatch")
            by_user[qa["persona_id"]].append(qa)
        self.records, self.articles = {}, {}
        for user in sources["histories"]:
            history = json.loads(
                (self.dataset_dir / "histories" / f"{user['persona_id']}.json").read_text()
            )
            if any(history[k] != user[k] for k in ("persona_id", "history_id", "split")):
                raise ValueError("PersonaMem history identity mismatch")
            if any(m["role"] not in {"user", "assistant"} for m in history["messages"]):
                raise ValueError(
                    "PersonaMem training history must contain only user/assistant messages"
                )
            blocks = evidence_blocks(history, by_user[user["persona_id"]])
            lengths = [
                len(ids)
                for ids in tokenizer(
                    [b["context"] + "\n\n" for b in blocks], add_special_tokens=False
                )["input_ids"]
            ]
            document = user["history_id"]
            self.records[document] = dict(
                document_id=document,
                persona_id=user["persona_id"],
                split=user["split"],
                input_tokens=sum(lengths),
                paragraph_tokens=lengths,
                qas=len(by_user[user["persona_id"]]),
            )
            self.articles[document] = dict(
                document_id=document,
                persona_id=user["persona_id"],
                split=user["split"],
                paragraphs=blocks,
            )
        # This identity belongs to an experiment's evaluation plan, never to a saved dataset index.
        self.index = dict(
            format="personamem-runtime-v1",
            dataset_dir=str(self.dataset_dir.resolve()),
            tokenizer=dict(
                name_or_path=str(tokenizer.name_or_path),
                tokenizer_class=type(tokenizer).__name__,
                vocabulary=tokenizer.get_vocab(),
                special_tokens_map=tokenizer.special_tokens_map,
            ),
            serialization="role: content; newline between messages; two newlines after each block; "
            "tokenize each block independently without BOS/EOS",
            articles=list(self.records.values()),
        )

    def episode(self, document_id, paragraph_count=None, paragraph_start=0):
        article = self.articles[document_id]
        paragraphs = article["paragraphs"]
        if type(paragraph_start) is not int or not 0 <= paragraph_start < len(paragraphs):
            raise ValueError("invalid dialogue block start")
        count = len(paragraphs) - paragraph_start if paragraph_count is None else paragraph_count
        if type(count) is not int or not 1 <= count <= len(paragraphs) - paragraph_start:
            raise ValueError("invalid dialogue block count")
        selected = paragraphs[paragraph_start : paragraph_start + count]
        ids, ends, sources, reads = [], [], [], []
        episode_id = f"{document_id}:blocks:{paragraph_start}:{paragraph_start + count}"
        for block_index, block in enumerate(selected, paragraph_start):
            start = len(ids)
            ids.extend(self.tokenizer.encode(block["context"] + "\n\n", add_special_tokens=False))
            end = len(ids)
            ends.append(end)
            sources.append(
                Source(
                    source_id=f"{episode_id}:block{block_index}",
                    document_id=document_id,
                    token_start=start,
                    token_end=end,
                    provenance=dict(
                        context=block["context"],
                        questions=block["qas"],
                        message_start=block["message_start"],
                        message_end_exclusive=block["message_end_exclusive"],
                        persona_id=article["persona_id"],
                    ),
                )
            )
            for qa in block["qas"]:
                reads.append(
                    Read(
                        read_id=f"{episode_id}:{qa['id']}",
                        task="qa",
                        prefix_end=end,
                        prompt="Answer the question using the information stored in memory. "
                        f"Give only the answer.\nQuestion: {qa['question']}\nAnswer:",
                        references=(Reference(qa["answers"][0]["text"], ((start, end),)),),
                    )
                )
        lengths = [b - a for a, b in zip((0, *ends[:-1]), ends)]
        expected = self.records[document_id]["paragraph_tokens"][
            paragraph_start : paragraph_start + count
        ]
        if lengths != expected:
            raise ValueError("PersonaMem tokenizer changed after the experiment was prepared")
        return Episode(episode_id, tuple(ids), tuple(ends), tuple(sources), tuple(reads))
