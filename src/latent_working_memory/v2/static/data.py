"""GMSA JSONL 输入；完整文本在内存中分词，不保存派生数据副本。"""

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class StaticDataset(Dataset):
    def __init__(self, path, tokenizer, stage, max_context_tokens, max_target_tokens):
        if stage not in {"autoencoding", "finetune"}:
            raise ValueError("unsupported static stage")
        if min(max_context_tokens, max_target_tokens) < 1 or tokenizer.eos_token_id is None:
            raise ValueError("positive token budgets and an EOS token are required")
        self.rows = []
        self.references = []
        with Path(path).open() as source:
            for line_number, line in enumerate(source, 1):
                row = json.loads(line)
                context = row["input"]
                if not isinstance(context, str) or not context.strip():
                    raise ValueError(f"{path}:{line_number}: input must be non-empty text")
                if stage == "autoencoding":
                    prompt = "Restate the aforementioned Text."
                    references = [context]
                else:
                    prompt, references = row["prompt"], row["answer"]
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError(f"{path}:{line_number}: prompt must be non-empty text")
                    # Canonical upstream JSONL representation: a list of reference strings.
                    if (
                        not isinstance(references, list)
                        or not references
                        or any(
                            not isinstance(value, str) or not value.strip() for value in references
                        )
                    ):
                        raise ValueError(
                            f"{path}:{line_number}: answer must be a non-empty text list"
                        )
                ids = tokenizer.encode(context, add_special_tokens=False)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                target = tokenizer.encode(references[0], add_special_tokens=False)
                if not ids or not prompt_ids or not target:
                    raise ValueError(f"{path}:{line_number}: empty token sequence")
                if len(ids) > max_context_tokens or len(target) + 1 > max_target_tokens:
                    raise ValueError(
                        f"{path}:{line_number}: sample exceeds token budget; no truncation"
                    )
                self.rows.append(
                    {
                        "context_ids": torch.tensor(ids),
                        "prompt_ids": torch.tensor(prompt_ids),
                        "labels": torch.tensor([*target, tokenizer.eos_token_id]),
                    }
                )
                self.references.append(references)
        if not self.rows:
            raise ValueError("dataset is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class StaticCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        result = {}
        for name in ("context_ids", "prompt_ids", "labels"):
            values = [row[name] for row in rows]
            result[name] = pad_sequence(
                values,
                batch_first=True,
                padding_value=-100 if name == "labels" else self.pad_token_id,
            )
            if name != "labels":
                result[name.replace("ids", "mask")] = (
                    torch.arange(result[name].shape[1])[None]
                    < torch.tensor([len(value) for value in values])[:, None]
                )
        return result
