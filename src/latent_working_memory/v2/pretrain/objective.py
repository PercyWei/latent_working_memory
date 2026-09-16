"""每次压缩后的累计历史 AE 与紧邻续文 LM，按 token、压缩次数、轨迹归一化。"""

import torch
from torch import nn


class ReconstructionTask(nn.Module):
    def __init__(self, codec, tokenizer, config):
        super().__init__()
        self.codec, self.config = codec, config
        if tokenizer.eos_token_id is None:
            raise ValueError("reconstruction requires an EOS token")
        self.eos_id = tokenizer.eos_token_id
        for name in ("ae_prompt", "lm_prompt"):
            self.register_buffer(
                name,
                torch.tensor(
                    tokenizer.encode(getattr(config, name), add_special_tokens=False),
                    dtype=torch.long,
                ),
                persistent=False,
            )

    def validate_data(self, datasets):
        limit = self.codec.max_positions
        for splits in datasets.values():
            for rows in splits.values():
                for row in rows:
                    length, k = row.write_ends[-1], row.capacity
                    q = len(row.token_ids) - length
                    if q < 1 or row.token_ids.ndim != 1:
                        raise ValueError("trajectory requires tokens and a final continuation")
                    # Also reserve the one-shot evaluation input window.
                    costs = [
                        length,
                        k + len(self.ae_prompt) + length + 1,
                        k + len(self.lm_prompt) + q + 1,
                    ]
                    for i, (start, end) in enumerate(
                        zip((0,) + row.write_ends[:-1], row.write_ends, strict=True)
                    ):
                        if end <= start or k > end - start + (k if i else 0):
                            raise ValueError("invalid write boundaries or capacity")
                        costs.append(end - start + (k if i else 0))
                    if max(costs) > limit:
                        raise ValueError(
                            f"trajectory {row.document_id} needs {max(costs)} positions; model supports {limit}"
                        )

    def forward(self, trajectory, include_lm=None, one_shot=False):
        include_lm = self.config.objective == "ae_lm" if include_lm is None else include_lm
        ids = trajectory.token_ids.to(self.ae_prompt.device)
        eos = ids.new_tensor([self.eos_id])
        ends = (trajectory.write_ends[-1],) if one_shot else trajectory.write_ends
        q = len(ids) - trajectory.write_ends[-1]
        memory, previous, losses, records = None, 0, [], []
        for step, end in enumerate(ends, 1):
            memory = self.codec.write(memory, ids[previous:end], trajectory.capacity)
            ae = self.codec.read_loss(memory, self.ae_prompt, torch.cat((ids[:end], eos)))
            lm = (
                self.codec.read_loss(memory, self.lm_prompt, torch.cat((ids[end : end + q], eos)))
                if include_lm
                else None
            )
            loss = ae
            if lm is not None and self.config.objective == "ae_lm":
                loss = loss + self.config.lm_weight * lm
            losses.append(loss)
            records.append(
                {
                    "round": step,
                    "seen_tokens": end,
                    "ae": float(ae.detach()),
                    "lm": float(lm.detach()) if lm is not None else None,
                    "ae_tokens": end + 1,
                    "lm_tokens": q + 1 if lm is not None else 0,
                }
            )
            previous = end
        return {"loss": torch.stack(losses).mean(), "rounds": records}
