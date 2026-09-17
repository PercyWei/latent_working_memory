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

    def forward(self, trajectory, read_task, one_shot=False):
        single = not isinstance(trajectory, (tuple, list))
        rows = [trajectory] if single else trajectory
        both = isinstance(read_task, str) and read_task == "both"
        tasks = [read_task] * len(rows) if isinstance(read_task, str) else list(read_task)
        if len(tasks) != len(rows) or (not both and any(t not in {"ae", "lm"} for t in tasks)):
            raise ValueError("read_task must be ae/lm per trajectory, or both for evaluation")
        ids = [row.token_ids.to(self.ae_prompt.device) for row in rows]
        eos = ids[0].new_tensor([self.eos_id])
        ends = [(row.write_ends[-1],) if one_shot else row.write_ends for row in rows]
        capacity = rows[0].capacity
        if any(row.capacity != capacity for row in rows):
            raise ValueError("a microbatch requires equal memory capacities")
        q = [len(tokens) - row.write_ends[-1] for tokens, row in zip(ids, rows)]
        losses, records = [[] for _ in rows], [[] for _ in rows]
        memory, active = None, list(range(len(rows)))
        values, locations, batch_sizes = [], [], []
        for step in range(max(map(len, ends))):
            keep = [position for position, i in enumerate(active) if step < len(ends[i])]
            if len(keep) != len(active):
                # Differentiable compaction: finished trajectories receive no dummy writes.
                memory = memory.index_select(0, torch.tensor(keep, device=memory.device))
                active = [active[position] for position in keep]
            batch_sizes.append(len(active))
            segments = [ids[i][ends[i][step - 1] if step else 0 : ends[i][step]] for i in active]
            memory = (
                self.codec.write(memory, segments[0], capacity)
                if single
                else self.codec.write_batch(memory, segments, capacity)
            )
            for i in active:
                records[i].append(
                    {
                        "round": step + 1,
                        "seen_tokens": ends[i][step],
                        "ae": None,
                        "lm": None,
                        "ae_tokens": 0,
                        "lm_tokens": 0,
                    }
                )
            for objective in ("ae", "lm") if both else (None,):
                names = [objective if both else tasks[i] for i in active]
                targets = [
                    torch.cat(
                        (
                            ids[i][: ends[i][step]]
                            if name == "ae"
                            else ids[i][ends[i][step] : ends[i][step] + q[i]],
                            eos,
                        )
                    )
                    for i, name in zip(active, names, strict=True)
                ]
                prompts = [getattr(self, f"{name}_prompt") for name in names]
                reads = (
                    self.codec.read_loss(memory, prompts[0], targets[0])[None]
                    if single
                    else self.codec.read_loss_batch(memory, prompts, targets)
                )
                values.append(reads)
                for position, (i, name, target) in enumerate(
                    zip(active, names, targets, strict=True)
                ):
                    weight = (
                        ((1 - self.config.lm_ratio) if name == "ae" else self.config.lm_ratio)
                        if both
                        else 1
                    )
                    losses[i].append(reads[position] * weight)
                    records[i][-1][f"{name}_tokens"] = len(target)
                    locations.append((records[i][-1], name))
        # One host transfer per microbatch, even as the active batch shrinks.
        for (record, name), value in zip(
            locations, torch.cat(values).detach().cpu().tolist(), strict=True
        ):
            record[name] = value
        sample_losses = torch.stack(
            [torch.stack(reads).sum() / len(cuts) for reads, cuts in zip(losses, ends, strict=True)]
        )
        if single:
            return {"loss": sample_losses[0], "rounds": records[0], "batch_sizes": batch_sizes}
        return {
            "loss": sample_losses.mean(),
            "sample_losses": sample_losses,
            "rounds": records,
            "batch_sizes": batch_sizes,
        }
