"""五种 token-memory 写入流程，按题目、更新点和轨迹依次平均训练目标。"""

from contextlib import contextmanager
import hashlib
import random
from time import perf_counter

import torch
from torch import nn

from latent_working_memory.v3.config import DYNAMIC_METHODS


def memory_change_score(old, rewritten, epsilon):
    """逐 slot RMS 归一化仅用于评分，不改变保存的记忆。"""
    old, rewritten = old.float(), rewritten.float()
    old = old / (old.square().mean(dim=-1, keepdim=True) + epsilon).sqrt()
    rewritten = rewritten / (rewritten.square().mean(dim=-1, keepdim=True) + epsilon).sqrt()
    return torch.linalg.vector_norm(rewritten - old) / (torch.linalg.vector_norm(old) + epsilon)


def damage_action(l0, lrw, lapp, threshold_d, threshold_g, eta):
    damage, gain = lrw - l0, lrw - lapp
    return (gain > threshold_g) or (damage > threshold_d and gain > eta)


def example_rng(seed, epoch, identifier):
    value = hashlib.sha256(f"{seed}:{epoch}:{identifier}".encode()).digest()
    return random.Random(int.from_bytes(value[:8], "big"))


@contextmanager
def measured(device, event, key):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        event[key] += perf_counter() - start


class TokenMemoryTask(nn.Module):
    def __init__(self, codec, tokenizer, config):
        super().__init__()
        self.codec, self.tokenizer, self.cfg = codec, tokenizer, config

    @property
    def device(self):
        return self.codec.memory_embeddings.device

    def ids(self, values):
        return torch.tensor(values, dtype=torch.long, device=self.device)

    def prompt_ids(self, question):
        return self.ids(
            self.tokenizer.encode(
                self.cfg.qa_prompt.format(question=question), add_special_tokens=False
            )
        )

    def qa_losses(self, blocks, trajectory, qa_ids):
        return self._qa_losses_batch([(blocks, trajectory, qa_ids)])[0]

    def _qa_losses_batch(self, requests, events=None, timing_key="read_seconds"):
        """每轮每条轨迹最多 qa_batch_size 题，合并读取并还原每题损失顺序。"""
        memories = [torch.cat(blocks) for blocks, _, _ in requests]
        pieces = [[] for _ in requests]
        for start in range(0, max(len(ids) for _, _, ids in requests), self.cfg.qa_batch_size):
            indices, counts, memory_rows, prompts, answers = [], [], [], [], []
            for index, (_, trajectory, qa_ids) in enumerate(requests):
                questions = [
                    trajectory.qas[qid] for qid in qa_ids[start : start + self.cfg.qa_batch_size]
                ]
                if not questions:
                    continue
                indices.append(index)
                counts.append(len(questions))
                memory_rows.extend([memories[index]] * len(questions))
                prompts.extend(self.prompt_ids(qa.question) for qa in questions)
                answers.extend(self.ids(qa.answer_ids) for qa in questions)
            timer = {timing_key: 0.0}
            with measured(self.device, timer, timing_key):
                values = self.codec.answer_nll(memory_rows, prompts, answers)
            for index, values in zip(indices, values.split(counts), strict=True):
                pieces[index].append(values)
                if events is not None:
                    # 共享调用的耗时按实际参与轨迹均分，汇总时不会重复计时。
                    events[index][timing_key] += timer[timing_key] / len(indices)
        return [torch.cat(values) for values in pieces]

    def _write_batch(self, inputs, histories, events, output_slots=None):
        timer = {"write_seconds": 0.0}
        with measured(self.device, timer, "write_seconds"):
            results = self.codec.compress_batch(
                [self.ids(ids) for ids in inputs], histories, output_slots
            )
        for event in events:
            event["write_calls"] += 1
            event["write_seconds"] += timer["write_seconds"] / len(events)
        return results

    @staticmethod
    def _event(trajectory, step):
        return {
            "step": step,
            "segment_id": trajectory.segments[step].segment_id,
            "action": "initial",
            "slots": 0,
            "write_calls": 0,
            "write_seconds": 0.0,
            "gate_qa_reads": 0,
            "gate_seconds": 0.0,
            "read_seconds": 0.0,
            "scores": {},
        }

    def _states_batch(self, trajectories, epoch=0, force_policy=False):
        blocks = [[] for _ in trajectories]
        rngs = [example_rng(self.cfg.seed, epoch, row.trajectory_id) for row in trajectories]
        single = self.cfg.method == "icae_single"
        steps = 1 if single else max(len(row.segments) for row in trajectories)
        for step in range(steps):
            active = [i for i, row in enumerate(trajectories) if step < len(row.segments)]
            events = {
                i: self._event(
                    trajectories[i], len(trajectories[i].segments) - 1 if single else step
                )
                for i in active
            }

            def write(indices, histories, output_slots=None):
                return self._write_batch(
                    [
                        trajectories[i].full_input_ids
                        if single
                        else trajectories[i].segments[step].input_ids
                        for i in indices
                    ],
                    histories,
                    [events[i] for i in indices],
                    output_slots,
                )

            if single or self.cfg.method in {"icae_multi", "autocompressors"} or step == 0:
                histories = [
                    blocks[i] if self.cfg.method == "autocompressors" else [] for i in active
                ]
                candidates = write(active, histories)
                for i, candidate in zip(active, candidates, strict=True):
                    blocks[i] = blocks[i] + [candidate]
                    events[i]["action"] = (
                        "single" if single else "initial" if step == 0 else "append"
                    )
                del candidates, candidate
            elif self.cfg.stage == "warmup" and not force_policy:
                choices = {i: rngs[i].random() < self.cfg.append_probability for i in active}
                candidates = write(
                    active,
                    [[] if choices[i] else [blocks[i][-1]] for i in active],
                    [self.cfg.append_slots if choices[i] else len(blocks[i][-1]) for i in active],
                )
                for i, candidate in zip(active, candidates, strict=True):
                    append = choices[i]
                    blocks[i] = blocks[i] + [candidate] if append else blocks[i][:-1] + [candidate]
                    events[i]["action"] = "append" if append else "overwrite"
                del candidates, candidate
            else:
                # 覆盖保留末块的原有大小，使新旧记忆始终可以逐 slot 比较。
                rewritten = write(
                    active, [[blocks[i][-1]] for i in active], [len(blocks[i][-1]) for i in active]
                )
                if self.cfg.method == "memory_change":
                    timer = {"gate_seconds": 0.0}
                    with torch.no_grad(), measured(self.device, timer, "gate_seconds"):
                        scores = [
                            float(
                                memory_change_score(blocks[i][-1], candidate, self.cfg.rms_epsilon)
                            )
                            for i, candidate in zip(active, rewritten, strict=True)
                        ]
                    choices = {
                        i: score >= self.cfg.threshold_i
                        for i, score in zip(active, scores, strict=True)
                    }
                    for i, score in zip(active, scores, strict=True):
                        events[i]["scores"] = {"I": score}
                        events[i]["gate_seconds"] += timer["gate_seconds"] / len(active)
                    indices = [i for i in active if choices[i]]
                    appended = (
                        dict(
                            zip(
                                indices,
                                write(
                                    indices,
                                    [[] for _ in indices],
                                    [self.cfg.append_slots] * len(indices),
                                ),
                                strict=True,
                            )
                        )
                        if indices
                        else {}
                    )
                else:
                    appended = dict(
                        zip(
                            active,
                            write(
                                active, [[] for _ in active], [self.cfg.append_slots] * len(active)
                            ),
                            strict=True,
                        )
                    )
                    gate_ids = [trajectories[i].usage[step].gate_qa_ids for i in active]
                    if not all(gate_ids):
                        raise ValueError(
                            "information-loss updates require historical gate questions"
                        )
                    with torch.no_grad():
                        losses = []
                        for states in (
                            [blocks[i] for i in active],
                            [
                                blocks[i][:-1] + [candidate]
                                for i, candidate in zip(active, rewritten, strict=True)
                            ],
                            [blocks[i] + [appended[i]] for i in active],
                        ):
                            values = self._qa_losses_batch(
                                [
                                    (state, trajectories[i], ids)
                                    for i, state, ids in zip(active, states, gate_ids, strict=True)
                                ],
                                [events[i] for i in active],
                                "gate_seconds",
                            )
                            losses.append([float(value.mean()) for value in values])
                    choices = {}
                    for i, ids, l0, lrw, lapp in zip(active, gate_ids, *losses, strict=True):
                        events[i]["gate_qa_reads"] = 3 * len(ids)
                        events[i]["scores"] = {
                            "L0": l0,
                            "Lrw": lrw,
                            "Lapp": lapp,
                            "d": lrw - l0,
                            "g": lrw - lapp,
                        }
                        choices[i] = damage_action(
                            l0, lrw, lapp, self.cfg.threshold_d, self.cfg.threshold_g, self.cfg.eta
                        )
                    del states, values
                for i, candidate in zip(active, rewritten, strict=True):
                    append = choices[i]
                    blocks[i] = (
                        blocks[i] + [appended[i]] if append else blocks[i][:-1] + [candidate]
                    )
                    events[i]["action"] = "append" if append else "overwrite"
                # 未选中的候选不进入后续计算图，不添加辅助损失。
                del rewritten, appended, candidate
            for i in active:
                events[i]["slots"] = sum(map(len, blocks[i]))
            yield [(i, blocks[i], events[i]) for i in active]

    def _states(self, trajectory, epoch=0, force_policy=False):
        for states in self._states_batch([trajectory], epoch, force_policy):
            _, blocks, event = states[0]
            yield blocks, event

    def build_memory(self, trajectory, epoch=0, force_policy=True):
        events = []
        for blocks, event in self._states(trajectory, epoch, force_policy):
            events.append(event)
        return blocks, events

    def _qa_objective(self, trajectories, epoch):
        losses = [[] for _ in trajectories]
        events = [[] for _ in trajectories]
        old_losses, new_losses = [[] for _ in trajectories], [[] for _ in trajectories]
        indices, requests, current_events, new_counts = [], [], [], []
        updates = (
            1
            if self.cfg.method == "icae_single"
            else max(len(row.segments) for row in trajectories)
        )
        for update, states in enumerate(self._states_batch(trajectories, epoch)):
            for i, blocks, event in states:
                trajectory = trajectories[i]
                events[i].append(event)
                if self.cfg.method in DYNAMIC_METHODS:
                    usage = trajectory.usage[event["step"]]
                    new_ids, old_ids = usage.new_qa_ids, usage.old_qa_ids
                elif event["step"] == len(trajectory.segments) - 1:
                    last_segment = trajectory.segments[-1].segment_id
                    new_ids = tuple(
                        qid
                        for qid, qa in trajectory.qas.items()
                        if qa.role != "gate" and qa.segment_id == last_segment
                    )
                    old_ids = tuple(
                        qid
                        for qid, qa in trajectory.qas.items()
                        if qa.role != "gate" and qa.segment_id != last_segment
                    )
                else:
                    continue
                indices.append(i)
                requests.append((blocks, trajectory, new_ids + old_ids))
                current_events.append(event)
                new_counts.append(len(new_ids))
            # Baseline 只读取最终记忆；变长轨迹结束后合并读取，避免逐条执行 reader。
            if not requests or (self.cfg.method not in DYNAMIC_METHODS and update + 1 < updates):
                continue
            values = self._qa_losses_batch(requests, current_events)
            for i, value, count in zip(indices, values, new_counts, strict=True):
                # 每题先平均答案 token，再以实际题数平均；最后平均更新点。
                losses[i].append(value.mean())
                new_losses[i].extend(value[:count].detach().tolist())
                old_losses[i].extend(value[count:].detach().tolist())
            indices, requests, current_events, new_counts = [], [], [], []
        metrics = []
        for row_events, new, old in zip(events, new_losses, old_losses, strict=True):
            row = self._memory_metrics(row_events)
            row.update(
                {
                    "task_qa_reads": float(len(new) + len(old)),
                    "read_seconds": sum(event["read_seconds"] for event in row_events),
                    "qa_new_nll": sum(new) / len(new),
                    "qa_old_nll": sum(old) / len(old) if old else 0.0,
                    "qa_new_count": float(len(new)),
                    "qa_old_count": float(len(old)),
                }
            )
            metrics.append(row)
        return [torch.stack(values).mean() for values in losses], metrics

    def _pretrain_objective(self, examples):
        chunks = []
        for example in examples:
            chunks.append(
                [
                    example.input_ids[i : i + self.cfg.segment_tokens]
                    for i in range(0, len(example.input_ids), self.cfg.segment_tokens)
                ]
                if self.cfg.method == "icae_multi"
                else [example.input_ids]
            )
        blocks = [[] for _ in examples]
        for step in range(max(map(len, chunks))):
            indices = [i for i, values in enumerate(chunks) if step < len(values)]
            candidates = self.codec.compress_batch([self.ids(chunks[i][step]) for i in indices])
            for i, candidate in zip(indices, candidates, strict=True):
                blocks[i].append(candidate)
        targets = [example.target_ids + (self.tokenizer.eos_token_id,) for example in examples]
        losses = self.codec.answer_nll(
            [torch.cat(values) for values in blocks],
            [
                self.ids(
                    self.tokenizer.encode(
                        self.cfg.ae_prompt if example.task == "ae" else self.cfg.lm_prompt,
                        add_special_tokens=False,
                    )
                )
                for example in examples
            ],
            [self.ids(target) for target in targets],
        )
        metrics = [
            {
                "input_tokens": float(len(example.input_ids)),
                "target_tokens": float(len(target)),
                "slots_final": float(sum(map(len, values))),
                "write_calls": float(len(values)),
                "ae_samples": float(example.task == "ae"),
                "lm_samples": float(example.task == "continuation"),
            }
            for example, target, values in zip(examples, targets, blocks, strict=True)
        ]
        return list(losses.unbind()), metrics

    def _ac_objective(self, examples, epoch):
        # 每个训练子块最多两段；子块结束后 detach 全部累计摘要。
        # 冻结读取端与写入 LoRA 分开执行，但 LM 只预测尚未进入摘要的原文。
        segments = []
        for example in examples:
            tokens = (
                example.input_ids
                if example.task == "ae"
                else example.input_ids + example.target_ids
            )
            if len(tokens) < 3:
                raise ValueError(
                    "AutoCompressors has no trainable next-token target in the second segment"
                )
            rng = example_rng(self.cfg.seed, epoch, example.sample_id)
            parts, offset = [], 0
            while offset < len(tokens):
                maximum = self.cfg.ac_max_segment_tokens
                if offset == 0:
                    # 短输入也保留第二段监督，使冻结 reader 的损失可经首段记忆反传。
                    maximum = min(maximum, len(tokens) - 2)
                size = rng.randint(min(self.cfg.ac_min_segment_tokens, maximum), maximum)
                parts.append(tokens[offset : offset + size])
                offset += size
            if len(parts) < 2 or self.cfg.ac_bptt_steps < 2:
                raise ValueError(
                    "frozen-reader AutoCompressors LM needs at least two segments per BPTT group"
                )
            if len(parts) == 2 and len(parts[-1]) < 2:
                raise ValueError(
                    "AutoCompressors has no trainable next-token target in the second segment"
                )
            segments.append(parts)
        blocks, losses = [[] for _ in examples], [[] for _ in examples]
        targets = [0 for _ in examples]
        for step in range(max(map(len, segments))):
            active = [i for i, parts in enumerate(segments) if step < len(parts)]
            group_end = {
                i: (step + 1) % self.cfg.ac_bptt_steps == 0 or step == len(segments[i]) - 1
                for i in active
            }
            indices, answers = [], []
            for i in active:
                target = segments[i][step][1:]
                # 原始 next-token 目标跨同一子块分段边界；子块首 token 不计入损失。
                if not group_end[i]:
                    target += segments[i][step + 1][:1]
                if target:
                    indices.append(i)
                    answers.append(self.ids(target))
            if indices:
                values = self.codec.answer_nll(
                    [
                        torch.cat(blocks[i]) if blocks[i] else self.codec.memory_embeddings[:0]
                        for i in indices
                    ],
                    [self.ids(segments[i][step][:1]) for i in indices],
                    answers,
                )
                for i, value, answer in zip(indices, values, answers, strict=True):
                    losses[i].append(value * len(answer))
                    targets[i] += len(answer)
            indices = [i for i in active if step < len(segments[i]) - 1]
            if indices:
                candidates = self.codec.compress_batch(
                    [self.ids(segments[i][step]) for i in indices], [blocks[i] for i in indices]
                )
                for i, candidate in zip(indices, candidates, strict=True):
                    blocks[i].append(candidate)
            for i in active:
                if group_end[i]:
                    blocks[i] = [block.detach() for block in blocks[i]]
        metrics = [
            {
                "input_tokens": float(sum(map(len, parts))),
                "target_tokens": float(target),
                "slots_final": float(sum(map(len, values))),
                "write_calls": float(len(parts) - 1),
                "segments": float(len(parts)),
            }
            for parts, target, values in zip(segments, targets, blocks, strict=True)
        ]
        return [
            torch.stack(values).sum() / count for values, count in zip(losses, targets, strict=True)
        ], metrics

    @staticmethod
    def _memory_metrics(events):
        metrics = {
            "slots_final": float(events[-1]["slots"]),
            "slots_mean": sum(event["slots"] for event in events) / len(events),
            "appends": float(sum(event["action"] == "append" for event in events)),
            "overwrites": float(sum(event["action"] == "overwrite" for event in events)),
            "write_calls": float(sum(event["write_calls"] for event in events)),
            "gate_qa_reads": float(sum(event["gate_qa_reads"] for event in events)),
            "write_seconds": sum(event["write_seconds"] for event in events),
            "gate_seconds": sum(event["gate_seconds"] for event in events),
        }
        for score in ("I", "L0", "Lrw", "Lapp", "d", "g"):
            values = [event["scores"][score] for event in events if score in event["scores"]]
            if values:
                metrics[f"gate_{score}"] = sum(values) / len(values)
        return metrics

    def forward(self, example, epoch=0, differentiable=True, batched=False):
        examples = list(example) if batched else [example]
        if not examples:
            raise ValueError("a microbatch must contain at least one example")
        with torch.set_grad_enabled(differentiable):
            if self.cfg.stage == "pretrain":
                losses, metrics = self._pretrain_objective(examples)
            elif self.cfg.stage == "lm":
                losses, metrics = self._ac_objective(examples, epoch)
            else:
                losses, metrics = self._qa_objective(examples, epoch)
            loss = torch.stack(losses).mean()
        names = set().union(*(row.keys() for row in metrics))
        return {
            "loss": loss,
            "metrics": {
                name: sum(row.get(name, 0.0) for row in metrics) / len(examples) for name in names
            },
        }

    def trainable_state_dict(self):
        return self.codec.trainable_state_dict()

    def load_trainable_state_dict(self, state):
        self.codec.load_trainable_state_dict(state)
