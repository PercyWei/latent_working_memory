"""五种 token-memory 写入流程，按题目、更新点和轨迹依次平均训练目标。"""

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from latent_working_memory.v3.config import DYNAMIC_METHODS, DYNAMIC_PRETRAIN_METHODS
from latent_working_memory.v3.segmentation import ac_plan, example_rng, icae_multi_plan


@dataclass
class QAWindowState:
    step: int
    blocks: list
    rngs: list
    events: list
    new_losses: list
    old_losses: list


def memory_change_score(old, rewritten, epsilon):
    """逐 slot RMS 归一化仅用于评分，不改变保存的记忆。"""
    old, rewritten = old.float(), rewritten.float()
    old = old / (old.square().mean(dim=-1, keepdim=True) + epsilon).sqrt()
    rewritten = rewritten / (rewritten.square().mean(dim=-1, keepdim=True) + epsilon).sqrt()
    return torch.linalg.vector_norm(rewritten - old) / (torch.linalg.vector_norm(old) + epsilon)


def damage_action(l0, lrw, lapp, threshold_d, threshold_g, eta):
    damage, gain = lrw - l0, lrw - lapp
    return (gain > threshold_g) or (damage > threshold_d and gain > eta)


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

    def _write_batch(self, inputs, histories, events, output_slots=None, actions=None):
        timer = {"write_seconds": 0.0}
        with measured(self.device, timer, "write_seconds"):
            if self.cfg.writer_mode == "local":
                results = self.codec.compress_batch(
                    [self.ids(ids) for ids in inputs], histories, output_slots
                )
            else:
                results = self.codec.compress_batch(
                    [self.ids(ids) for ids in inputs], histories, output_slots, actions=actions
                )
        for event in events:
            event["write_calls"] += 1
            event["write_seconds"] += timer["write_seconds"] / len(events)
        return results

    @staticmethod
    def _event(step, segment_id):
        return {
            "step": step,
            "segment_id": segment_id,
            "action": "initial",
            "slots": 0,
            "write_calls": 0,
            "write_seconds": 0.0,
            "gate_qa_reads": 0,
            "gate_seconds": 0.0,
            "read_seconds": 0.0,
            "scores": {},
        }

    def _num_updates(self, trajectories):
        if self.cfg.method == "icae_single":
            return 1
        if self.cfg.method == "icae_multi":
            return max(
                len(
                    icae_multi_plan(
                        row.full_input_ids,
                        self.codec.memory_slots,
                        self.cfg.icae_min_segments,
                        self.cfg.icae_max_segments,
                        self.cfg.seed,
                        row.trajectory_id,
                    )[0]
                )
                for row in trajectories
            )
        if self.cfg.method == "autocompressors":
            return self.cfg.ac_num_segments
        return max(len(row.segments) for row in trajectories)

    def _states_batch(
        self, trajectories, epoch=0, force_policy=False, blocks=None, rngs=None, start=0, stop=None
    ):
        if self.cfg.method == "dynamic":
            raise ValueError("shared dynamic pretraining has no FactQA memory policy")
        if blocks is None:
            blocks = [[] for _ in trajectories]
        if rngs is None:
            rngs = [example_rng(self.cfg.seed, epoch, row.trajectory_id) for row in trajectories]
        single = self.cfg.method == "icae_single"
        multi = self.cfg.method == "icae_multi"
        ac = self.cfg.method == "autocompressors"
        if single:
            chunks = [[row.full_input_ids] for row in trajectories]
        elif multi:
            plans = [
                icae_multi_plan(
                    row.full_input_ids,
                    self.codec.memory_slots,
                    self.cfg.icae_min_segments,
                    self.cfg.icae_max_segments,
                    self.cfg.seed,
                    row.trajectory_id,
                )
                for row in trajectories
            ]
            chunks = [plan[0] for plan in plans]
            slot_counts = [plan[1] for plan in plans]
        elif ac:
            plans = [
                ac_plan(
                    row.full_input_ids,
                    self.codec.memory_slots,
                    self.cfg.ac_num_segments,
                    self.cfg.bptt_steps,
                    self.codec.max_positions,
                )
                for row in trajectories
            ]
            chunks = [plan[0] for plan in plans]
            slot_counts = [plan[1] for plan in plans]
        else:
            chunks = [[segment.input_ids for segment in row.segments] for row in trajectories]
        steps = max(map(len, chunks))
        for step in range(start, steps if stop is None else min(stop, steps)):
            active = [i for i, row_chunks in enumerate(chunks) if step < len(row_chunks)]
            events = {
                i: self._event(
                    len(trajectories[i].segments) - 1 if single else step,
                    None
                    if multi or ac
                    else trajectories[i].segments[-1 if single else step].segment_id,
                )
                for i in active
            }

            def write(indices, histories, output_slots=None, actions=None):
                return self._write_batch(
                    [chunks[i][step] for i in indices],
                    histories,
                    [events[i] for i in indices],
                    [slot_counts[i][step] for i in indices] if multi or ac else output_slots,
                    actions,
                )

            def write_updates(indices, append):
                histories = [
                    ([] if choice else [blocks[i][-1]])
                    if self.cfg.writer_mode == "local"
                    else blocks[i]
                    for i, choice in zip(indices, append, strict=True)
                ]
                return write(
                    indices,
                    histories,
                    [
                        self.cfg.append_slots if choice else len(blocks[i][-1])
                        for i, choice in zip(indices, append, strict=True)
                    ],
                    ["append" if choice else "overwrite" for choice in append],
                )

            if single or self.cfg.method in {"icae_multi", "autocompressors"} or step == 0:
                histories = [
                    blocks[i] if self.cfg.method == "autocompressors" else [] for i in active
                ]
                candidates = write(active, histories, actions=["initial"] * len(active))
                for i, candidate in zip(active, candidates, strict=True):
                    blocks[i] = blocks[i] + [candidate]
                    events[i]["action"] = (
                        "single" if single else "initial" if step == 0 else "append"
                    )
                del candidates, candidate
            elif self.cfg.stage == "warmup" and not force_policy:
                choices = {i: rngs[i].random() < self.cfg.append_probability for i in active}
                candidates = write_updates(active, [choices[i] for i in active])
                for i, candidate in zip(active, candidates, strict=True):
                    append = choices[i]
                    blocks[i] = blocks[i] + [candidate] if append else blocks[i][:-1] + [candidate]
                    events[i]["action"] = "append" if append else "overwrite"
                del candidates, candidate
            else:
                # 覆盖保留末块的原有大小，使新旧记忆始终可以逐 slot 比较。
                rewritten = write_updates(active, [False] * len(active))
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
                                write_updates(indices, [True] * len(indices)),
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
                            write_updates(active, [True] * len(active)),
                            strict=True,
                        )
                    )
                    gate_ids = [trajectories[i].usage[step].gate_qa_ids for i in active]
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

    def _qa_objective(self, trajectories, epoch, window_steps=None, state=None):
        losses = [[] for _ in trajectories]
        if window_steps is not None:
            if state is None:
                state = QAWindowState(
                    0,
                    [[] for _ in trajectories],
                    [example_rng(self.cfg.seed, epoch, row.trajectory_id) for row in trajectories],
                    [[] for _ in trajectories],
                    [[] for _ in trajectories],
                    [[] for _ in trajectories],
                )
            events, new_losses, old_losses = state.events, state.new_losses, state.old_losses
            states = self._states_batch(
                trajectories,
                epoch,
                blocks=state.blocks,
                rngs=state.rngs,
                start=state.step,
                stop=state.step + window_steps,
            )
        else:
            events = [[] for _ in trajectories]
            old_losses, new_losses = [[] for _ in trajectories], [[] for _ in trajectories]
            states = self._states_batch(trajectories, epoch)
        indices, requests, current_events, new_counts = [], [], [], []
        updates = self._num_updates(trajectories)
        for update, current_states in enumerate(states):
            for i, blocks, event in current_states:
                trajectory = trajectories[i]
                events[i].append(event)
                if self.cfg.method in DYNAMIC_METHODS:
                    usage = trajectory.usage[event["step"]]
                    new_ids, old_ids = usage.new_qa_ids, usage.old_qa_ids
                # multi 在累计写满总 K 后监督，独立于数据原始段数。
                elif (
                    event["slots"] == self.codec.memory_slots
                    if self.cfg.method in {"icae_multi", "autocompressors"}
                    else event["step"] == len(trajectory.segments) - 1
                ):
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
        if window_steps is None:
            return [torch.stack(values).mean() for values in losses], metrics, None
        state.step = min(state.step + window_steps, updates)
        # 只把记忆数值交给下一窗口；当前 loss 持有本窗口的图供 engine 立即反传。
        state.blocks = [[block.detach() for block in blocks] for blocks in state.blocks]
        return (
            [
                torch.stack(values).sum() / len(trajectory.segments)
                if values
                else self.codec.memory_embeddings.sum() * 0
                for trajectory, values in zip(trajectories, losses, strict=True)
            ],
            metrics,
            state,
        )

    def _pretrain_objective(self, examples):
        multi = self.cfg.method == "icae_multi"
        if multi:
            plans = [
                icae_multi_plan(
                    example.input_ids,
                    self.codec.memory_slots,
                    self.cfg.icae_min_segments,
                    self.cfg.icae_max_segments,
                    self.cfg.seed,
                    example.sample_id,
                )
                for example in examples
            ]
            chunks = [plan[0] for plan in plans]
            slot_counts = [plan[1] for plan in plans]
        else:
            chunks = [[example.input_ids] for example in examples]
        blocks = [[] for _ in examples]
        for step in range(max(map(len, chunks))):
            indices = [i for i, values in enumerate(chunks) if step < len(values)]
            candidates = self.codec.compress_batch(
                [self.ids(chunks[i][step]) for i in indices],
                output_slots=[slot_counts[i][step] for i in indices] if multi else None,
            )
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
        # 正文固定 n 个压缩块，独立续文只作为最后一次写入后的监督。
        # 每 B 次写入接受紧邻后继文本监督后截断；末次写入由独立续文监督。
        plans = [
            ac_plan(
                example.input_ids,
                self.codec.memory_slots,
                self.cfg.ac_num_segments,
                self.cfg.bptt_steps,
                self.codec.max_positions,
                example_rng(self.cfg.seed, epoch, example.sample_id),
            )
            for example in examples
        ]
        segments = [plan[0] for plan in plans]
        slot_counts = [plan[1] for plan in plans]
        blocks, losses = [[] for _ in examples], [[] for _ in examples]
        targets = [0 for _ in examples]
        continuation = [example.target_ids + (self.tokenizer.eos_token_id,) for example in examples]
        for step in range(self.cfg.ac_num_segments):
            answers = []
            for i in range(len(examples)):
                target = segments[i][step][1:]
                # 下一段首 token 在当前原文末 token 上预测，正文各 target 只计一次。
                if step + 1 < self.cfg.ac_num_segments:
                    target += segments[i][step + 1][:1]
                answers.append(self.ids(target))
            values = self.codec.answer_nll(
                [torch.cat(row) if row else self.codec.memory_embeddings[:0] for row in blocks],
                [self.ids(parts[step][:1]) for parts in segments],
                answers,
            )
            for i, (value, answer) in enumerate(zip(values, answers, strict=True)):
                losses[i].append(value * len(answer))
                targets[i] += len(answer)
            if self.cfg.bptt_steps is not None and step > 0 and step % self.cfg.bptt_steps == 0:
                blocks = [[block.detach() for block in row] for row in blocks]
            candidates = self.codec.compress_batch(
                [self.ids(parts[step]) for parts in segments],
                blocks,
                [counts[step] for counts in slot_counts],
            )
            for row, candidate in zip(blocks, candidates, strict=True):
                row.append(candidate)
        values = self.codec.answer_nll(
            [torch.cat(row) for row in blocks],
            [
                self.ids(self.tokenizer.encode(self.cfg.lm_prompt, add_special_tokens=False))
                for _ in examples
            ],
            [self.ids(target) for target in continuation],
        )
        for i, (value, target) in enumerate(zip(values, continuation, strict=True)):
            losses[i].append(value * len(target))
            targets[i] += len(target)
        metrics = [
            {
                "input_tokens": float(sum(map(len, parts))),
                "target_tokens": float(target),
                "slots_final": float(sum(map(len, values))),
                "write_calls": float(len(parts)),
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

    def forward(
        self,
        example,
        epoch=0,
        differentiable=True,
        batched=False,
        window_steps=None,
        qa_state=None,
        sync_parameters=False,
    ):
        if sync_parameters:
            # 只同步已累积梯度，不执行写入、读取或决策，也不消耗动作 RNG。
            return {
                "loss": sum(
                    parameter.reshape(-1)[0] * 0
                    for parameter in self.parameters()
                    if parameter.requires_grad
                ),
                "metrics": {},
            }
        examples = list(example) if batched else [example]
        if not examples:
            raise ValueError("a microbatch must contain at least one example")
        with torch.set_grad_enabled(differentiable):
            method, stage = self.cfg.method, self.cfg.stage
            if method == "autocompressors" and stage in {"pretrain", "lm"}:
                losses, metrics = self._ac_objective(examples, epoch)
            elif (
                method in ("icae_single", "icae_multi", *DYNAMIC_PRETRAIN_METHODS)
                and stage == "pretrain"
            ):
                losses, metrics = self._pretrain_objective(examples)
            else:
                losses, metrics, qa_state = self._qa_objective(
                    examples, epoch, window_steps, qa_state
                )
            loss = torch.stack(losses).mean()
        names = metrics[0].keys()
        output = {
            "loss": loss,
            "metrics": {name: sum(row[name] for row in metrics) / len(examples) for name in names},
        }
        if window_steps is not None:
            output["qa_state"] = qa_state
        return output

    def trainable_state_dict(self):
        return self.codec.trainable_state_dict()

    def load_trainable_state_dict(self, state):
        self.codec.load_trainable_state_dict(state)
