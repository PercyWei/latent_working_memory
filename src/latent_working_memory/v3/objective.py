"""五种 token-memory 写入流程与逐轨迹训练目标。"""

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
        memory = torch.cat(blocks)
        losses = []
        for start in range(0, len(qa_ids), self.cfg.qa_batch_size):
            questions = [
                trajectory.qas[qid] for qid in qa_ids[start : start + self.cfg.qa_batch_size]
            ]
            losses.append(
                self.codec.answer_nll(
                    [memory] * len(questions),
                    [self.prompt_ids(qa.question) for qa in questions],
                    [self.ids(qa.answer_ids) for qa in questions],
                )
            )
        return torch.cat(losses)

    def _write(self, ids, blocks, event):
        with measured(self.device, event, "write_seconds"):
            result = self.codec.compress(self.ids(ids), blocks)
        event["write_calls"] += 1
        return result

    def _states(self, trajectory, epoch=0, force_policy=False):
        blocks = []
        rng = example_rng(self.cfg.seed, epoch, trajectory.trajectory_id)
        single = self.cfg.method == "icae_single"
        for step, segment in enumerate(trajectory.segments):
            if single and step < len(trajectory.segments) - 1:
                continue
            event = {
                "step": step,
                "segment_id": segment.segment_id,
                "action": "initial",
                "slots": 0,
                "write_calls": 0,
                "write_seconds": 0.0,
                "gate_qa_reads": 0,
                "gate_seconds": 0.0,
                "scores": {},
            }
            if single:
                blocks = [self._write(trajectory.full_input_ids, [], event)]
                event["action"] = "single"
            elif self.cfg.method == "icae_multi":
                blocks = blocks + [self._write(segment.input_ids, [], event)]
                event["action"] = "initial" if step == 0 else "append"
            elif self.cfg.method == "autocompressors":
                blocks = blocks + [self._write(segment.input_ids, blocks, event)]
                event["action"] = "initial" if step == 0 else "append"
            elif step == 0:
                blocks = [self._write(segment.input_ids, [], event)]
            else:
                warmup = self.cfg.stage == "warmup" and not force_policy
                if warmup:
                    append = rng.random() < self.cfg.append_probability
                    candidate = self._write(
                        segment.input_ids, [] if append else [blocks[-1]], event
                    )
                else:
                    rewritten = self._write(segment.input_ids, [blocks[-1]], event)
                    if self.cfg.method == "memory_change":
                        with torch.no_grad(), measured(self.device, event, "gate_seconds"):
                            score = float(
                                memory_change_score(blocks[-1], rewritten, self.cfg.rms_epsilon)
                            )
                        event["scores"] = {"I": score}
                        append = score >= self.cfg.threshold_i
                        candidate = (
                            self._write(segment.input_ids, [], event) if append else rewritten
                        )
                    else:
                        appended = self._write(segment.input_ids, [], event)
                        gate_ids = trajectory.usage[step].gate_qa_ids
                        if not gate_ids:
                            raise ValueError(
                                "information-loss updates require historical gate questions"
                            )
                        with torch.no_grad(), measured(self.device, event, "gate_seconds"):
                            l0 = float(self.qa_losses(blocks, trajectory, gate_ids).mean())
                            lrw = float(
                                self.qa_losses(
                                    blocks[:-1] + [rewritten], trajectory, gate_ids
                                ).mean()
                            )
                            lapp = float(
                                self.qa_losses(blocks + [appended], trajectory, gate_ids).mean()
                            )
                        event["gate_qa_reads"] = 3 * len(gate_ids)
                        event["scores"] = {
                            "L0": l0,
                            "Lrw": lrw,
                            "Lapp": lapp,
                            "d": lrw - l0,
                            "g": lrw - lapp,
                        }
                        append = damage_action(
                            l0, lrw, lapp, self.cfg.threshold_d, self.cfg.threshold_g, self.cfg.eta
                        )
                        candidate = appended if append else rewritten
                    # 未选中的候选不进入后续计算图，不添加辅助损失。
                    del rewritten
                    if self.cfg.method == "information_loss":
                        del appended
                blocks = blocks + [candidate] if append else blocks[:-1] + [candidate]
                event["action"] = "append" if append else "overwrite"
            event["slots"] = sum(len(block) for block in blocks)
            yield blocks, event

    def build_memory(self, trajectory, epoch=0, force_policy=True):
        events = []
        for blocks, event in self._states(trajectory, epoch, force_policy):
            events.append(event)
        return blocks, events

    def _qa_objective(self, trajectory, epoch):
        losses, events, old_losses, new_losses = [], [], [], []
        reads, seconds = 0, 0.0
        for blocks, event in self._states(trajectory, epoch):
            events.append(event)
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
            timer = {"read_seconds": 0.0}
            with measured(self.device, timer, "read_seconds"):
                values = self.qa_losses(blocks, trajectory, new_ids + old_ids)
            # 每题先平均答案 token，再以实际题数平均；最后平均更新点。
            losses.append(values.mean())
            new_losses.extend(values[: len(new_ids)].detach().tolist())
            old_losses.extend(values[len(new_ids) :].detach().tolist())
            reads += len(values)
            seconds += timer["read_seconds"]
        metrics = self._memory_metrics(events)
        metrics.update(
            {
                "task_qa_reads": float(reads),
                "read_seconds": seconds,
                "qa_new_nll": sum(new_losses) / len(new_losses),
                "qa_old_nll": sum(old_losses) / len(old_losses) if old_losses else 0.0,
                "qa_new_count": float(len(new_losses)),
                "qa_old_count": float(len(old_losses)),
            }
        )
        return torch.stack(losses).mean(), metrics

    def _pretrain_objective(self, example):
        if self.cfg.method in DYNAMIC_METHODS and len(example.input_ids) > self.cfg.segment_tokens:
            raise ValueError("dynamic pretraining requires a single segment within segment_tokens")
        chunks = [example.input_ids]
        if self.cfg.method == "icae_multi":
            chunks = [
                example.input_ids[i : i + self.cfg.segment_tokens]
                for i in range(0, len(example.input_ids), self.cfg.segment_tokens)
            ]
        blocks = [self.codec.compress(self.ids(chunk)) for chunk in chunks]
        prompt = self.cfg.ae_prompt if example.task == "ae" else self.cfg.lm_prompt
        target = example.target_ids + (self.tokenizer.eos_token_id,)
        loss = self.codec.answer_nll(
            [torch.cat(blocks)],
            [self.ids(self.tokenizer.encode(prompt, add_special_tokens=False))],
            [self.ids(target)],
        )[0]
        return loss, {
            "input_tokens": float(len(example.input_ids)),
            "target_tokens": float(len(target)),
            "slots_final": float(sum(map(len, blocks))),
            "write_calls": float(len(blocks)),
            "ae_samples": float(example.task == "ae"),
            "lm_samples": float(example.task == "continuation"),
        }

    def _ac_objective(self, example, epoch):
        # 每个训练子块最多两段；子块结束后 detach 全部累计摘要。
        # 冻结读取端与写入 LoRA 分开执行，但 LM 只预测尚未进入摘要的原文。
        tokens = (
            example.input_ids if example.task == "ae" else example.input_ids + example.target_ids
        )
        rng = example_rng(self.cfg.seed, epoch, example.sample_id)
        segments, offset = [], 0
        while offset < len(tokens):
            size = rng.randint(self.cfg.ac_min_segment_tokens, self.cfg.ac_max_segment_tokens)
            segments.append(tokens[offset : offset + size])
            offset += size
        if len(segments) < 2 or self.cfg.ac_bptt_steps < 2:
            raise ValueError(
                "frozen-reader AutoCompressors LM needs at least two segments per BPTT group"
            )
        if len(segments) == 2 and len(segments[-1]) < 2:
            raise ValueError(
                "AutoCompressors has no trainable next-token target in the second segment"
            )
        blocks, losses, targets, writes = [], [], 0, 0
        for index, segment in enumerate(segments):
            group_end = (index + 1) % self.cfg.ac_bptt_steps == 0 or index == len(segments) - 1
            target = segment[1:]
            # 原始 next-token 目标跨同一子块的分段边界；子块首 token 不计入损失。
            if not group_end:
                target += segments[index + 1][:1]
            if target:
                memory = torch.cat(blocks) if blocks else self.codec.memory_embeddings[:0]
                nll = self.codec.answer_nll([memory], [self.ids(segment[:1])], [self.ids(target)])[
                    0
                ]
                losses.append(nll * len(target))
                targets += len(target)
            if index < len(segments) - 1:
                blocks.append(self.codec.compress(self.ids(segment), blocks))
                writes += 1
            if group_end:
                blocks = [block.detach() for block in blocks]
        return torch.stack(losses).sum() / targets, {
            "input_tokens": float(len(tokens)),
            "target_tokens": float(targets),
            "slots_final": float(sum(map(len, blocks))),
            "write_calls": float(writes),
            "segments": float(len(segments)),
        }

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

    def forward(self, example, epoch=0, differentiable=True):
        with torch.set_grad_enabled(differentiable):
            if self.cfg.stage == "pretrain":
                loss, metrics = self._pretrain_objective(example)
            elif self.cfg.stage == "lm":
                loss, metrics = self._ac_objective(example, epoch)
            else:
                loss, metrics = self._qa_objective(example, epoch)
        return {"loss": loss, "metrics": metrics}

    def trainable_state_dict(self):
        return self.codec.trainable_state_dict()

    def load_trainable_state_dict(self, state):
        self.codec.load_trainable_state_dict(state)
