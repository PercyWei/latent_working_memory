"""QA 与 benchmark 共用的 checkpoint 模型加载和局部写入计时。"""

from dataclasses import replace
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from latent_working_memory.v4.config import ModelConfig
from latent_working_memory.v4.model import StreamingMemoryLM


class TimedStreamingMemoryLM(StreamingMemoryLM):
    """只在评估计量边界写入；CUDA events 同步后统一结算。"""

    def __init__(self, config):
        super().__init__(config)
        self.evaluation_phase = "nll"
        self.write_measurements = []

    def _write(self, state, differentiable, active_rows=None):
        before = state.write_events
        use_cuda = state.memories[0].device.type == "cuda"
        if use_cuda:
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
        else:
            started = time.perf_counter()
        super()._write(state, differentiable, active_rows)
        if use_cuda:
            ended.record()
        else:
            ended = time.perf_counter()
        self.write_measurements.append(
            (self.evaluation_phase, started, ended, state.write_events - before)
        )

    def write_timing(self, device):
        phases = {name: {"write_seconds": 0.0, "write_events": 0} for name in ("nll", "generation")}
        for phase, started, ended, events in self.write_measurements:
            elapsed = (
                started.elapsed_time(ended) / 1000 if device.type == "cuda" else ended - started
            )
            phases[phase]["write_seconds"] += elapsed
            phases[phase]["write_events"] += events
        return {
            **phases,
            "total": {
                name: sum(phase[name] for phase in phases.values())
                for name in ("write_seconds", "write_events")
            },
            "measurement": "CUDA events" if device.type == "cuda" else "CPU wall time",
            "scope": "boundary writes: reader targets, local SGD and cache trimming",
        }


def load_evaluation_model(checkpoint, device):
    """沿用训练时的基座 revision、tokenizer 和记忆读写参数。"""
    run = checkpoint["run"]
    config = ModelConfig(**run["model"])
    config = replace(config, revision=run["resolved_model_revision"] or config.revision)
    tokenizer_path = Path(run["training"]["output_dir"]).expanduser().resolve() / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = TimedStreamingMemoryLM(config).to(device).eval()
    model.load_trainable_state_dict(checkpoint["memory"])
    return model, tokenizer, config
