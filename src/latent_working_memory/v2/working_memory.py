"""v2 整体模型：记忆状态、更新器及 GMSA 读写组合。"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from latent_working_memory.v2.gmsa import GMSA


@dataclass(frozen=True)
class MemoryState:
    values: Tensor
    seen_tokens: int
    updates: int = 0

    def detached(self):
        """Explicit TBPTT boundary; ordinary updates keep the entire state graph."""
        return MemoryState(self.values.detach(), self.seen_tokens, self.updates)


class MemoryUpdater(nn.Module):
    """Gated residual update in native encoder width, with explicit non-negative growth.

    Initial write bypasses this module. Added rows are seeded from current pooled features;
    no operation claims to recover previously discarded historical information.
    """

    def __init__(self, width, heads=8, layers=2, slot_limit=1024):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (width, heads, layers, slot_limit)):
            raise ValueError("updater dimensions and slot_limit must be positive integers")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width, self.slot_limit = width, slot_limit
        self.blocks = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    width,
                    heads,
                    dim_feedforward=2 * width,
                    dropout=0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.old_type = nn.Parameter(torch.zeros(width))
        self.new_type = nn.Parameter(torch.zeros(width))
        self.norm = nn.LayerNorm(width)
        self.delta = nn.Linear(width, width)
        self.gate = nn.Linear(width, width)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2)

    def forward(self, state: MemoryState, current: Tensor, source_tokens: int, grow_by=0):
        if type(grow_by) is not int or grow_by < 0:
            raise ValueError("grow_by must be a non-negative integer")
        if type(source_tokens) is not int or source_tokens < 1:
            raise ValueError("source_tokens must be a positive integer")
        if (
            state.values.ndim != 2
            or current.ndim != 2
            or (state.values.shape[1] != self.width or current.shape[1] != self.width)
            or min(len(state.values), len(current)) < 1
        ):
            raise ValueError("old and new memories must be non-empty native-width matrices")
        if len(state.values) + grow_by > self.slot_limit:
            raise ValueError("growth exceeds slot_limit")
        if grow_by > len(current):
            raise ValueError("new rows cannot exceed current pooled features")
        # Contiguous averaging here only seeds new rows; old state is never pooled with new text.
        added = F.adaptive_avg_pool1d(current.T[None], grow_by)[0].T if grow_by else current[:0]
        base = torch.cat((state.values, added))
        source = torch.cat((state.values + self.old_type, current + self.new_type))[None]
        query = base[None]
        for block in self.blocks:
            query = block(query, source)
        hidden = self.norm(query[0])
        values = base + self.gate(hidden).sigmoid() * self.delta(hidden)
        return MemoryState(values, state.seen_tokens + source_tokens, state.updates + 1)


class WorkingMemory(nn.Module):
    """Frozen GMSA + trainable updater, including a differentiable multi-write QA forward."""

    def __init__(self, backbone: GMSA, updater: MemoryUpdater):
        super().__init__()
        if backbone.width != updater.width:
            raise ValueError("updater must use GMSA's native memory width")
        self.backbone, self.updater = backbone, updater
        self.backbone.set_stage("dynamic")

    def write(self, state, context_ids, ratio, grow_by=0):
        if context_ids.ndim != 1 or len(context_ids) == 0:
            raise ValueError("one write requires a non-empty token vector")
        if type(grow_by) is not int or grow_by < 0:
            raise ValueError("grow_by must be a non-negative integer")
        if state is None and grow_by:
            raise ValueError("first write allocates the native pooled length; grow_by must be zero")
        if state is not None and len(state.values) + grow_by > self.updater.slot_limit:
            raise ValueError("growth exceeds slot_limit")
        # Frozen raw-text encoding does not need an activation graph.
        with torch.no_grad():
            pooled = self.backbone.encode(
                context_ids[None], torch.ones_like(context_ids[None], dtype=torch.bool), ratio
            )[0]
        if state is None:
            if len(pooled) > self.updater.slot_limit:
                raise ValueError("initial pooled memory exceeds slot_limit")
            return MemoryState(pooled, len(context_ids))
        return self.updater(state, pooled, len(context_ids), grow_by)

    def forward(self, segments, ratios, growth, prompt_ids, prompt_mask, labels):
        if not segments or not len(segments) == len(ratios) == len(growth):
            raise ValueError("segments, ratios, growth must be non-empty and aligned")
        state = None
        for segment, ratio, grow_by in zip(segments, ratios, growth, strict=True):
            state = self.write(state, segment, ratio, grow_by)
        # Do not use no_grad: decoder and LSA are frozen, but the updater needs this gradient.
        return self.backbone.read([state.values], prompt_ids, prompt_mask, labels)
