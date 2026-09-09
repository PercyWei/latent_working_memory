from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from latent_working_memory.v1.config import GROWTH_ACTIONS
from latent_working_memory.v1.state import MemoryState


def sinusoidal_positions(
    length: int,
    width: int,
    device: torch.device | str,
    dtype: torch.dtype,
    start: int = 0,
) -> Tensor:
    if type(length) is not int or length <= 0:
        raise ValueError("length must be a positive integer")
    if type(width) is not int or width <= 0:
        raise ValueError("width must be a positive integer")
    if type(start) is not int or start < 0:
        raise ValueError("start must be a non-negative integer")

    positions = torch.arange(start, start + length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class UpdaterBlock(nn.Module):
    def __init__(self, d_mem: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(d_mem)
        self.self_attention = nn.MultiheadAttention(
            d_mem,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_query_norm = nn.LayerNorm(d_mem)
        self.cross_source_norm = nn.LayerNorm(d_mem)
        self.cross_attention = nn.MultiheadAttention(
            d_mem,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_mem)
        self.ffn = nn.Sequential(
            nn.Linear(d_mem, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_mem),
        )

    def forward(
        self, queries: Tensor, source: Tensor, query_padding: Tensor, source_padding: Tensor
    ) -> Tensor:
        normalized_queries = self.self_norm(queries)
        attended, _ = self.self_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            key_padding_mask=query_padding,
            need_weights=False,
        )
        queries = queries + attended

        normalized_source = self.cross_source_norm(source)
        attended, _ = self.cross_attention(
            self.cross_query_norm(queries),
            normalized_source,
            normalized_source,
            key_padding_mask=source_padding,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class JointMemoryWriter(nn.Module):
    def __init__(
        self,
        d_mem: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        slot_limit: int = 512,
    ) -> None:
        super().__init__()
        if d_mem <= 0 or num_layers <= 0 or num_heads <= 0 or ffn_dim <= 0:
            raise ValueError("writer dimensions and layer counts must be positive")
        if d_mem % num_heads != 0:
            raise ValueError("d_mem must be divisible by num_heads")
        if type(slot_limit) is not int or slot_limit <= 0:
            raise ValueError("slot_limit must be a positive integer")

        self.d_mem = d_mem
        self.growth_actions = GROWTH_ACTIONS
        self.slot_limit = slot_limit
        self.old_type = nn.Parameter(torch.zeros(d_mem))
        self.new_type = nn.Parameter(torch.zeros(d_mem))
        self.blocks = nn.ModuleList(
            UpdaterBlock(d_mem, num_heads, ffn_dim) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(d_mem)
        self.output_projection = nn.Linear(d_mem, d_mem)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)

    def initialize_state(self, dtype: torch.dtype | None = None) -> MemoryState:
        parameter = self.old_type
        return MemoryState(parameter.new_empty((0, self.d_mem), dtype=dtype), seen_tokens=0)

    def forward(
        self,
        state: MemoryState,
        features: Tensor,
        grow_by: int = 0,
        first_slots: int | None = None,
    ) -> MemoryState:
        return self.update_batch([state], [features], [grow_by], [first_slots])[0]

    def update_batch(
        self,
        states: list[MemoryState],
        features: list[Tensor],
        growth: list[int],
        first_slots: list[int | None],
    ) -> list[MemoryState]:
        if not states or not (len(states) == len(features) == len(growth) == len(first_slots)):
            raise ValueError("states, features, growth and first_slots must align and be non-empty")
        bases, queries, sources = [], [], []
        for state, current, grow_by, first in zip(
            states, features, growth, first_slots, strict=True
        ):
            if (
                state.width != self.d_mem
                or current.ndim != 2
                or current.shape[0] == 0
                or current.shape[1] != self.d_mem
            ):
                raise ValueError("state and non-empty features must match d_mem")
            if (
                state.values.device != self.old_type.device
                or current.device != self.old_type.device
            ):
                raise ValueError("state, features, and writer must be on the same device")
            if state.values.dtype != current.dtype or current.dtype != features[0].dtype:
                raise ValueError("state and features must have the same dtype")
            if not bool(torch.isfinite(current.detach()).all()):
                raise ValueError("features must contain only finite values")
            if state.num_slots == 0:
                if type(first) is not int or not 1 <= first <= self.slot_limit or grow_by != 0:
                    raise ValueError(
                        "empty memory requires first_slots in [1, slot_limit] and growth=0"
                    )
                added = first
            else:
                if (
                    first is not None
                    or type(grow_by) is not int
                    or grow_by not in self.growth_actions
                ):
                    raise ValueError(
                        "existing memory requires a legal growth action and no first_slots"
                    )
                added = grow_by
            if state.num_slots + added > self.slot_limit:
                raise ValueError("growth action exceeds slot_limit")
            base = torch.cat((state.values, current.new_zeros(added, self.d_mem)))
            bases.append(base)
            queries.append(
                base + sinusoidal_positions(len(base), self.d_mem, base.device, base.dtype)
            )
            sources.append(
                torch.cat(
                    (
                        state.values + self.old_type.to(current.dtype),
                        current + self.new_type.to(current.dtype),
                    )
                )
            )
        query_batch = pad_sequence(queries, batch_first=True)
        source_batch = pad_sequence(sources, batch_first=True)
        device = query_batch.device
        query_padding = (
            torch.arange(query_batch.shape[1], device=device)[None, :]
            >= torch.tensor([len(q) for q in queries], device=device)[:, None]
        )
        source_padding = (
            torch.arange(source_batch.shape[1], device=device)[None, :]
            >= torch.tensor([len(s) for s in sources], device=device)[:, None]
        )
        for block in self.blocks:
            query_batch = block(query_batch, source_batch, query_padding, source_padding)
        delta = self.output_projection(self.output_norm(query_batch)).to(features[0].dtype)
        return [
            MemoryState(base + row[: len(base)], state.seen_tokens + len(current))
            for base, row, state, current in zip(bases, delta, states, features, strict=True)
        ]


class GrowthValueNetwork(nn.Module):
    def __init__(
        self,
        d_mem: int = 512,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if d_mem <= 0 or hidden_dim <= 0:
            raise ValueError("d_mem and hidden_dim must be positive")
        self.d_mem = d_mem
        self.growth_actions = GROWTH_ACTIONS
        self.input_dim = 4 * d_mem + 3
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(GROWTH_ACTIONS)),
        )

    def policy_features(self, state: MemoryState, features: Tensor) -> Tensor:
        if state.width != self.d_mem:
            raise ValueError(f"state width must be {self.d_mem}")
        if features.ndim != 2 or features.shape[0] <= 0 or features.shape[1] != self.d_mem:
            raise ValueError(f"features must have shape [c, {self.d_mem}] with c > 0")
        if state.values.device != features.device:
            raise ValueError("state and features must be on the same device")

        if state.num_slots == 0:
            raise ValueError("capacity policy requires allocated memory")
        memory = state.values.detach().to(dtype=torch.float32)
        current = features.detach().to(dtype=torch.float32)
        counts = torch.tensor(
            [state.num_slots, state.seen_tokens, features.shape[0]],
            device=features.device,
            dtype=torch.float32,
        ).log1p()
        return torch.cat(
            (
                memory.mean(dim=0),
                memory.std(dim=0, unbiased=False),
                current.mean(dim=0),
                current.std(dim=0, unbiased=False),
                counts,
            )
        )

    def forward(self, policy_features: Tensor) -> Tensor:
        if policy_features.shape[-1] != self.input_dim:
            raise ValueError(f"policy_features must end in dimension {self.input_dim}")
        return self.network(policy_features.to(dtype=torch.float32))

    def predict(self, state: MemoryState, features: Tensor) -> Tensor:
        return self(self.policy_features(state, features))
