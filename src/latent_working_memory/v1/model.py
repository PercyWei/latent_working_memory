from __future__ import annotations

import math

import torch
from torch import Tensor, nn

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

    def forward(self, queries: Tensor, source: Tensor) -> Tensor:
        normalized_queries = self.self_norm(queries)
        attended, _ = self.self_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False,
        )
        queries = queries + attended

        normalized_source = self.cross_source_norm(source)
        attended, _ = self.cross_attention(
            self.cross_query_norm(queries),
            normalized_source,
            normalized_source,
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
        self.initial_seed = nn.Parameter(torch.zeros(d_mem))
        self.initial_projection = nn.Linear(d_mem, d_mem, bias=False)
        self.old_type = nn.Parameter(torch.zeros(d_mem))
        self.new_type = nn.Parameter(torch.zeros(d_mem))
        self.birth_seed = nn.Parameter(torch.zeros(d_mem))
        self.birth_from_input = nn.Linear(d_mem, d_mem)
        self.birth_from_memory = nn.Linear(d_mem, d_mem)
        self.blocks = nn.ModuleList(
            UpdaterBlock(d_mem, num_heads, ffn_dim) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(d_mem)
        self.output_projection = nn.Linear(d_mem, d_mem)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)

    def initialize_state(
        self,
        num_slots: int,
        dtype: torch.dtype | None = None,
    ) -> MemoryState:
        if type(num_slots) is not int or num_slots <= 0:
            raise ValueError("num_slots must be a positive integer")
        if num_slots > self.slot_limit:
            raise ValueError("num_slots must not exceed slot_limit")

        parameter = self.initial_seed
        positions = sinusoidal_positions(
            num_slots,
            self.d_mem,
            parameter.device,
            parameter.dtype,
        )
        values = self.initial_seed + self.initial_projection(positions)
        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError("state dtype must be floating-point")
            values = values.to(dtype=dtype)
        return MemoryState(values, seen_tokens=0)

    def forward(self, state: MemoryState, features: Tensor, grow_by: int) -> MemoryState:
        self._validate_update(state, features, grow_by)
        old_values = state.values
        next_slots = state.num_slots + grow_by
        activation_dtype = old_values.dtype
        source = (
            torch.cat(
                (
                    old_values + self.old_type.to(dtype=activation_dtype),
                    features + self.new_type.to(dtype=activation_dtype),
                ),
                dim=0,
            )
            .to(dtype=activation_dtype)
            .unsqueeze(0)
        )

        if grow_by:
            birth = (
                self.birth_seed.to(dtype=activation_dtype)
                + self.birth_from_input(features.mean(dim=0))
                + self.birth_from_memory(old_values.mean(dim=0))
            ).to(dtype=activation_dtype)
            born_values = birth.unsqueeze(0).expand(grow_by, -1)
            queries = torch.cat((old_values, born_values), dim=0)
        else:
            queries = old_values
        queries = queries + sinusoidal_positions(
            next_slots,
            self.d_mem,
            queries.device,
            queries.dtype,
        )
        queries = queries.unsqueeze(0)

        for block in self.blocks:
            queries = block(queries, source)
        delta = self.output_projection(self.output_norm(queries.squeeze(0))).to(
            dtype=activation_dtype
        )
        base = torch.cat((old_values, old_values.new_zeros(grow_by, self.d_mem)), dim=0)
        return MemoryState(base + delta, state.seen_tokens + features.shape[0])

    def _validate_update(self, state: MemoryState, features: Tensor, grow_by: int) -> None:
        if state.width != self.d_mem:
            raise ValueError(f"state width must be {self.d_mem}")
        if features.ndim != 2 or features.shape[0] <= 0 or features.shape[1] != self.d_mem:
            raise ValueError(f"features must have shape [c, {self.d_mem}] with c > 0")
        if not features.is_floating_point():
            raise TypeError("features must have a floating-point dtype")
        if not bool(torch.isfinite(features.detach()).all()):
            raise ValueError("features must contain only finite values")
        if grow_by not in self.growth_actions:
            raise ValueError(f"grow_by must be one of {self.growth_actions}")
        if state.num_slots + grow_by > self.slot_limit:
            raise ValueError("growth action exceeds slot_limit")
        parameter = self.initial_seed
        if state.values.device != parameter.device or features.device != parameter.device:
            raise ValueError("state, features, and writer must be on the same device")
        if state.values.dtype != features.dtype:
            raise ValueError("state and features must have the same dtype")


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
