"""逐层 token memory：共享局部 reader、压缩查询与可微内循环。"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerMemory(nn.Module):
    """慢参数定义读写接口，显式传入的 slots 是每个序列的 fast state。

    模块默认使用 fp32；主模型的 autocast 不改变局部读写精度。
    ``double()`` 可用于数值梯度检查。内循环只求 slots 的偏导，不写入参数的 ``.grad``。
    """

    def __init__(self, hidden_size, q_output_size, config):
        super().__init__()
        self.query_dim = config.query_dim
        self.query_mode = config.query_mode
        self.query_normalization = config.query_normalization
        self.read_query_norm = config.read_query_norm
        self.compression_query_norm = config.compression_query_norm
        self.inner_steps = config.inner_steps
        self.inner_lr = config.inner_lr
        self.correction_scale = config.correction_scale

        self.source_norm = nn.LayerNorm(hidden_size)
        self.source_projection = nn.Linear(hidden_size, config.memory_dim, bias=False)
        self.runtime_query = nn.Linear(hidden_size, config.query_dim, bias=False)
        self.key_projection = nn.Linear(config.memory_dim, config.query_dim, bias=False)
        self.value_projection = nn.Linear(config.memory_dim, config.value_dim, bias=False)
        self.query_correction = nn.Linear(config.value_dim, q_output_size, bias=False)
        self.output_correction = nn.Linear(config.value_dim, hidden_size, bias=False)

        # 不同的初始 slots 打破共享 reader 的置换对称性。
        self.initial_slots = nn.Parameter(
            torch.randn(config.num_slots, config.memory_dim) / math.sqrt(config.memory_dim)
        )
        self.probe_seeds = nn.Parameter(
            torch.randn(config.num_probes, config.query_dim) / math.sqrt(config.query_dim)
        )
        if self.query_mode == "conditioned":
            self.probe_projection = nn.Linear(config.memory_dim, config.query_dim, bias=False)

    def _normalize_query(self, query: Tensor, target_norm: float) -> Tensor:
        if self.query_normalization == "none":
            return query
        return F.normalize(query, dim=-1) * target_norm

    def source(self, hidden: Tensor) -> Tensor:
        """将待压缩段的层输入映射到与 slots 相同的 memory 坐标系。"""
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.to(dtype=self.initial_slots.dtype)
            return self.source_projection(self.source_norm(hidden))

    def read(self, query: Tensor, bank: Tensor) -> Tensor:
        """Q:[batch, probes, dq]，bank:[batch, tokens, dm]；返回局部响应。"""
        with torch.autocast(device_type=bank.device.type, enabled=False):
            query = query.to(dtype=self.initial_slots.dtype)
            bank = bank.to(dtype=self.initial_slots.dtype)
            keys = self.key_projection(bank)
            values = self.value_projection(bank)
            scores = query @ keys.transpose(-1, -2) / math.sqrt(self.query_dim)
            # 显式 matmul/softmax，避免 fused attention 不支持 double backward。
            return scores.softmax(dim=-1) @ values

    def queries(self, source: Tensor) -> Tensor:
        """每次写入只生成一次压缩查询；默认不限制范数。"""
        with torch.autocast(device_type=source.device.type, enabled=False):
            source = source.to(dtype=self.initial_slots.dtype)
            seeds = self._normalize_query(self.probe_seeds, self.compression_query_norm)
            seeds = seeds.unsqueeze(0).expand(source.shape[0], -1, -1)
            if self.query_mode == "conditioned":
                projected = self.probe_projection(source)
                scores = seeds @ projected.transpose(-1, -2) / math.sqrt(self.query_dim)
                seeds = seeds + scores.softmax(dim=-1) @ projected
                seeds = self._normalize_query(seeds, self.compression_query_norm)
            return seeds

    def initial_state(self, batch_size, differentiable):
        state = self.initial_slots.unsqueeze(0).expand(batch_size, -1, -1).clone()
        return state if differentiable else state.detach()

    def corrections(self, hidden: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        """先读取旧 memory，再给主模型提供 query/output 两条修正。"""
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            normalized = self.source_norm(hidden.to(dtype=self.initial_slots.dtype))
            query = self._normalize_query(self.runtime_query(normalized), self.read_query_norm)
            response = self.read(query, memory)
            delta_q = self.correction_scale * self.query_correction(response)
            delta_o = self.correction_scale * self.output_correction(response)
            return delta_q.to(dtype=hidden.dtype), delta_o.to(dtype=hidden.dtype)

    @staticmethod
    def _write_losses(response: Tensor, target: Tensor) -> Tensor:
        # 每样本对 probes 平均、对 value 维求和。更新步长不依赖 batch 大小。
        return 0.5 * (response - target).square().sum(dim=-1).mean(dim=-1)

    def write(self, memory: Tensor, source: Tensor, differentiable):
        """执行固定步数的局部 TTT，返回新状态及 detached 标量诊断。

        内循环对 state 求偏导，Q/Y 固定；外循环保留 teacher 对 Q 的梯度。
        ``differentiable=False`` 在调用方 ``no_grad`` 下也能执行，只返回 detached state。
        """
        with torch.autocast(device_type=memory.device.type, enabled=False):
            with torch.set_grad_enabled(differentiable):
                query = self.queries(source)
                target = self.read(query, source)

            with torch.enable_grad():
                initial = memory if differentiable else memory.detach()
                # source 可能依赖旧 memory；独立 candidate 保证这里求的是固定 source 的偏导。
                # clone 保留外层经过旧 memory/source 的完整梯度路径。
                state = initial.to(dtype=self.initial_slots.dtype).clone()
                state.requires_grad_(True)
                before = None
                for _ in range(self.inner_steps):
                    losses = self._write_losses(self.read(query, state), target)
                    if before is None:
                        before = losses.detach().mean()
                    gradient = torch.autograd.grad(
                        losses.sum(), state, create_graph=differentiable
                    )[0]
                    state = state - self.inner_lr * gradient
                    if not differentiable:
                        state = state.detach().requires_grad_(True)

            with torch.no_grad():
                after = self._write_losses(self.read(query, state), target).mean()
                metrics = {"write_loss_before": before, "write_loss_after": after}
            return (state if differentiable else state.detach()), metrics
