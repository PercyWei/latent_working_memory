"""容量驱动的压缩：连续均值、组内加权、可训练谱域变换。"""

import torch
from torch import nn


class SlotCompression(nn.Module):
    def __init__(self, width, method="mean", bottleneck=256):
        super().__init__()
        if method not in {"mean", "weighted", "spectral"}:
            raise ValueError("compression must be mean, weighted, or spectral")
        self.method = method
        if method == "weighted":
            self.score = nn.Linear(width, 1, bias=False)
            nn.init.zeros_(self.score.weight)
        elif method == "spectral":
            self.before = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, width),
            )
            self.after = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, width),
            )

    def forward(self, hidden, capacity):
        n = len(hidden)
        if hidden.ndim != 2 or not 1 <= capacity <= n:
            raise ValueError("compression requires [tokens, width] and 1 <= K <= tokens")
        if self.method == "spectral":
            # Learned channel mixing precedes global Fourier length reduction.
            features = hidden + self.before(hidden)
            spectrum = torch.fft.rfft(features.float(), dim=0, norm="forward")
            spectrum = spectrum[: capacity // 2 + 1].clone()
            if capacity < n and capacity % 2 == 0:
                # The two frequencies coalesce into the new real Nyquist coefficient.
                spectrum[-1] = 2 * spectrum[-1].real
            reduced = torch.fft.irfft(spectrum, n=capacity, dim=0, norm="forward")
            return reduced + self.after(reduced)
        if self.method == "mean":
            # Reduce each contiguous group directly. Atomic index_add can vary its summation
            # order between forward and checkpoint replay, perturbing recurrent memories.
            boundaries = torch.arange(capacity + 1, device=hidden.device) * n // capacity
            return torch.segment_reduce(hidden.float(), "mean", lengths=boundaries.diff())
        # Inverse of b_j=floor(j*N/K): disjoint groups, including all valid positions.
        groups = ((torch.arange(n, device=hidden.device) + 1) * capacity - 1) // n
        values = hidden.float()
        scores = self.score(hidden).flatten().float()
        maxima = scores.new_full((capacity,), -torch.inf)
        maxima.scatter_reduce_(0, groups, scores.detach(), reduce="amax")
        weights = (scores - maxima[groups]).exp()
        denominator = values.new_zeros(capacity).index_add(0, groups, weights)
        numerator = values.new_zeros(capacity, hidden.shape[-1]).index_add(
            0, groups, values * weights[:, None]
        )
        return numerator / denominator[:, None]
