"""从已到达原文的因果表示读取提交概率。"""

import torch
from torch import nn


class BoundaryHead(nn.Module):
    def __init__(self, width, hidden_size, device):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, hidden_size, device=device, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(hidden_size, 1, device=device, dtype=torch.float32),
        )
        # 首次采样不预设长段或短段偏好；隐藏层仍可获得后续策略梯度。
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, source_hidden):
        return self.network(source_hidden.float()).squeeze(-1)
