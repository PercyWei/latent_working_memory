"""按官方非 fused 公式维护分块因果 LaCT 的推理状态。"""

from lact_model.ttt_operation import silu_backprop, zeropower_via_newtonschulz5
import torch
import torch.nn.functional as F


class TTTState:
    """缓存已完成 chunk 的状态；满尾段在下一 token 到来时才写入。"""

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
    def __init__(self, w0, w1, w2, chunk_size, prenorm, use_muon, use_momentum):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.prenorm = prenorm
        self.use_muon = use_muon
        self.use_momentum = use_momentum

        self.w0, self.w1, self.w2 = w0.detach(), w1.detach(), w2.detach()
        self.w0_norm = self.w0.norm(dim=2, keepdim=True)
        self.w1_norm = self.w1.norm(dim=2, keepdim=True)
        self.w2_norm = self.w2.norm(dim=2, keepdim=True)
        self.w0_main = self.w0 if prenorm else None
        self.w1_main = self.w1 if prenorm else None
        self.w2_main = self.w2 if prenorm else None

        self.dw0_momentum = torch.zeros_like(self.w0) if use_momentum else None
        self.dw1_momentum = torch.zeros_like(self.w1) if use_momentum else None
        self.dw2_momentum = torch.zeros_like(self.w2) if use_momentum else None

        self.pending_k = []
        self.pending_v = []
        self.pending_lr0 = []
        self.pending_lr1 = []
        self.pending_lr2 = []
        self.pending_momentum = []
        self.pending_length = 0
        self.seen = 0
        self.n_updates = 0

    def _commit(self, ki, vi, lr0i, lr1i, lr2i, momentum):
        gate_before_act = torch.bmm(self.w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(self.w2, ki.transpose(1, 2))
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
        dhidden = torch.bmm(self.w1.transpose(1, 2), vi)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if self.use_momentum:
            m_i = momentum.mean(dim=1, keepdim=True)
            dw0 = dw0 + self.dw0_momentum * m_i
            dw1 = dw1 + self.dw1_momentum * m_i
            dw2 = dw2 + self.dw2_momentum * m_i
            self.dw0_momentum = dw0
            self.dw1_momentum = dw1
            self.dw2_momentum = dw2

        if self.use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)

        if self.prenorm:
            self.w1_main = self.w1_main + dw1
            self.w0_main = self.w0_main + dw0
            self.w2_main = self.w2_main + dw2
            self.w0 = self.w0_main / (self.w0_main.norm(dim=2, keepdim=True) + 1e-5) * self.w0_norm
            self.w1 = self.w1_main / (self.w1_main.norm(dim=2, keepdim=True) + 1e-5) * self.w1_norm
            self.w2 = self.w2_main / (self.w2_main.norm(dim=2, keepdim=True) + 1e-5) * self.w2_norm
        else:
            self.w1 = self.w1 + dw1
            self.w0 = self.w0 + dw0
            self.w2 = self.w2 + dw2
            self.w0 = self.w0 / (self.w0.norm(dim=2, keepdim=True) + 1e-5) * self.w0_norm
            self.w1 = self.w1 / (self.w1.norm(dim=2, keepdim=True) + 1e-5) * self.w1_norm
            self.w2 = self.w2 / (self.w2.norm(dim=2, keepdim=True) + 1e-5) * self.w2_norm

        self.n_updates += 1

    def _commit_pending(self):
        self._commit(
            torch.cat(self.pending_k, dim=1),
            torch.cat(self.pending_v, dim=1).transpose(1, 2),
            torch.cat(self.pending_lr0, dim=1),
            torch.cat(self.pending_lr1, dim=1),
            torch.cat(self.pending_lr2, dim=1),
            torch.cat(self.pending_momentum, dim=1) if self.use_momentum else None,
        )
        self.pending_k.clear()
        self.pending_v.clear()
        self.pending_lr0.clear()
        self.pending_lr1.clear()
        self.pending_lr2.clear()
        self.pending_momentum.clear()
        self.pending_length = 0

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, q, k, v, lr0, lr1, lr2, momentum):
        """接收 (B×heads, L, D) 表征，返回当前片段的原 dtype 读出。"""
        if self.use_momentum != (momentum is not None):
            raise ValueError("momentum must be supplied exactly when use_momentum is enabled")
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        output = torch.zeros_like(v)
        seq_len = k.shape[1]
        cursor = 0
        while cursor < seq_len:
            if self.pending_length == self.chunk_size:
                self._commit_pending()

            direct_commit = self.pending_length == 0 and seq_len - cursor > self.chunk_size
            end = min(seq_len, cursor + self.chunk_size - self.pending_length)
            qi = q[:, :, cursor:end]
            h = torch.bmm(self.w2, qi)
            gate = F.silu(torch.bmm(self.w0, qi), inplace=True)
            output[:, :, cursor:end] = torch.bmm(self.w1, gate * h)

            if direct_commit:
                self._commit(
                    k[:, cursor:end, :],
                    v[:, :, cursor:end],
                    lr0[:, cursor:end, :],
                    lr1[:, cursor:end, :],
                    lr2[:, cursor:end, :],
                    momentum[:, cursor:end, :] if self.use_momentum else None,
                )
            else:
                # 独立保存尾段，避免视图长期持有完整 prefill 的底层张量。
                self.pending_k.append(k[:, cursor:end, :].clone())
                self.pending_v.append(v[:, :, cursor:end].transpose(1, 2).clone())
                self.pending_lr0.append(lr0[:, cursor:end, :].clone())
                self.pending_lr1.append(lr1[:, cursor:end, :].clone())
                self.pending_lr2.append(lr2[:, cursor:end, :].clone())
                if self.use_momentum:
                    self.pending_momentum.append(momentum[:, cursor:end, :].clone())
                self.pending_length += end - cursor
            length = end - cursor
            self.seen += length
            cursor = end
        return output.transpose(1, 2)
