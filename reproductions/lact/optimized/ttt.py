"""按官方非 fused 公式维护分块因果 LaCT 的推理状态。"""

from lact_model.ttt_operation import silu_backprop, zeropower_via_newtonschulz5
import torch
import torch.nn.functional as F


# 直接编译各自函数体，避免共用 autocast wrapper 的 code object 和重编译缓存。
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
@torch.compile()
def _ttt_initial_norms(w0, w1, w2):
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)
    return w0_norm, w1_norm, w2_norm


@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
@torch.compile()
def _ttt_read(w0, w1, w2, q):
    h = torch.bmm(w2, q)
    gate = F.silu(torch.bmm(w0, q), inplace=True)
    return torch.bmm(w1, gate * h)


@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
@torch.compile()
def _ttt_commit(
    w0,
    w1,
    w2,
    w0_norm,
    w1_norm,
    w2_norm,
    w0_main,
    w1_main,
    w2_main,
    dw0_momentum,
    dw1_momentum,
    dw2_momentum,
    ki,
    vi,
    lr0i,
    lr1i,
    lr2i,
    momentum,
    prenorm,
    use_muon,
    use_momentum,
):
    gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
    hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
    dhidden = torch.bmm(w1.transpose(1, 2), vi)
    dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

    dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))
    dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
    dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

    if use_momentum:
        m_i = momentum.mean(dim=1, keepdim=True)
        dw0 = dw0 + dw0_momentum * m_i
        dw1 = dw1 + dw1_momentum * m_i
        dw2 = dw2 + dw2_momentum * m_i
        dw0_momentum = dw0
        dw1_momentum = dw1
        dw2_momentum = dw2

    if use_muon:
        dw1 = zeropower_via_newtonschulz5(dw1)
        dw0 = zeropower_via_newtonschulz5(dw0)
        dw2 = zeropower_via_newtonschulz5(dw2)

    if prenorm:
        w1_main = w1_main + dw1
        w0_main = w0_main + dw0
        w2_main = w2_main + dw2
        w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
    else:
        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2
        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    return w0, w1, w2, w0_main, w1_main, w2_main, dw0_momentum, dw1_momentum, dw2_momentum


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
        self.w0_norm, self.w1_norm, self.w2_norm = _ttt_initial_norms(self.w0, self.w1, self.w2)
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
        (
            self.w0,
            self.w1,
            self.w2,
            self.w0_main,
            self.w1_main,
            self.w2_main,
            self.dw0_momentum,
            self.dw1_momentum,
            self.dw2_momentum,
        ) = _ttt_commit(
            self.w0,
            self.w1,
            self.w2,
            self.w0_norm,
            self.w1_norm,
            self.w2_norm,
            self.w0_main,
            self.w1_main,
            self.w2_main,
            self.dw0_momentum,
            self.dw1_momentum,
            self.dw2_momentum,
            ki,
            vi,
            lr0i,
            lr1i,
            lr2i,
            momentum,
            self.prenorm,
            self.use_muon,
            self.use_momentum,
        )
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
            output[:, :, cursor:end] = _ttt_read(self.w0, self.w1, self.w2, qi)

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


class BatchedTTTState:
    """独立真实进度、固定 pending 缓冲与合并的 fast-weight 矩阵运算。"""

    _components = (
        "w0", "w1", "w2", "w0_main", "w1_main", "w2_main",
        "dw0_momentum", "dw1_momentum", "dw2_momentum",
    )
    _pending = ("pending_k", "pending_v", "pending_lr0", "pending_lr1", "pending_lr2", "pending_momentum")

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
    def __init__(self, w0, w1, w2, batch_size, chunk_size, prenorm, use_muon, use_momentum):
        if batch_size <= 0 or chunk_size <= 0:
            raise ValueError("batch_size and chunk_size must be positive")
        self.batch_size, self.heads, self.chunk_size = batch_size, w0.shape[0], chunk_size
        self.prenorm, self.use_muon, self.use_momentum = prenorm, use_muon, use_momentum
        self._rows = list(range(batch_size))
        self.device = w0.device
        self._head_offsets = torch.arange(self.heads, device=self.device)
        norms = _ttt_initial_norms(w0, w1, w2)
        self.w0, self.w1, self.w2 = (value.detach().repeat(batch_size, 1, 1) for value in (w0, w1, w2))
        self.w0_norm, self.w1_norm, self.w2_norm = (value.repeat(batch_size, 1, 1) for value in norms)
        self.w0_main = self.w0 if prenorm else None
        self.w1_main = self.w1 if prenorm else None
        self.w2_main = self.w2 if prenorm else None
        self.dw0_momentum = torch.zeros_like(self.w0) if use_momentum else None
        self.dw1_momentum = torch.zeros_like(self.w1) if use_momentum else None
        self.dw2_momentum = torch.zeros_like(self.w2) if use_momentum else None
        # 某些分量在首轮Momentum更新时升精度；未更新行仍按原dtype参与commit。
        self._dtypes = {name: [getattr(self, name).dtype] * batch_size
                        for name in self._components if getattr(self, name) is not None}
        self.seen = [0] * batch_size
        self.n_updates = [0] * batch_size
        self.pending_length = [0] * batch_size
        for name in self._pending:
            setattr(self, name, None)

    def _indices(self, rows):
        if rows == self._rows:
            return None, None
        indices = torch.tensor(rows, dtype=torch.long, device="cpu").to(self.device, non_blocking=True)
        heads = (indices[:, None] * self.heads + self._head_offsets).reshape(-1)
        return indices, heads

    def _take(self, name, rows, heads, logical_dtype=True):
        value = getattr(self, name)
        if value is None:
            return None
        selected = value if heads is None else value.index_select(0, heads)
        if logical_dtype and name in self._dtypes:
            selected = selected.to(self._dtypes[name][rows[0]])
        return selected

    def _store(self, name, value, rows, heads):
        if value is None:
            return
        if heads is None:
            setattr(self, name, value)
        else:
            current = getattr(self, name)
            dtype = torch.promote_types(current.dtype, value.dtype)
            if current.dtype != dtype:
                current = current.to(dtype)
                setattr(self, name, current)
            elif self.prenorm and name in ("w0", "w1", "w2") and getattr(self, f"{name}_main") is current:
                # 首次只更新部分行时，effective W不能覆盖尚与之共享storage的main。
                current = current.clone()
                setattr(self, name, current)
            current.index_copy_(0, heads, value.to(dtype))
        for row in rows:
            self._dtypes[name][row] = value.dtype

    def _gather(self, values, rows, starts, count):
        if rows == self._rows and len(set(starts)) == 1:
            return [None if value is None else value[:, starts[0]:starts[0] + count] for value in values]
        indices = torch.tensor(rows, dtype=torch.long, device="cpu").to(self.device, non_blocking=True)
        positions = torch.tensor(starts, dtype=torch.long, device="cpu").to(self.device, non_blocking=True)[:, None] + torch.arange(count, device=self.device)
        return [None if value is None else value[indices[:, None], positions] for value in values]

    def _commit_rows(self, rows, values, pending=False):
        groups = {}
        for index, row in enumerate(rows):
            key = tuple(self._dtypes[name][row] for name in self._components if name in self._dtypes)
            groups.setdefault(key, []).append((index, row))
        for group in groups.values():
            selected_rows = [row for _, row in group]
            _, heads = self._indices(selected_rows)
            selected = values
            if len(group) != len(rows):
                indices = torch.tensor([index for index, _ in group], dtype=torch.long, device="cpu").to(self.device, non_blocking=True)
                selected = [None if value is None else value.index_select(0, indices) for value in values]
            if pending:
                # 对齐旧pending的cat布局：K/LR为[H,C,D]连续，V随后transpose。
                selected = [None if value is None else value.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
                            for value in selected]
            count = len(selected_rows) * self.heads
            ki, vi, lr0, lr1, lr2, momentum = selected
            result = _ttt_commit(
                *(self._take(name, selected_rows, heads) for name in ("w0", "w1", "w2")),
                *(self._take(name, selected_rows, heads) for name in ("w0_norm", "w1_norm", "w2_norm")),
                *(self._take(name, selected_rows, heads) for name in self._components[3:]),
                ki.permute(0, 2, 1, 3).reshape(count, self.chunk_size, ki.shape[-1]),
                vi.permute(0, 2, 3, 1).reshape(count, vi.shape[-1], self.chunk_size),
                *(value.permute(0, 2, 1, 3).reshape(count, self.chunk_size, value.shape[-1])
                  for value in (lr0, lr1, lr2)),
                None if momentum is None else momentum.permute(0, 2, 1, 3).reshape(count, self.chunk_size, 1),
                self.prenorm, self.use_muon, self.use_momentum,
            )
            for name, value in zip(self._components, result, strict=True):
                self._store(name, value, selected_rows, heads)
            for row in selected_rows:
                self.n_updates[row] += 1

    def _commit_pending(self, rows):
        indices, _ = self._indices(rows)
        values = [None if getattr(self, name) is None else (
            getattr(self, name) if indices is None else getattr(self, name).index_select(0, indices)
        ) for name in self._pending]
        self._commit_rows(rows, values, pending=True)
        for row in rows:
            self.pending_length[row] = 0

    def _append(self, rows, values, count):
        if self.pending_k is None:
            for name, value in zip(self._pending, values, strict=True):
                if value is not None:
                    setattr(self, name, value.new_empty(self.batch_size, self.chunk_size, self.heads, value.shape[-1]))
        starts = [self.pending_length[row] for row in rows]
        if rows == self._rows and len(set(starts)) == 1:
            for name, value in zip(self._pending, values, strict=True):
                if value is not None:
                    getattr(self, name)[:, starts[0]:starts[0] + count].copy_(value)
        else:
            indices = torch.tensor(rows, dtype=torch.long, device="cpu").to(self.device, non_blocking=True)
            positions = torch.tensor(starts, dtype=torch.long, device="cpu").to(self.device, non_blocking=True)[:, None] + torch.arange(count, device=self.device)
            for name, value in zip(self._pending, values, strict=True):
                if value is not None:
                    getattr(self, name)[indices[:, None], positions] = value
        for row in rows:
            self.pending_length[row] += count

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, q, k, v, lr0, lr1, lr2, momentum, lengths):
        """输入[B,L,H,D]，仅调度每行末尾lengths个真实token。"""
        if self.use_momentum != (momentum is not None):
            raise ValueError("momentum must be supplied exactly when use_momentum is enabled")
        width = q.shape[1]
        if len(lengths) != self.batch_size or any(not 0 <= length <= width for length in lengths):
            raise ValueError("lengths must match the batch and physical input width")
        if momentum is not None:
            momentum = momentum.unsqueeze(-1)
        inputs = (k, v, lr0, lr1, lr2, momentum)
        output = torch.zeros_like(v)
        cursors = [0] * self.batch_size
        active = [row for row, length in enumerate(lengths) if length]
        while active:
            ready = [row for row in active if self.pending_length[row] == self.chunk_size]
            if ready:
                self._commit_pending(ready)
            groups = {}
            for row in active:
                count = min(lengths[row] - cursors[row], self.chunk_size - self.pending_length[row])
                groups.setdefault(count, []).append(row)
            for count, rows in groups.items():
                starts = [width - lengths[row] + cursors[row] for row in rows]
                query, *values = self._gather((q, *inputs), rows, starts, count)
                indices, heads = self._indices(rows)
                result = _ttt_read(
                    *(self._take(name, rows, heads, logical_dtype=False) for name in ("w0", "w1", "w2")),
                    query.permute(0, 2, 3, 1).reshape(len(rows) * self.heads, query.shape[-1], count),
                ).reshape(len(rows), self.heads, v.shape[-1], count).permute(0, 3, 1, 2)
                if indices is None and len(set(starts)) == 1:
                    output[:, starts[0]:starts[0] + count] = result
                else:
                    if indices is None:
                        indices = torch.tensor(rows, dtype=torch.long, device="cpu").to(self.device, non_blocking=True)
                    positions = torch.tensor(starts, dtype=torch.long, device="cpu").to(self.device, non_blocking=True)[:, None] + torch.arange(count, device=self.device)
                    output[indices[:, None], positions] = result
                direct = [index for index, row in enumerate(rows)
                          if count == self.chunk_size and cursors[row] + count < lengths[row]]
                for chosen, commit in ((direct, True), ([index for index in range(len(rows)) if index not in direct], False)):
                    if not chosen:
                        continue
                    selected_rows = [rows[index] for index in chosen]
                    selected = values
                    if len(chosen) != len(rows):
                        selection = torch.tensor(chosen, dtype=torch.long, device="cpu").to(self.device, non_blocking=True)
                        selected = [None if value is None else value.index_select(0, selection) for value in values]
                    if commit:
                        self._commit_rows(selected_rows, selected)
                    else:
                        self._append(selected_rows, selected, count)
                for row in rows:
                    cursors[row] += count
                    self.seen[row] += count
            active = [row for row in active if cursors[row] < lengths[row]]
        return output
