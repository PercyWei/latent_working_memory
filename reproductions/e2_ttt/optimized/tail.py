"""Preserve the native momentum-tail decision separately for each sample."""

import torch

from e2_ttt.ops.e2_ttt_mlp.chunk_momentum import gradient_clip


def momentum_tail_update(
    beta_lr, num_heads, hidden, keys, grad_pre, sbeta, alpha,
    grad_inputs, previous, prod_beta, use_grad_clip,
):
    # Native B1 uses the minimum prunable prefix across its heads. Batch
    # samples must keep that same decision, even when another sample changes.
    rows, length, _ = beta_lr.shape
    batch = rows // num_heads
    prunable = torch.cumprod(beta_lr < 1e-9, dim=1).sum(dim=1)
    prunable = prunable.reshape(batch, num_heads, -1).amin(dim=(1, 2))
    keep = length - prunable
    rounded = torch.pow(2, torch.ceil(torch.log2(keep.clamp_min(1).float()))).long()
    keep = torch.where(keep > 0, rounded.clamp_min(64), 0)
    groups = {}
    for sample, count in enumerate(keep.tolist()):
        groups.setdefault(count, []).append(sample)

    result = [prod_beta * state for state in previous]
    all_rows = torch.arange(rows, device=beta_lr.device).reshape(batch, num_heads)
    for count, samples in groups.items():
        if count == 0:
            continue
        indices = None if len(samples) == batch else all_rows[samples].flatten()

        def select(value):
            return value if indices is None else value.index_select(0, indices)

        h_tail = select(hidden)[..., -count:]
        k_tail = select(keys)[:, -count:]
        ratio = (
            select(sbeta)[:, -count:] / (select(alpha)[:, -count:] + 1e-12)
        ).transpose(1, 2).to(keys.dtype)
        grad_beta = select(grad_pre)[..., -count:] * ratio
        additions = [
            -torch.bmm(select(gradient)[..., -count:] * ratio, k_tail)
            for gradient in grad_inputs
        ]
        additions.insert(1, -torch.bmm(grad_beta, h_tail.transpose(1, 2)))
        if use_grad_clip:
            additions = [gradient_clip(addition, max_norm=1.0) for addition in additions]
        for position, addition in enumerate(additions):
            if indices is None:
                result[position] = result[position] + addition
            else:
                result[position].index_copy_(0, indices, select(result[position]) + addition)
    return tuple(result)
