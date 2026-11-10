"""Local reference mlp TTT kernels with isolated batch state.

Derived from upstream/e2_ttt/ops/e2_ttt_mlp/chunk_momentum.py. The
closed-form update and cache tuple stay native. Norm parameters expand before
constructing decode targets; momentum tail pruning is per original sample.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from e2_ttt.ops.e2_ttt_mlp.chunk_momentum import gelu_derivative, gradient_clip, ln_reconstruction_target
from e2_ttt.ops.norm_utils import layernorm_fwd, layernorm_bwd
from e2_ttt.ops.scan_utils import build_token_weights_chunk_log

from .tail import momentum_tail_update


def update_chunk_momentum(
    q_chunk, k_chunk, v_chunk,  # [b, d, seq_len]
    lr_chunk, log_decay_chunk, log_beta_chunk,  # [b, seq_len, 1]
    w0, w1,  # w0: [b, dh, d]; w1: [b, d, dh]
    m_init_w0, m_init_w1,
    w0_norm, w1_norm,
    ttt_norm_weight, ttt_norm_bias,
    num_heads,
    verbose=False,
    use_grad_clip=True,
):
    """
    Standalone TTT update function for 2-layer MLP: x + LN(w1 @ GeLU(w0 @ x))
    """
    # Retrieve memory (Read out)
    # layernorm + residual will be done once after all chunks are processed
    h = F.gelu(torch.bmm(w0, q_chunk))
    out = torch.bmm(w1, h)

    # Forward
    # MLPs
    Z = torch.bmm(w0, k_chunk)  # [b, dh, seq_len]
    H = F.gelu(Z)  # [b, dh, seq_len]
    pre = torch.bmm(w1, H).float()  # we use residual here  [b, d, seq_len]
    # RMSNorm + Residual (Residual in already included in the v_chunk, see self.ln_restruction_target)
    normalized, mu, inv_s = layernorm_fwd(pre, ttt_norm_weight, ttt_norm_bias)

    # MSE loss
    Err = normalized - v_chunk  # [b, d, seq_len]

    log_alpha, sbeta, carry_u, carry_m = build_token_weights_chunk_log(log_beta_chunk, log_decay_chunk)
    carry_u, carry_m = carry_u.to(k_chunk.dtype), carry_m.to(v_chunk.dtype)
    alpha = torch.exp(log_alpha)

    alpha_lr = (alpha * lr_chunk).to(k_chunk.dtype)
    Err_alpha = Err * alpha_lr.transpose(1, 2)

    # LayerNorm backward
    grad_pre = layernorm_bwd(pre, Err_alpha, ttt_norm_weight, mu, inv_s)
    grad_H = torch.bmm(w1.transpose(1, 2), grad_pre)  # [b, dh, seq_len]
    gelu_prime_val = gelu_derivative(Z)
    grad_Z = grad_H * gelu_prime_val

    dw1 = -torch.bmm(grad_pre, H.transpose(1, 2))
    k_t = k_chunk.transpose(1, 2)  # [b, seq_len, d]
    dw0 = -torch.bmm(grad_Z, k_t)

    if verbose:
        rel_step = dw0.norm() / w0.norm()
        print(f"before clipping rel_step: {rel_step}, dw0 norm {dw0.norm()}, w0 norm {w0.norm()}")
        print(f"carry_u: {carry_u[:, 0, 0].tolist()}")

    # gradient clip
    if use_grad_clip:
        dw0 = gradient_clip(dw0)
        dw1 = gradient_clip(dw1)

    momentum_w0 = carry_m * m_init_w0
    momentum_w1 = carry_m.transpose(1, 2) * m_init_w1

    # We now calculate the weights for the last token in the chunk
    upd_w0 = carry_u * w0 + momentum_w0 + dw0
    upd_w1 = carry_u * w1 + momentum_w1 + dw1

    if verbose:
        rel_step = dw0.norm() / w0.norm()
        print(f"rel_step: {rel_step}, dw0 norm {dw0.norm()}, w0 norm {w0.norm()}")
        print(f"MSE: {Err.pow(2).mean()}")

    # Weight normalization (same as in test training done right)
    upd_w0 = upd_w0 / (upd_w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
    upd_w1 = upd_w1 / (upd_w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm

    # We now calculate the momentum of the last token for the next chunk
    prod_beta = log_beta_chunk.sum(dim=1, keepdim=True).exp().to(k_chunk.dtype)  # [b,1,1]
    beta_lr = (sbeta * lr_chunk).to(k_chunk.dtype)

    # The native single-sample reduction spans heads, never other samples.
    m_next_w0, m_next_w1 = momentum_tail_update(
        beta_lr, num_heads, H, k_t, grad_pre, sbeta, alpha,
        (grad_Z,), (m_init_w0, m_init_w1), prod_beta,
        use_grad_clip=use_grad_clip,
    )

    return out, upd_w0, upd_w1, m_next_w0, m_next_w1


def chunk_ttt_momentum_with_cache(
    q, k, v,  # [b, seq_len, h, d]
    lr, log_decay, log_momentum,  # [b, seq_len, h, 1]
    recurrent_state,  # w0: [h, dh, d], w1: [h, d, dh], buf_k, buf_v, buf_lr
    ttt_norm_weight, ttt_norm_bias,  # [h, d]
    chunk_size=1024,
    verbose=False,
    use_closed_form=True,
    use_grad_clip=True,
):
    b, seq_len, h, d = q.shape
    assert seq_len == 1, 'This function is designed for token-by-token generation.'

    # We move head dimension into batch dimension
    q, k, v, lr = [
        rearrange(x, 'b seq_len h d -> (b h) seq_len d') for x in [q, k, v, lr]
    ]
    log_decay = rearrange(log_decay, 'b seq_len h d -> (b h) seq_len d') if log_decay is not None else None
    log_momentum = rearrange(log_momentum, 'b seq_len h d -> (b h) seq_len d')

    # expand weights
    ttt_norm_weight = repeat(ttt_norm_weight, 'h d -> (b h) d', b=b)
    ttt_norm_bias = repeat(ttt_norm_bias, 'h d -> (b h) d', b=b)

    v = ln_reconstruction_target(v, k, ttt_norm_weight, ttt_norm_bias)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    # Read from cache
    w0, w1, buf_k, buf_v, buf_lr, buf_log_decay, buf_log_momentum, m_next_w0, m_next_w1 = recurrent_state

    if w0.shape[0] < b * h:
        w0 = repeat(w0, 'h dh d -> (b h) dh d', b=b)
        w1 = repeat(w1, 'h d dh -> (b h) d dh', b=b)

    # cache new k, v, and others
    buf_k_new = torch.cat([buf_k, k], dim=-1)
    buf_v_new = torch.cat([buf_v, v], dim=-1)
    buf_lr_new = torch.cat([buf_lr, lr], dim=1)
    buf_log_decay_new = torch.cat([buf_log_decay, log_decay], dim=1) if log_decay is not None else None
    buf_log_momentum_new = torch.cat([buf_log_momentum, log_momentum], dim=1)

    # We only cache new k,v tokens without updating the memory
    if buf_k_new.shape[-1] < chunk_size:
        # Forward with cached weights
        h = F.gelu(torch.bmm(w0, q))
        out = torch.bmm(w1, h)  # [b d seq_len]
        x = out.transpose(1, 2)  # [b, seq_len, d]
        x_norm = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-8)
        out = x_norm * ttt_norm_weight.unsqueeze(1).float() + ttt_norm_bias.unsqueeze(1).float()
        out = out.to(q.dtype) + q.transpose(1, 2)
        out = rearrange(out, '(b h) seq_len d -> b seq_len h d', b=b)

        return out, (w0, w1, buf_k_new, buf_v_new, buf_lr_new, buf_log_decay_new, buf_log_momentum_new, m_next_w0, m_next_w1)

    w0_norm = w0.norm(dim=-1, keepdim=True)
    w1_norm = w1.norm(dim=-1, keepdim=True)

    update_fn = update_chunk_momentum
    out, w0, w1, m_next_w0, m_next_w1 = update_fn(
        q, buf_k_new, buf_v_new,
        buf_lr_new, buf_log_decay_new, buf_log_momentum_new,
        w0, w1,  # w0: [b, dh, d]; w1: [b, d, dh]
        m_next_w0, m_next_w1,
        w0_norm, w1_norm,
        ttt_norm_weight.unsqueeze(-1), ttt_norm_bias.unsqueeze(-1),
        num_heads=h,
        verbose=verbose,
        use_grad_clip=use_grad_clip,
    )

    # Do the layernorm + residual once
    x = out.transpose(1, 2)  # [b, seq_len, d]
    x_norm = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-8)
    out = x_norm * ttt_norm_weight.unsqueeze(1).float() + ttt_norm_bias.unsqueeze(1).float()
    out = out.to(q.dtype) + q.transpose(1, 2)
    out = rearrange(out, '(b h) seq_len d -> b seq_len h d', b=b)

    # Clear the cache
    buf_k = q.new_zeros(b * h, d, 0)
    buf_v = q.new_zeros(b * h, d, 0)
    buf_lr = q.new_zeros(b * h, 0, lr.shape[-1])
    buf_log_decay = q.new_zeros(b * h, 0, log_decay.shape[-1]) if log_decay is not None else None
    buf_log_momentum = q.new_zeros(b * h, 0, log_momentum.shape[-1])

    return out, (w0, w1, buf_k, buf_v, buf_lr, buf_log_decay, buf_log_momentum, m_next_w0, m_next_w1)


def chunk_ttt_momentum(
    q, k, v,  # [b, seq_len, h, d]
    lr, log_decay, log_momentum,  # [b, seq_len, h, 1]
    initial_state,  # w0: [h, dh, d], w1: [h, d, dh]
    ttt_norm_weight, ttt_norm_bias,  # [h, d]
    chunk_size=1024,
    verbose=False,
    is_training=True,
    use_closed_form=True,
    use_grad_clip=True,
):
    b, seq_len, h, d = q.shape
    rem = seq_len % chunk_size

    w0, w1 = initial_state

    # We move head dimension into batch dimension
    q, k, v, lr = [
        rearrange(x, 'b seq_len h d -> (b h) seq_len d') for x in [q, k, v, lr]
    ]
    log_decay = rearrange(log_decay, 'b seq_len h d -> (b h) seq_len d') if log_decay is not None else None
    log_momentum = rearrange(log_momentum, 'b seq_len h d -> (b h) seq_len d')

    # expand weights
    w0 = repeat(w0, 'h dh d -> (b h) dh d', b=b)
    w1 = repeat(w1, 'h d dh -> (b h) d dh', b=b)
    ttt_norm_weight = repeat(ttt_norm_weight, 'h d -> (b h) d', b=b)
    ttt_norm_bias = repeat(ttt_norm_bias, 'h d -> (b h) d', b=b)

    v = ln_reconstruction_target(v, k, ttt_norm_weight, ttt_norm_bias)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    w0_norm = w0.norm(dim=-1, keepdim=True)
    w1_norm = w1.norm(dim=-1, keepdim=True)

    m_next_w0 = torch.zeros_like(w0)
    m_next_w1 = torch.zeros_like(w1)

    q_chunks, k_chunks, v_chunks = [
        x.split(chunk_size, dim=-1) for x in [q, k, v]
    ]
    num_chunks = len(q_chunks)

    lr_chunks = lr.split(chunk_size, dim=1)
    log_decay_chunks = log_decay.split(chunk_size, dim=1) if log_decay is not None else [None] * num_chunks
    log_momentum_chunks = log_momentum.split(chunk_size, dim=1)

    s_index = 0
    output = torch.zeros_like(q)
    for i in range(num_chunks):
        e_index = s_index + chunk_size
        # We do not update the last chunk, as the updated weights do not contribute to the final loss
        if i == num_chunks - 1:
            if is_training or rem != 0:
                continue

        if verbose:
            print(f"{i + 1}-th chunk")

        update_fn = update_chunk_momentum
        out, w0, w1, m_next_w0, m_next_w1 = update_fn(
            q_chunks[i], k_chunks[i], v_chunks[i],
            lr_chunks[i], log_decay_chunks[i], log_momentum_chunks[i],
            w0, w1,  # w0: [b, dh, d]; w1: [b, d, dh]
            m_next_w0, m_next_w1,
            w0_norm, w1_norm,
            ttt_norm_weight.unsqueeze(-1), ttt_norm_bias.unsqueeze(-1),
            num_heads=h,
            verbose=verbose,
            use_grad_clip=use_grad_clip,
        )

        output[..., s_index:e_index] = out
        s_index = e_index

    # processing last chunk, direct read out
    if is_training or rem != 0:
        q_chunk = q_chunks[-1]
        hidden = F.gelu(torch.bmm(w0, q_chunk))
        tail = torch.bmm(w1, hidden)
        output[:, :, s_index:] = tail

    # Do the layernorm + residual once
    x = output.transpose(1, 2)  # [b, seq_len, d]
    x_norm = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-8)
    out = x_norm * ttt_norm_weight.unsqueeze(1).float() + ttt_norm_bias.unsqueeze(1).float()
    out = out.to(q.dtype) + q.transpose(1, 2)

    out = rearrange(out, '(b h) seq_len d -> b seq_len h d', b=b)

    # Prepare buf
    if rem != 0:
        buf_k = k_chunks[-1]
        buf_v = v_chunks[-1]
        buf_lr = lr_chunks[-1]
        buf_log_decay = log_decay_chunks[-1]
        buf_log_momentum = log_momentum_chunks[-1]
    else:
        buf_k = q.new_zeros(b * h, d, 0)
        buf_v = q.new_zeros(b * h, d, 0)
        buf_lr = q.new_zeros(b * h, 0, lr.shape[-1])
        buf_log_decay = q.new_zeros(b * h, 0, log_decay.shape[-1]) if log_decay is not None else None
        buf_log_momentum = q.new_zeros(b * h, 0, log_momentum.shape[-1])

    return out, (w0, w1, buf_k, buf_v, buf_lr, buf_log_decay, buf_log_momentum, m_next_w0, m_next_w1)
