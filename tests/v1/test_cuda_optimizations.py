"""CUDA kernel contracts: nonuniform token weights, frozen heads, and optimizer resume."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.dynamic.config import DynamicConfig
from latent_working_memory.v1.engine import MemoryEngine
from test_reader_projection import make_backbone
from latent_working_memory.v1.backbone import ReadTokens

from latent_working_memory.v1.objectives import chunked_linear_token_nll

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels require a GPU")


@pytest.mark.parametrize("config", [ExperimentConfig, DynamicConfig])
def test_optimizer_flag_is_boolean(config):
    with pytest.raises(ValueError, match="optimizer_fused"):
        config(optimizer_fused="true")


def test_loss_backend_contract():
    with pytest.raises(ValueError, match="reader_loss_backend"):
        ExperimentConfig(reader_loss_backend="unknown")
    with pytest.raises(ValueError, match="CUDA"):
        MemoryEngine(torch.nn.Linear(2, 2), "cpu", 1e-3, 0.01, 1, True)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("train_head", [False, True])
def test_fused_ce_preserves_nonuniform_upstream_gradients(dtype, train_head):
    torch.manual_seed(19)
    x = torch.randn(37, 64, device="cuda", dtype=dtype, requires_grad=True)
    w = (torch.randn(2053, 64, device="cuda", dtype=dtype) / 8).requires_grad_(train_head)
    target = torch.arange(37, device="cuda") * 17
    weights = torch.linspace(0, 1, 37, device="cuda") ** 2 / 37
    result = []
    for fused in (False, True):
        xx, ww = [t.detach().clone().requires_grad_(t.requires_grad) for t in (x, w)]
        loss = (
            chunked_linear_token_nll(xx, ww, target, chunk_size=7)
            if fused else F.cross_entropy(F.linear(xx, ww).float(), target, reduction="none")
        )
        (loss * weights).sum().backward()
        result.append((loss.detach(), xx.grad, ww.grad))
    torch.testing.assert_close(result[0][0], result[1][0], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        result[0][1:], result[1][1:],
        rtol=0.03 if dtype == torch.bfloat16 else 1e-4,
        atol=2e-4 if dtype == torch.bfloat16 else 2e-7,
    )


@cuda
@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_fused_reader_preserves_memory_and_lora_gradients(architecture, checkpointing):
    torch.manual_seed(82)
    baseline = make_backbone(architecture).to("cuda")
    baseline.language_model.to(dtype=torch.bfloat16)
    if checkpointing:
        baseline.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    actual = deepcopy(baseline)
    actual.reader_loss_backend = "liger"
    memories = [torch.randn(n, 8, device="cuda") for n in (2, 5, 3)]
    tasks = [ReadTokens((11,), tuple(range(4, 4 + n)) + (2,)) for n in (2, 4, 7)]
    rows = []
    for model in (baseline, actual):
        mm = [m.clone().requires_grad_() for m in memories]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.read_batch(mm, tasks)
            loss = sum(w * o.mean_nll for w, o in zip((.2, .3, .5), outputs))
        loss.backward()
        rows.append((loss.detach(), [m.grad for m in mm], {
            n: p.grad for n, p in model.named_parameters() if p.requires_grad
        }))
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert any("lora_B" in n and p.grad is not None for n, p in model.named_parameters())
    torch.testing.assert_close(rows[0][0], rows[1][0], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(rows[0][1:], rows[1][1:], rtol=.03, atol=2e-4)


@cuda
def test_fused_adamw_resume():
    torch.manual_seed(2)
    x = torch.nn.Parameter(torch.randn(31, 17, device="cuda"))
    y = torch.nn.Parameter(x.detach().clone())
    base = torch.optim.AdamW([x], lr=3e-5)
    fused = torch.optim.AdamW([y], lr=3e-5, fused=True)
    for step in range(3):
        gradient = torch.randn_like(x)
        x.grad, y.grad = gradient.clone(), gradient.clone()
        base.step()
        fused.step()
        torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-7)
        assert base.state[x]["step"].item() == fused.state[y]["step"].item()
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(base.state[x][key], fused.state[y][key], rtol=1e-5, atol=1e-7)
        if step == 0:
            saved = deepcopy(fused.state_dict())
            fused = torch.optim.AdamW([y], lr=3e-5, fused=True)
            fused.load_state_dict(saved)
            assert fused.param_groups[0]["fused"] is True
