# 20261108_LaCT 本地接入执行记录

创建时间：20261108 17:03:39 Asia/Shanghai

最后修订时间：20261108 17:06:26 Asia/Shanghai

## 执行范围与结果

执行环境为 macOS arm64、Python 3.11、PyTorch 2.6.0，`torch.cuda.is_available()` 为 `False`。当前用户安排先完成代码迁移；GPU 任务仅允许在 `zcy-memory` 物理 GPU 1 空闲时执行，本次未启动 GPU 任务。

已完成：

1. 从官方 GitHub 获取 LaCT 和其指定 flame trainer，固定 revision 并迁入源码；来源见 [UPSTREAM.md](UPSTREAM.md)。
2. 对迁入的 52 个原始 Python 文件进行 `ast.parse`，全部通过；对五份官方模型 JSON 执行 JSON 解析，全部通过。新增注册文件后共 53 个 Python 文件通过解析；单卡 TOML 的格式和 batch 设置检查通过，启动脚本通过 `bash -n`。
3. 直接加载官方 `lact_llm/lact_model/ttt_operation.py`，运行 `block_causal_lact_swiglu` 的 GD、Momentum、Muon 三个分支。使用 batch 1、序列长 12、chunk 4、向量维度 8、seed 42、CPU float32，禁用 TorchDynamo 编译。
4. 三个分支输出均为 `(1,12,8)`，输出及初始 fast weights、Q/K/V 的外层梯度均有限。改变第二个 chunk 的 values 后，前两个 chunk 输出保持完全相同，第三个 chunk 输出发生变化，符合该实现“使用旧状态读出、段末更新供后续段使用”的时序。

检查直接执行官方函数，未使用替代 attention 或伪造 CUDA/FLA 模块。由于未装 GPU 依赖且无 CUDA，未执行完整 `LaCTForCausalLM`、FlashAttention、Triton kernel 或 flame 分布式训练。

CPU 检查的调用结构如下；`TORCHDYNAMO_DISABLE=1` 仅为本次本地检查，不写入正式训练入口：

```bash
TORCHDYNAMO_DISABLE=1 .venv/bin/python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('reproductions/lact/upstream/lact_llm/lact_model/ttt_operation.py')
spec = importlib.util.spec_from_file_location('lact_official_ttt_operation', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.manual_seed(42)
for name, muon, momentum_enabled in [('gd', False, False), ('momentum', False, True), ('muon', True, True)]:
    weights = [torch.randn(1, 8, 8, requires_grad=True) for _ in range(3)]
    q, k, v = [torch.randn(1, 12, 8, requires_grad=True) for _ in range(3)]
    rates = [torch.full((1, 12, 1), 0.001) for _ in range(3)]
    momentum = torch.full((1, 12, 1), 0.9) if momentum_enabled else None
    output = module.block_causal_lact_swiglu(*weights, q, k, v, *rates, chunk_size=4, use_muon=muon, momentum=momentum)
    output.square().mean().backward()
    changed_v = v.detach().clone()
    changed_v[:, 4:] += 10
    changed = module.block_causal_lact_swiglu(*weights, q, k, changed_v, *rates, chunk_size=4, use_muon=muon, momentum=momentum)
    assert torch.isfinite(output).all()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in [*weights, q, k, v])
    assert torch.equal(output[:, :8], changed[:, :8])
    assert not torch.equal(output[:, 8:], changed[:, 8:])
    print(name, tuple(output.shape), 'forward/backward finite; chunk timing passed')
PY
```

这只验证本地写入函数和时序，没有验证训练质量、完整模型逐 token 解码或论文指标。
