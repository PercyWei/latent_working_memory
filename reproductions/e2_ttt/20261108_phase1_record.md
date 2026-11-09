# 20261108_E²-TTT 第一阶段执行记录

创建时间：20261108 17:02:06 Asia/Shanghai

最后修订时间：20261108 18:49:08 Asia/Shanghai

## 执行范围

已迁入固定 revision 的官方完整源码和评估框架，未安装 Linux CUDA 依赖，未下载训练数据或模型权重，未执行 GPU 训练或 benchmark。CPU 检查使用本机 macOS arm64、项目根环境 Python 3.11、PyTorch 2.6.0。

| 检查 | 实际结果 |
|---|---|
| 模型与训练源码 AST 解析 | `e2_ttt/` 和 `training/` 共 58 个 Python 文件通过 |
| Shell 语法 | 本地单卡入口、官方 `train.sh`、官方 `eval.sh` 均通过 `bash -n` |
| 上游源码比对 | 相对于固定 revision 的导出，排除运行产生的 `__pycache__` 后无差异 |
| 必要上游源码的 Git 可跟踪性 | 已用局部 `.gitignore` 处理 `models/`、`docs/` 的泛化忽略；未忽略任何迁入的上游文件 |
| 官方验证 CLI 直接运行 | 本地未安装完整模型依赖，包级导入在缺少 `einops` 时失败，同时报告缺少 `fla`；未改源码绕过模型依赖 |
| 原始标量核与原始验证函数直接加载 | CPU 数值检查通过，具体设置与结果如下 |

## CPU 代数检查

直接加载官方 `e2_ttt/ops/scan_utils.py` 和 `training/verify_state_transition.py`。仅将验证函数的标量核加载入口指向已读取的官方函数，未修改文件、标量核或验证计算。

运行设置：FP32，seed 0，batch 1，128 chunks，chunk size 512，共 65,536 tokens；feature dimension 64、state dimension 8；`eta_base=0.01`、`tau=32.0`、`alpha_base=0.1`，最终相对误差容差 `1e-5`。

```text
End-of-sequence relative error
weight   max=1.204009e-06 mean=1.204009e-06
momentum max=1.754196e-07 mean=1.754196e-07

Chunk-boundary relative error
weight   max=1.991641e-06 mean=1.400498e-06
momentum max=3.078282e-07 mean=1.906486e-07

PASS=True (rtol=1.0e-05)
```

从项目根目录可重跑本次数值检查：

```bash
.venv/bin/python - <<'PY'
import importlib.util
from pathlib import Path
import torch

root = Path('reproductions/e2_ttt/upstream')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

scan = load('e2_scan_original', root / 'e2_ttt/ops/scan_utils.py')
check = load('e2_check_original', root / 'training/verify_state_transition.py')
check.load_build_token_weights_chunk_log = lambda: scan.build_token_weights_chunk_log
torch.set_num_threads(1)
args = check.parse_args()
args.device = 'cpu'
args.feature_dim = 64
args.state_dim = 8
check.verify(args)
PY
```

该检查在给定固定梯度序列的条件下比较 token 递归与段末闭式状态；矩阵维度小于官方 README 的默认值。它没有验证真实 MLP 梯度、GPU kernel、BF16、权重归一化、完整模型训练或论文 benchmark 表现。

## 待执行

用户指定等待 `zcy-memory` 的物理 GPU 1 空闲后运行。目前仅完成代码迁移与本地检查，不占用该卡。

后续使用官方 Linux 环境评估公开的 1.3B MLP 和 SwiGLU checkpoint，运行 `recall ruler`，覆盖所选 SQuAD、SWDE 与原文 RULER 完整范围。运行前保存实际 checkpoint、数据选择、环境和输出目录；下载及评估命令见 [README.md](README.md)。当前没有观测指标，尚不能判断是否复现论文表现。已迁入的 340M 单卡训练入口保留为备用参考。
