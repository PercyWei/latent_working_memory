# 20261109_TTT checkpoint 评估环境

创建时间：20261109 10:58:35 Asia/Shanghai

最后修订时间：20261109 11:35:08 Asia/Shanghai

## 配置原则与实际版本

三个 checkpoint 评估环境已安装在 `zcy-memory`，主项目 `.venv` 保持不变。原研究固定版本优先；未固定依赖尽量采用其他方法已确定且兼容的版本。本轮只安装 checkpoint 评估依赖，训练专用的 flame／torchtitan／VeOmni 等组件不在本轮范围内。

下表为服务器实际安装版本；“—”表示该评估路径不需要且未安装。CUDA runtime 指 PyTorch 使用的运行时，区别于服务器共用的 CUDA Toolkit 12.8（nvcc V12.8.93）。

| 项目 | 主环境 `.venv` | LaCT | E²-TTT | In-Place TTT |
|---|---|---|---|---|
| Python | 3.11.15 | 3.11.15 | 3.12.3 | 3.11.15 |
| PyTorch | 2.6.0 cu124 | 2.6.0 cu124 | 2.8.0 cu128 | 2.8.0 cu128 |
| CUDA runtime | 12.4 | 12.4 | 12.8 | 12.8 |
| Triton | 3.2.0 | 3.2.0 | 3.4.0 | 3.4.0 |
| FlashAttention | — | 2.7.4.post1 | 2.8.3 | 2.8.3 |
| Torch／FlashAttention CXX11 ABI | FALSE | FALSE | TRUE | TRUE |
| Flash Linear Attention（FLA） | — | 0.4.1 | 0.4.1 | — |
| Transformers | 4.57.6 | 4.57.3 | 4.57.1 | 4.57.3 |
| NumPy | 1.26.4 | 1.26.4 | 2.3.3 | 1.26.4 |
| datasets | 5.0.1 | — | 4.2.0 | 3.6.0 |
| Accelerate | 1.14.0 | — | 1.10.1 | 1.10.1 |
| huggingface-hub | 0.36.2 | 0.36.2 | 0.36.2 | 0.36.2 |
| tokenizers | 0.22.2 | 0.22.2 | 0.22.2 | 0.22.2 |
| safetensors | 0.8.0 | 0.8.0 | 0.8.0 | 0.8.0 |
| 评估框架 | 项目代码 | 本地 RULER 入口 | 上游随附 lm_eval 0.4.9.2 | OpenCompass 0.5.5 |

版本依据与统一范围：

- LaCT 的 [原始依赖](lact/upstream/lact_llm/requirement.txt) 要求 PyTorch 2.6.x、FlashAttention 2.7.4.post1，因此保留 Torch 2.6.0 cu124／Triton 3.2.0；原仓库未固定的 FLA、Transformers 分别与 E²-TTT、In-Place TTT 统一。
- E²-TTT 按 [environment.yml](e2_ttt/upstream/environment.yml) 保留 Python 3.12、Torch 2.8.0、FlashAttention 2.8.3、FLA 0.4.1、Transformers 4.57.1、NumPy 2.3.3、datasets 4.2.0 等明确版本。
- In-Place TTT 按 [原安装说明](in_place_ttt/upstream/README.md#environment-setup) 保留 Python 3.11、Torch 2.8.0 cu128、FlashAttention 2.8.3、Transformers 4.57.3。原仓库未固定 OpenCompass，本轮选择 0.5.5，并安装其要求的 mmengine-lite 0.11.0rc3。
- OpenCompass 0.5.5 要求 NumPy <2、datasets <4，因此 In-Place TTT 使用 NumPy 1.26.4／datasets 3.6.0，不能与 E²-TTT 的两个固定版本统一。
- 三个评估环境统一 einops 0.8.1、huggingface-hub 0.36.2、tokenizers 0.22.2、safetensors 0.8.0、setuptools 80.9.0；E²-TTT 和 In-Place TTT 统一 Accelerate 1.10.1、SciPy 1.16.3。其余显式依赖见各方法 `requirements-eval.txt`。

## 安装与检查结果

| 方法 | 环境目录 | 依赖检查 | 导入检查 |
|---|---|---|---|
| LaCT | `~/rivermind-data/envs/ttt-lact/` | `uv pip check` 通过 | Torch、Transformers、FlashAttention 通过；完整模型模块待 GPU 可用后检查 |
| E²-TTT | `~/rivermind-data/envs/ttt-e2_ttt/` | `uv pip check` 通过 | 本地评估入口、原模型注册、harness 及 SQuAD／SWDE 任务通过 |
| In-Place TTT | `~/rivermind-data/envs/ttt-in_place_ttt/` | `uv pip check` 通过 | 本地评估入口、原模型注册、OpenCompass wrapper 与 RULER 评分器通过 |

导入检查均隐藏 CUDA 设备，且结束时 `torch.cuda.is_initialized()` 为 `False`。LaCT 所用 Triton 3.2.0 在导入 FLA 模块时会初始化 GPU 驱动，因此尚未完成该模型模块的导入检查。以上结果不代表 GPU kernel、完整 checkpoint 加载或 benchmark 已验证；本轮未新增测试，也未启动训练或 benchmark。

当前会话 `nvidia-smi` 仅暴露一张 A800，编号 0，设备节点为 `/dev/nvidia7`，与物理 GPU 1 的分配不一致。用户已明确要求继续仅使用 GPU 1，等待其映射恢复；恢复且该卡空闲前不启动 GPU 检查，评估入口继续保持 `CUDA_VISIBLE_DEVICES=1`。

安装从同步后的服务器项目仓库执行。GitHub 已发布配置，服务器因无法连接 GitHub，通过 Gitee 同步同一提交。安装日志、实际包版本保存在服务器项目 `artifacts/reproductions/environment-setup_20261109/`，分别为 `<method>_install.log`、`<method>_versions.json`，主环境对照为 `main_versions.json`，检查结果为 `validation.json`。服务器系统为 Ubuntu 24.04.4 x86_64，驱动 580.126.20，CUDA Toolkit 12.8（nvcc V12.8.93）。

## 安装命令

使用 uv 0.9.8，系统需已有 Python 3.11 和 3.12。三个 `requirements-eval.txt` 的注释记录了官方 FlashAttention 预编译 wheel 地址；预先准备与各环境 Python／Torch／CXX11 ABI 配套的 wheel，统一保存到 `~/rivermind-data/wheels/ttt/` 并保留原文件名。安装器使用该目录中的 wheel，不从源码编译。当前服务器无法直连 GitHub 发布页，已由可联网机器下载官方 wheel，再通过 SSH 传入。服务器 uv 安装于 `~/rivermind-data/tools/uv/bin/`，将该目录加入当前 shell 的 PATH 后，从项目根目录执行：

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh lact
bash reproductions/setup_eval_env.sh e2_ttt
bash reproductions/setup_eval_env.sh in_place_ttt
```

环境目录为 `~/rivermind-data/envs/ttt-lact/`、`ttt-e2_ttt/`、`ttt-in_place_ttt/`。运行评估前分别激活对应目录下的 `bin/activate`。安装器遵守各方法 `requirements-eval.txt`，只在 E²-TTT 环境安装其本地模型包和随附 harness。
