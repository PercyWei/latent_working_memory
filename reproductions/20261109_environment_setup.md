# 20261109_TTT checkpoint 评估环境

创建时间：20261109 10:58:35 Asia/Shanghai

最后修订时间：20261109 11:06:17 Asia/Shanghai

## 执行计划

为 checkpoint 评估建立独立环境，保留主项目 `.venv`。原研究固定版本优先；未固定依赖尽量与其他方法已固定版本一致。暂不安装训练专用的 flame／torchtitan／VeOmni 等组件。

| 项目 | LaCT | E²-TTT | In-Place TTT |
|---|---|---|---|
| Python | 3.11 | 3.12，沿用原环境 | 3.11，沿用原环境 |
| PyTorch | 2.6.0，遵守原 2.6.x 范围 | 2.8.0 cu128 | 2.8.0 cu128 |
| FlashAttention | 2.7.4.post1 | 2.8.3 | 2.8.3 |
| Transformers | 4.57.3，与 In-Place TTT 一致 | 4.57.1 | 4.57.3 |
| FLA | 0.4.1，与 E²-TTT 一致 | 0.4.1 | 不需要 |
| OpenCompass | 不需要 | 随附 lm-evaluation-harness | 0.5.5 |

OpenCompass 0.5.5 要求 NumPy <2、datasets <4，因此 In-Place TTT 采用 NumPy 1.26.4／datasets 3.6.0；LaCT 同用 NumPy 1.26.4，E²-TTT 保留原固定 NumPy 2.3.3／datasets 4.2.0。可共用的依赖固定为 FLA 0.4.1（LaCT／E²-TTT）、einops 0.8.1、Transformers 4.57.3（LaCT／In-Place TTT）、Accelerate 1.10.1（E²-TTT／In-Place TTT）、huggingface-hub 0.36.2、tokenizers 0.22.2、safetensors 0.8.0。

服务器安装从经 GitHub 同步的项目仓库执行。依赖环境保存到 `~/rivermind-data/envs/ttt-<method>/`，安装命令与结果保存到项目 `artifacts/reproductions/environment-setup_20261109/`。先核对磁盘与版本约束，再安装和导入验证；不下载模型或数据，不启动训练或 benchmark，不新增测试。

## GPU 可见性

当前会话 `nvidia-smi` 仅暴露一张 A800，编号 0，设备节点为 `/dev/nvidia7`。与此前物理 GPU 1 的授权不一致，未确认映射前不启动 CUDA 计算，不改评估脚本的设备分配。

## 状态

正在核对版本并建立环境，尚未完成安装。CUDA Toolkit 12.8.93、驱动 580.126.20、Ubuntu 24.04.4 x86_64 满足此前核对的系统条件。

## 安装命令

使用 uv 0.9.8，系统需已有 Python 3.11 和 3.12。服务器 uv 安装于 `~/rivermind-data/tools/uv/bin/`，将该目录加入当前 shell 的 PATH 后，从项目根目录执行：

```bash
export PATH="$HOME/rivermind-data/tools/uv/bin:$PATH"
bash reproductions/setup_eval_env.sh lact
bash reproductions/setup_eval_env.sh e2_ttt
bash reproductions/setup_eval_env.sh in_place_ttt
```

环境目录为 `~/rivermind-data/envs/ttt-lact/`、`ttt-e2_ttt/`、`ttt-in_place_ttt/`。运行评估前分别激活对应目录下的 `bin/activate`。安装器遵守各方法 `requirements-eval.txt`，只在 E²-TTT 环境安装其本地模型包和随附 harness。
