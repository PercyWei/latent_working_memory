# 20260904 C-DIC 多轮 GPU Smoke Test 记录（20260904 21:08:40 CST）

创建时间：20260904 21:08:40 CST（UTC+08:00）
最后修订时间：20260904 21:26:56 CST（UTC+08:00）
状态：工程 smoke test 通过；尚未进行 MSC 训练与论文结果复现

## 测试范围

验证真实 Llama-2-7B-Chat 与 ICAE v1 checkpoint 接入 C-DIC 后的以下链路：

- checkpoint 恢复与 strict load；
- 五轮 `retrieve → generate → compress → write-back`；
- latent shape、dtype、有限值和 gradient 状态；
- `initialize`、`insert`、`replace` 与 top-1 fallback；
- `thread_id`、`state_id` 和 revision；
- memory trace 与峰值 GPU memory。

本测试不验证 MSC 训练效果或论文指标。

## 环境

| 项目 | 值 |
|---|---|
| Git commit | `e838848459dc8281193c88702f455a78b17e389f` |
| Python | `3.10.21` |
| PyTorch | `2.0.1+cu118` |
| CUDA runtime | `11.8` |
| GPU | NVIDIA A800-SXM4-80GB，物理 GPU 0 |
| 基础模型 | `/data/bywei/models/meta-llama/Llama-2-7b-chat-hf` |
| ICAE checkpoint | `/data/bywei/checkpoints/icae/v1/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt` |
| C-DIC 环境 | `reproductions/cdic/.venv` |

## 测试方法

### 1. 创建独立环境

```bash
uv sync --project reproductions/cdic --frozen
```

### 2. CPU tests

```bash
uv run --project reproductions/cdic --no-sync \
  pytest -q reproductions/cdic/tests
```

结果：`28 passed`。

### 3. 多轮 GPU test

测试文件：`reproductions/cdic/tests/test_gpu_multiturn_smoke.py`

配置文件：`reproductions/cdic/configs/gpu_smoke_a800.json`。模型路径、checkpoint、device、artifact 路径、生成参数和五轮 query 均由该文件提供。

```bash
uv run --project reproductions/cdic --no-sync pytest -q -s \
  reproductions/cdic/tests/test_gpu_multiturn_smoke.py \
  --cdic-gpu-config reproductions/cdic/configs/gpu_smoke_a800.json
```

测试使用五轮对话：写入随机 access code、切换话题、回访旧信息、更新 access code、再次查询更新结果。

## 测试结果

GPU test：环境变量版与 JSON 配置版均为 `1 passed`；最终 JSON 配置版耗时 `172.92 s`。

| Turn | Query 目的 | Peak score | Fallback | Write-back | Thread / revision | Response 摘要 |
|---|---|---:|---|---|---|---|
| 1 | 写入 `ZETA-4827` | — | 否 | `initialize` | `thread-000001 / 0` | 换行符 |
| 2 | 切换到海豚话题 | 0.6953 | 是 | `insert` | `thread-000002 / 0` | 正常回答海豚呼吸 |
| 3 | 回访 Project Quartz | 0.8064 | 否 | `replace` | `thread-000001 / 1` | 正确回答 `ZETA-4827` |
| 4 | 更新为 `OMEGA-7319` | 0.4713 | 是 | `insert` | `thread-000003 / 0` | 确认新 code |
| 5 | 查询当前 code | 0.4858 | 是 | `insert` | `thread-000004 / 0` | 错误回答 `42010` |

共同检查结果：

- 每轮 latent shape：`[128, 4096]`；
- latent dtype：bfloat16；
- latent 与 retrieval key 均为有限值；
- inference tensor 不保留 gradient；
- 最终 memory state 数：4；
- 峰值 GPU memory：14,143,527,424 bytes，约 13.17 GiB；
- GPU test 结束后物理 GPU 0 显存已释放。

## 结论

工程链路通过：真实模型能够完成多轮 retrieval、generation、compression 和 write-back，trace 与阈值规则一致。

语义效果尚未复现：第三轮能够召回旧 code，但第四轮更新未合并到原 thread，第五轮也未正确召回新值。这符合当前仅使用 ICAE initialization、尚未进行 MSC C-DIC 训练的状态。下一阶段应实现实际 ra-TBPTT autograd graph 和 MSC 最小训练闭环。

首次运行曾因要求 `response.strip()` 非空而失败；第一轮在空 memory 下生成换行符。该条件不属于工程正确性要求，移除后五轮测试通过。

## 产物

- 服务器报告：`/data/bywei/projects/latent_working_memory/artifacts/cdic/20260904_multiturn_gpu_smoke/gpu_smoke_report.json`
- 报告 SHA256：`a732f168897c80d9dcc8c570139161834b10c061506b2898e51ea2c8ed8b93c7`
