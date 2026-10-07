# GPU 训练任务启动

终端与 GPU 网站共用同一脚本。当前服务器使用 `--gpus 4,5`，脚本通过 `torchrun` 启动两个训练进程，最终评估使用 GPU 4。参数默认卡号保留 `0,1`，每次启动按实际分配显式指定。

训练通过 `--micro-batch-size-per-gpu` 与 `--gradient-accumulation-steps` 控制批处理。全局 batch 根据 **GPU 数 × 每卡 microbatch × 累积步数** 计算，写入运行计划、本地 `run.json` 和 SwanLab config；不再单独传入 `--global-batch-size`。

| 双卡配置 | 每卡一次并行样本数 | 每次更新累积次数 | 全局 batch |
|---|---:|---:|---:|
| 默认 | 1 | 4 | 8 |
| `--micro-batch-size-per-gpu 2 --gradient-accumulation-steps 2` | 2 | 2 | 8 |
| `--micro-batch-size-per-gpu 4 --gradient-accumulation-steps 1` | 4 | 1 | 8 |

microbatch 内合并实际写入和读取调用，支持不同长度及不同更新动作。`qa_batch_size` 是每条轨迹一次并行读取的题数，因此训练时一次读取最多包含 `microbatch × qa_batch_size` 道题。尾批按真实样本数归一；增加卡数或改变这两个参数会改变全局 batch。H20 上可从第二行试跑，依据吞吐和显存峰值调整；预训练与 QA 可分别选择设置。

## 第一次运行

环境需有 `uv`，脚本用 `uv run --frozen` 按仓库 `uv.lock` 准备根目录 `.venv`。SwanLab project 已确定为 **`latent-working-memory-v3`**。

默认 `cuda124` 依赖组使用 Python 3.11、PyTorch 2.6.0。根目录 `pyproject.toml` 与 `uv.lock` 统一使用[清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/)；Linux x86-64 的 PyTorch 2.6.0 PyPI 构建使用 CUDA 12.4，适配当前 NVIDIA 550.144.03 驱动，macOS 安装对应平台构建。

当前终端服务器先同步仓库并安装环境，下载不使用代理：

```bash
cd ~/percyw/latent_working_memory
git pull --ff-only origin dev
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
uv sync
```

已有 Python 3.11 时无需重复安装；首次准备且没有该解释器时，先执行 `uv python install 3.11`。`uv sync` 按统一镜像及锁定依赖准备 `.venv`，训练入口继续使用 `uv run --frozen`。

在项目根目录创建 `.env`，填写自己的 SwanLab API Key。首次创建可复制模板；已有文件时直接编辑：

```bash
test -e .env || cp .env.example .env
nano .env
```

文件内容：

```dotenv
SWANLAB_API_KEY=你的SwanLab_API_Key
```

v3 的凭据选择规则如下；训练、续训与评估均从项目根目录启动：

| 条件 | 使用的凭据 |
|---|---|
| 项目根目录存在 `.env` | 仅使用文件中的 `SWANLAB_API_KEY`，优先于终端变量 |
| 没有 `.env` | 使用终端或网站任务注入的 `SWANLAB_API_KEY` |
| 所选来源未提供有效值 | 在线运行报错，不回退到共享账号保存的登录凭据 |

`.env` 已被 Git 忽略，需在服务器本地填写；只读取其中的 `SWANLAB_API_KEY`，其他字段不自动加载。无需执行 `swanlab login`，不会改写共享账号的登录文件。key 仅用于认证及传入任务子进程，不写入任务配置、命令行或本地实验记录。`--dry-run` 和 `--tracking disabled` 不要求提供 key。

脚本的默认仓库位置为 `/dfs/data/latent_working_memory`。当前终端服务器使用 `~/percyw/latent_working_memory` 时，先设置 `LWM_REPO_DIR`：

```bash
export LWM_REPO_DIR="$HOME/percyw/latent_working_memory"
bash "$LWM_REPO_DIR/src/latent_working_memory/v3/scripts/run_gpu.sh" \
  --mode smoke --method dynamic --gpus 4,5 --run-id view-check-01
```

默认数据位置：

| 用途 | 相对仓库路径 |
|---|---|
| 动态方法 AE＋LM | `data/fineweb-4096-doc100k_20260910/semantic` |
| QA | `data/fineweb-factqa-train1000_20260930` |

数据目录内需有 `train.jsonl`、`dev.jsonl`、`test.jsonl`，QA 还需 `preparation.json`。AE＋LM 使用之前构造的成品，**不把原始 Parquet 目录直接传给训练入口**。可用 `--pretrain-data`、`--qa-data` 指定实际存放位置。

仓库位于默认路径时，可在网站的启动命令栏填写：

```bash
bash /dfs/data/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode smoke --method dynamic --gpus 4,5 --run-id view-check-01
```

这条命令会依次执行：

```text
共享单段 AE＋LM 预训练
 ├─ 按记忆变化：动作预热 → 策略训练 → 开发集 QA 评估
 └─ 按信息损失：动作预热 → 策略训练 → 开发集 QA 评估
最后导出两个方法的质量—容量比较点
```

每次训练继承前一阶段实际产出的 checkpoint。两种动态方法共享基础预训练 checkpoint，之后各自训练；阶段间的参数和 optimizer 初始化沿用实验协议。

## 控制运行程度

下表为**每个训练阶段**的上限；`full` 沿用所选预设的训练轮数，当前默认 1 epoch。

| 参数 | `smoke` | `pilot` | `full` |
|---|---:|---:|---:|
| 训练样本 | 最多 16 | 最多 256 | 全部符合条件的样本 |
| 开发集样本 | 最多 4 | 最多 32 | 全部 |
| optimizer steps | 最多 2 | 最多 20 | 无额外上限 |
| 验证/保存间隔 | 每步 | 每 5 步 | 每 25 步 |
| 最终 QA 评估 | dev 最多 2 条轨迹 | dev 最多 16 条轨迹 | 完整 test |

- 各档均使用 Qwen3-4B、64 slots 和相同写入实现；QA 保留整条原文、全部更新点与原题池。
- AE＋LM 在长度筛选后按任务分层选样，train/dev 均保留两类任务。QA 按 seed 选完整轨迹；两种动态方法使用相同样本。
- 限步结束时仍执行开发集验证并保存 checkpoint，因此短跑也会有 `train`、`dev` 与最终 `evaluation` 展示。
- 样本上限控制计算使用的内存索引，不创建数据副本；首次加载仍读取并分词来源数据。更换档位不会缩短模型加载时间。
- `smoke`、`pilot` 用于检查实现、显存、耗时和展示；短跑质量不作为研究结论。

检查通过后，在新任务中把 `--mode` 改为 `pilot` 或 `full`。不同档位使用不同输出目录和 group，正式训练从头按该档位执行；不会自动继承试跑权重。

```bash
bash /dfs/data/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode full --method dynamic --gpus 4,5 --run-id main-01
```

也可精确覆盖档位中的预算：

```bash
bash /dfs/data/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode pilot --method dynamic --gpus 4,5 --run-id pilot-02 \
  --max-steps 10 --train-samples 80 --dev-samples 8 --eval-trajectories 4
```

`--max-steps`、`--train-samples`、`--dev-samples`、`--eval-trajectories` 传 `0` 表示不设该上限。`--epochs` 修改所选阶段的训练轮数；若希望各阶段轮数不同，分别用下述单阶段入口。实际样本不足时不会复制样本凑数。

## 方法与阶段

| 参数 | 范围 |
|---|---|
| `--method dynamic` | 两个动态方法，基础预训练仅执行一次；默认 |
| `--method all` | 三个 baseline＋两个动态方法 |
| `--method icae_single / icae_multi / autocompressors / memory_change / information_loss` | 指定一个方法 |
| `--stage auto` | 默认执行全部训练阶段；动态方法提供 `--init-checkpoint` 时从已有预训练开始，连续执行 warmup → policy → 评估 |
| `--stage pretrain / qa / warmup / policy / lm` | 只执行一个明确方法的指定阶段 |

ICAE 与 AutoCompressors 使用已有长文本预设，运行它们时需提供 `--long-pretrain-data`。该目录同样采用原有 `TextSample` 格式，输入长度筛选为 4096–12288 个当前模型 tokens。此前 4096 上限的短片段成品不能普遍覆盖此范围；脚本保留长历史比较要求，缺少目录时直接说明原因。

```bash
bash /dfs/data/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode smoke --method all --gpus 4,5 --run-id all-check-01 \
  --long-pretrain-data /dfs/data/fineweb-long-text-samples
```

只运行信息损失策略阶段时，显式传入已训练的相同模型配置 checkpoint：

```bash
bash /dfs/data/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode pilot --method information_loss --stage policy --gpus 4,5 --run-id loss-policy-01 \
  --init-checkpoint /absolute/path/to/warmup/checkpoints/step-000002.pt
```

单独 `pretrain` 阶段只运行 AE＋LM 训练/开发集验证；`qa`、`warmup`、`policy`、`lm` 结束后运行 QA 评估。`warmup` checkpoint 的最终 QA 评估按真实门控策略构建记忆，训练中开发集验证仍使用预热动作日程。

已有共享预训练 checkpoint 时，一条命令连续完成某个动态方法的预热、策略训练及最终评估：

```bash
cd ~/percyw/latent_working_memory
export LWM_REPO_DIR="$PWD"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

bash src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode smoke --method memory_change --stage auto --gpus 4,5 \
  --init-checkpoint /absolute/path/to/pretrain/checkpoints/step-NNNNNN.pt \
  --model-path "$HOME/models/Qwen3-4B-Instruct-2507" \
  --qa-data "$LWM_REPO_DIR/data/fineweb-factqa-train1000_20260930" \
  --micro-batch-size-per-gpu 2 --gradient-accumulation-steps 2 --qa-batch-size 8 \
  --group dynamic-qa-smoke_shared-pretrain \
  --run-id memory-change-smoke-01
```

将 `--method` 改为 `information_loss`、`--run-id` 改为 `information-loss-smoke-01`，并保持相同的预训练 checkpoint 与 group，即可单独运行另一方法。也可使用 `--method dynamic`，在同一任务中顺序完成两组并导出比较表。

此模式只需要 QA 数据，不读取 AE＋LM 数据；模型配置须与预训练 checkpoint 一致。warmup 的实际最终 checkpoint 自动传入 policy；阶段切换只继承可训练权重，优化器重新初始化。最终评估在 policy 后执行，warmup 的训练中验证照常进行。`--mode`、batch 参数和训练步数上限同时作用于 warmup 与 policy；需要分别调参时仍使用单阶段入口。

## SwanLab 与本地记录

同一启动任务的全部 runs 使用同一 group；每个训练阶段一个 run，其最终评估追加到该 run。默认 group 为 `capacity-<mode>_<run-id>`，可用 `--group` 显式设置。

| 展示位置 | 内容 |
|---|---|
| `train` / `dev` | 真实 optimizer step 的增量 loss、slots 曲线；训练梯度范数 |
| `resources` | 每步耗时、CUDA 峰值显存 |
| `evaluation` | NLL、EM、F1 三张合并 all/old/new 的汇总图；容量、成本、距离分层与生成样例表 |

完整动作、门控分数和逐题结果保存在本地；SwanLab 中的信息损失结果也注明 `offline_oracle`。API key 不写入任务计划或配置快照。

产物根目录为 `artifacts/v3/capacity-<mode>_<run-id>/`：

```text
plan/     任务计划、各阶段解析配置、控制台日志、最终执行结果
train/    每个阶段的运行配置、metrics.jsonl、SwanLab 身份与 checkpoints
eval/     最终记忆 QA 评估的 trajectories.jsonl 与 summary.json
compare/  同题池质量—容量点 points.json / points.csv
```

已有任务目录不会覆盖；重跑用新的 `--run-id`。阶段失败即停止后续阶段，`plan/result.json` 记录失败和已完成的 checkpoint。需要从中断位置续训时，使用原训练模块的 `--resume` 和保存的 `config.json`，具体命令见上级 [实现说明](../README.md)。

## 其他常用参数

| 参数 | 作用 |
|---|---|
| `--dry-run` | 打印预算、阶段依赖、解析配置和命令；不加载模型/数据，不连接 SwanLab，不创建实验目录 |
| `--gpus 4,5` | 逗号分隔的非重复物理卡号，训练进程数随卡数变化；默认 `0,1` |
| `--model-path /absolute/model/path` | 使用共享盘上的 Qwen3-4B 模型，避免启动时从 Hub 获取 |
| `--micro-batch-size-per-gpu 2` | 每卡一次并行处理的样本/轨迹数，默认 1 |
| `--gradient-accumulation-steps 2` | 每次参数更新累积的 microbatch 数，默认 4；全局 batch 自动计算 |
| `--qa-batch-size 8` | 每条轨迹单次读取的 QA 数 |
| `--init-checkpoint` | 单阶段初始化权重；动态 auto 流程中指定共享预训练 checkpoint |
| `--threshold-i / --threshold-d / --threshold-g / --eta` | 覆盖门控阈值，记录于训练配置 |
| `--max-new-tokens 64` | 最终评估生成上限 |
| `--tracking disabled` | 仅保存本地记录；默认 online |
| `--swanlab-project` / `--group` | 项目与实验系列 |
| `--output-root` | 产物父目录，默认 `artifacts/v3` |

仓库位置不同，可通过环境变量覆盖，脚本会正确处理带空格的路径：

```bash
LWM_REPO_DIR=/another/repository \
  bash /another/repository/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode smoke --gpus 4,5 --dry-run
```

本地测试验证预算、阶段衔接、数据选择、原生指标和评估图表构造；服务器上的实际显存、训练速度及云端展示需要通过第一轮 `smoke` 任务核验。
