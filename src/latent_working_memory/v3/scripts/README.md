# GPU 训练任务启动

终端与 GPU 网站共用同一脚本。通过 `--gpus` 指定物理卡号；当前服务器使用 `--gpus 4,5,6,7`，脚本内部通过 `torchrun` 启动四个训练进程，最终评估只使用列表中的第一张卡（GPU 4）。全局 batch 仍为 8，完整 batch 时每个进程累积两条轨迹。参数默认值保留 `0,1`，每次启动按实际分配显式指定。

## 第一次运行

环境需有 `uv`，脚本用 `uv run --frozen` 按仓库 `uv.lock` 准备根目录 `.venv`。通过终端环境变量或网站密钥设置提供 `SWANLAB_API_KEY`，不用写入脚本或提交到仓库。SwanLab project 已确定为 **`latent-working-memory-v3`**。

默认 `cuda124` 依赖组使用 Python 3.11、PyTorch 2.6.0。根目录 `pyproject.toml` 与 `uv.lock` 统一使用[清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/)；Linux x86-64 的 PyTorch 2.6.0 PyPI 构建使用 CUDA 12.4，适配当前 NVIDIA 550.144.03 驱动，macOS 安装对应平台构建。

当前终端服务器先同步仓库并安装环境，下载不使用代理：

```bash
cd ~/percyw/latent_working_memory
git pull --ff-only origin dev
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
uv sync
```

已有 Python 3.11 时无需重复安装；首次准备且没有该解释器时，先执行 `uv python install 3.11`。`uv sync` 按统一镜像及锁定依赖准备 `.venv`，训练入口继续使用 `uv run --frozen`。

脚本的默认仓库位置为 `/dfs/data/latent_working_memory`。当前终端服务器使用 `~/percyw/latent_working_memory` 时，先设置 `LWM_REPO_DIR`：

```bash
export LWM_REPO_DIR="$HOME/percyw/latent_working_memory"
bash "$LWM_REPO_DIR/src/latent_working_memory/v3/scripts/run_gpu.sh" \
  --mode smoke --method dynamic --gpus 4,5,6,7 --run-id view-check-01
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
  --mode smoke --method dynamic --gpus 4,5,6,7 --run-id view-check-01
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
  --mode full --method dynamic --gpus 4,5,6,7 --run-id main-01
```

也可精确覆盖档位中的预算：

```bash
bash /dfs/data/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode pilot --method dynamic --gpus 4,5,6,7 --run-id pilot-02 \
  --max-steps 10 --train-samples 80 --dev-samples 8 --eval-trajectories 4
```

`--max-steps`、`--train-samples`、`--dev-samples`、`--eval-trajectories` 传 `0` 表示不设该上限。`--epochs` 修改所选阶段的训练轮数；若希望各阶段轮数不同，分别用下述单阶段入口。实际样本不足时不会复制样本凑数。

## 方法与阶段

| 参数 | 范围 |
|---|---|
| `--method dynamic` | 两个动态方法，基础预训练仅执行一次；默认 |
| `--method all` | 三个 baseline＋两个动态方法 |
| `--method icae_single / icae_multi / autocompressors / memory_change / information_loss` | 指定一个方法 |
| `--stage auto` | 自动执行该方法全部训练阶段，再评估；默认 |
| `--stage pretrain / qa / warmup / policy / lm` | 只执行一个明确方法的指定阶段 |

ICAE 与 AutoCompressors 使用已有长文本预设，运行它们时需提供 `--long-pretrain-data`。该目录同样采用原有 `TextSample` 格式，输入长度筛选为 4096–12288 个当前模型 tokens。此前 4096 上限的短片段成品不能普遍覆盖此范围；脚本保留长历史比较要求，缺少目录时直接说明原因。

```bash
bash /dfs/data/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode smoke --method all --gpus 4,5,6,7 --run-id all-check-01 \
  --long-pretrain-data /dfs/data/fineweb-long-text-samples
```

只运行信息损失策略阶段时，显式传入已训练的相同模型配置 checkpoint：

```bash
bash /dfs/data/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode pilot --method information_loss --stage policy --gpus 4,5,6,7 --run-id loss-policy-01 \
  --init-checkpoint /absolute/path/to/warmup/checkpoints/step-000002.pt
```

单独 `pretrain` 阶段只运行 AE＋LM 训练/开发集验证；`qa`、`warmup`、`policy`、`lm` 结束后运行 QA 评估。`warmup` checkpoint 的最终 QA 评估按真实门控策略构建记忆，训练中开发集验证仍使用预热动作日程。

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
| `--gpus 4,5,6,7` | 逗号分隔的非重复物理卡号，训练进程数随卡数变化；默认 `0,1` |
| `--model-path /absolute/model/path` | 使用共享盘上的 Qwen3-4B 模型，避免启动时从 Hub 获取 |
| `--global-batch-size 8` | 全局 batch，可按显存/运行时间调整 |
| `--qa-batch-size 8` | 单次读取的 QA 数 |
| `--threshold-i / --threshold-d / --threshold-g / --eta` | 覆盖门控阈值，记录于训练配置 |
| `--max-new-tokens 64` | 最终评估生成上限 |
| `--tracking disabled` | 仅保存本地记录；默认 online |
| `--swanlab-project` / `--group` | 项目与实验系列 |
| `--output-root` | 产物父目录，默认 `artifacts/v3` |

仓库位置不同，可通过环境变量覆盖，脚本会正确处理带空格的路径：

```bash
LWM_REPO_DIR=/another/repository \
  bash /another/repository/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode smoke --gpus 4,5,6,7 --dry-run
```

本地测试验证预算、阶段衔接、数据选择、原生指标和评估图表构造；服务器上的实际显存、训练速度及云端展示需要通过第一轮 `smoke` 任务核验。
