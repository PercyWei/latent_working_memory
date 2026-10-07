# GPU 训练任务启动

终端与 GPU 网站共用同一脚本，默认运行五种方法的 `smoke` 完整流程。脚本开头的 `LWM_REPO_DIR` 固定为当前服务器路径 `/data/zhangdw12/percyw/latent_working_memory`，启动时切换到该目录并设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。当前服务器使用 `--gpus 4,5`，通过 `torchrun` 启动两个训练进程，最终评估使用 GPU 4；参数默认卡号仍为 `0,1`。

模型默认读取 `~/models/Qwen3-4B-Instruct-2507`，数据使用下文的仓库内路径，SwanLab project 为 `latent-working-memory-v3`。新运行的 `run-id` 自动按上海时间生成，无需预先设置路径环境变量或运行标识。

```bash
# 五方法试跑
bash /data/zhangdw12/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh --mode smoke --gpus 4,5

# 五方法正式训练
bash /data/zhangdw12/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh --mode full --gpus 4,5
```

公司 GPU 网站直接填写上述绝对路径启动命令。换机器时修改脚本开头的 `LWM_REPO_DIR` 常量和启动命令中的脚本路径；GPU、方法和 microbatch 等仍通过命令行参数指定。

训练通过 `--micro-batch-size-per-gpu` 与 `--gradient-accumulation-steps` 控制批处理。全局 batch 根据 **GPU 数 × 每卡 microbatch × 累积步数** 计算，写入运行计划、本地 `run.json` 和 SwanLab config；不再单独传入 `--global-batch-size`。

| 双卡配置 | 每卡一次并行样本数 | 每次更新累积次数 | 全局 batch |
|---|---:|---:|---:|
| 默认 | 2 | 2 | 8 |
| `--micro-batch-size-per-gpu 4 --gradient-accumulation-steps 1` | 4 | 1 | 8 |

microbatch 内合并实际写入和读取调用，支持不同长度及不同更新动作。`qa_batch_size` 是每条轨迹一次并行读取的题数，因此训练时一次读取最多包含 `microbatch × qa_batch_size` 道题。尾批按真实样本数归一；增加卡数或改变这两个参数会改变全局 batch。第一组实验采用两张 H20、每卡 microbatch 2、累积 2，全局 batch 8；实际显存与吞吐先由 `smoke` 确认。

默认启用独立编码器／冻结解码器、逐层激活重计算、CUDA BF16 autocast 和纯 causal attention。预训练在每卡既定 batch 内按长度分组；运行命令无需增加优化参数。预设中的 `model.gradient_checkpointing` 可用于关闭逐层重算进行对照，实际值随配置保存。保持当前 PyTorch 2.6 环境，无需新增 attention 依赖。

## 第一次运行

环境需有 `uv`，脚本用 `uv run --frozen` 按仓库 `uv.lock` 准备根目录 `.venv`。SwanLab project 已确定为 **`latent-working-memory-v3`**。

默认 `cuda124` 依赖组使用 Python 3.11、PyTorch 2.6.0。根目录 `pyproject.toml` 与 `uv.lock` 统一使用[清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/)；Linux x86-64 的 PyTorch 2.6.0 PyPI 构建使用 CUDA 12.4，适配当前 NVIDIA 550.144.03 驱动，macOS 安装对应平台构建。

当前终端服务器先同步仓库并安装环境，下载不使用代理：

```bash
cd /data/zhangdw12/percyw/latent_working_memory
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

v3 的凭据选择规则如下；脚本自动在项目根目录启动训练与评估：

| 条件 | 使用的凭据 |
|---|---|
| 项目根目录存在 `.env` | 仅使用文件中的 `SWANLAB_API_KEY`，优先于终端变量 |
| 没有 `.env` | 使用终端或网站任务注入的 `SWANLAB_API_KEY` |
| 所选来源未提供有效值 | 在线运行报错，不回退到共享账号保存的登录凭据 |

`.env` 已被 Git 忽略，需在服务器本地填写；只读取其中的 `SWANLAB_API_KEY`，其他字段不自动加载。无需执行 `swanlab login`，不会改写共享账号的登录文件。key 仅用于认证及传入任务子进程，不写入任务配置、命令行或本地实验记录。`--dry-run` 和 `--tracking disabled` 不要求提供 key。

默认数据位置：

| 用途 | 相对仓库路径 |
|---|---|
| 五方法共用的 AE／LM 索引 | `data/fineweb-reconstruction-k512-doc100k_20260917` |
| 索引引用的原文 | `data/raw/HuggingFaceFW-fineweb/sample-10BT/` |
| QA | `data/fineweb-factqa-train1000_20260930` |

以上数据在 [ModelScope 数据仓库](https://modelscope.cn/datasets/percyWeeei/latent-working-memory/files) 的根目录分别为 `fineweb-reconstruction-k512-doc100k_20260917/`、`raw/` 与 `fineweb-factqa-train1000_20260930/`。服务器需保留索引与 `raw` 同级的布局；索引保存原文位置，训练时从对应 Parquet 恢复文本。

预训练根目录包含 `preparation.json`，以及 `single/`、`multi/` 各自的 `train.jsonl`、`dev.jsonl`、`test.jsonl`；QA 根目录包含三个 split 文件和 `preparation.json`。`--pretrain-data` 指向索引根目录，`--qa-data` 指向 QA 根目录。

| 方法 | 预训练数据视图与目标 |
|---|---|
| 两个动态方法 | 从 `multi` 仅选择首个切点落在 768–1024 tokens 的记录，派生该单段的 AE＋LM，共享预训练一次 |
| ICAE-single | `single` 最终切点的 1024–4096 tokens 原文，派生 AE＋LM；每例一次压缩 |
| ICAE-multi | 同一 `single` 原文与 AE＋LM 目标，写入时按 1024 tokens 独立分段，拼接记忆读出 |
| AutoCompressors | 同一 `single` 的一份 continuation 样本，拼接输入与后续 512 tokens 做随机分段 LM，不另复制 AE 样本 |

两个 ICAE 随后在完整 QA 长轨迹上分别训练全长压缩和多块拼接读出；两个动态方法在同一长轨迹上执行 warmup 与 policy。QA 已标注的文本、段界与题池保持原样。AutoCompressors 仅训练 LM，再在同一 QA 测试集评估。

20261007 使用 Qwen3-4B tokenizer 读取发布数据：QA 为 1007／118／120 条 train／dev／test 轨迹，每条 6–10 段；训练轨迹实际为 3307–15448 tokens，字符估算的 6–8K 不是实际分词长度筛选条件。动态基础预训练保留 3532 条训练原文，派生 7064 个 AE／LM 样本；开发集为 12 条原文、24 个样本。基线保留 31957 条训练原文：每个 ICAE 使用 63914 个 AE／LM 样本，AutoCompressors 使用 31957 个 LM 样本。两种预训练视图均与 QA 来源文档和去重簇无交集。

默认五方法试跑依次执行：

```text
共享单段 AE＋LM 预训练
ICAE-single：AE＋LM → QA → 开发集 QA 评估
ICAE-multi：AE＋LM → QA → 开发集 QA 评估
AutoCompressors：多段 LM → 开发集 QA 评估
共享 checkpoint
 ├─ 按记忆变化：动作预热 → 策略训练 → 开发集 QA 评估
 └─ 按信息损失：动作预热 → 策略训练 → 开发集 QA 评估
最后导出五个方法的质量—容量比较点
```

每次训练继承前一阶段实际产出的 checkpoint。两种动态方法共享基础预训练 checkpoint，之后各自训练；跨阶段继承可训练参数，optimizer 重建。第一组门控阈值 `threshold_i`、`threshold_d`、`threshold_g` 均为 0.1，`eta=0`；这些是初始设置，尚未做容量校准。

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
- AE＋LM 在长度筛选后按任务分层选样，train/dev 均保留两类任务；AutoCompressors 只使用 continuation 视图。QA 按 seed 选完整轨迹；两种动态方法使用相同样本。
- 限步结束时仍执行开发集验证并保存 checkpoint，因此短跑也会有 `train`、`dev` 与最终 `evaluation` 展示。
- 样本上限控制计算使用的内存索引，不创建数据副本；首次加载仍读取并分词来源数据。更换档位不会缩短模型加载时间。
- `smoke`、`pilot` 用于检查实现、显存、耗时和展示；短跑质量不作为研究结论。

检查通过后，将 `--mode` 改为 `pilot` 或 `full` 即可。默认每次生成新的 `run-id`；也可显式指定，同一标识的不同档位使用各自目录。未指定 `--init-checkpoint` 时，不会继承试跑权重。

命令行参数覆盖 `configs/v3/` 中各方法的预设；`all` 下的覆盖应用于所有方法、所有训练阶段。例如提高每卡并行量、保持双卡全局 batch 为 8：

```bash
bash /data/zhangdw12/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode smoke --gpus 4,5 --micro-batch-size-per-gpu 4 --gradient-accumulation-steps 1
```

也可覆盖档位中的预算：

```bash
bash /data/zhangdw12/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode pilot --gpus 4,5 \
  --max-steps 10 --train-samples 80 --dev-samples 8 --eval-trajectories 4
```

`--max-steps`、`--train-samples`、`--dev-samples`、`--eval-trajectories` 传 `0` 表示不设该上限。`--epochs` 修改流程中各阶段的训练轮数。`smoke`、`pilot` 仍完成全部训练阶段和最终评估，只截断各阶段预算；实际样本不足时不会复制样本凑数。

## 完整方法入口

| 参数 | 流程 |
|---|---|
| `--method all` | 三个 baseline 与两个动态方法的完整流程；默认 |
| `--method dynamic` | 共享预训练一次，再分别执行两个动态方法的 warmup → policy → 评估 |
| `--method memory_change / information_loss` | 共享预训练 → 该方法 warmup → policy → 评估 |
| `--method icae_single / icae_multi` | 该方法 pretrain → QA → 评估 |
| `--method autocompressors` | LM → 评估 |
| `--method shared_pretrain` | 仅生成两个动态方法共用的 AE＋LM 预训练 checkpoint，并执行预训练开发集验证 |

只运行一个方法时增加 `--method`，例如：

```bash
bash /data/zhangdw12/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode smoke --gpus 4,5 --method icae_single
```

五方法统一使用 `--pretrain-data`，各预设决定读取哪个索引视图。动态方法提供 `--init-checkpoint` 时，从已有共享预训练开始，自动执行 warmup → policy → 最终评估。

先单独准备共享预训练时：

```bash
bash /data/zhangdw12/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode smoke --method shared_pretrain --gpus 4,5 --run-id shared-01
```

已有共享预训练 checkpoint 时，一条命令完成某个动态方法的完整 QA 训练流程：

```bash
bash /data/zhangdw12/percyw/latent_working_memory/src/latent_working_memory/v3/scripts/run_gpu.sh \
  --mode smoke --method memory_change --gpus 4,5 \
  --init-checkpoint /absolute/path/to/shared-pretrain-k64_smoke_shared-01/pretrain/checkpoints/step-NNNNNN.pt
```

`--run-id` 自动继承 checkpoint 同阶段 `run.json` 中的 `config.training.experiment_id`；显式传入不同值会报错。将 `--method` 改为 `information_loss`，保持同一 checkpoint 和其余参数，即可在同一系列分别启动另一方法，无需改 `run-id`。也可使用 `--method dynamic` 顺序完成两组并导出比较表。

这一路径只需要 QA 数据，不读取 AE＋LM 数据；模型配置须与共享预训练一致。warmup 的实际最终 checkpoint 自动传入 policy，阶段切换只继承可训练权重，optimizer 重新初始化。`--mode`、batch 参数、训练轮数与步数上限同时作用于两个阶段；warmup 的开发集验证保留预热动作日程，最终 QA 评估在 policy 结束后执行。

## SwanLab 与本地记录

一个完整方法对应一个 SwanLab run，内部阶段共用身份并累计 optimizer step。正式运行名称不含 `full` 或阶段名；`smoke`、`pilot` 分别保留对应档位标记，具体预算保存在 config 中。

| 正式 run 名称 | 同一 run 内的阶段 |
|---|---|
| `shared-pretrain-k64_<run-id>` | 两个动态方法共用的 pretrain |
| `memory-change-k64_<run-id>` | warmup → policy |
| `information-loss-k64_<run-id>` | warmup → policy |
| `icae-single-k64_<run-id>` | pretrain → QA |
| `icae-multi-k64_<run-id>` | pretrain → QA |
| `autocompressors-k64_<run-id>` | LM |

因此 `--method all` 完整流程共六个 run。试跑名称为 `<method>-k64_<mode>_<run-id>`，例如 `memory-change-k64_smoke_shared-01`，对应共享预训练名为 `shared-pretrain-k64_smoke_shared-01`；pilot 同理使用 `pilot` 标记。两个动态方法与共享预训练保持相同 `run-id` 后缀，config 另记录来源 checkpoint 的准确路径、step、SwanLab ID 和 URL。

阶段由独立训练进程执行：前一进程 finish，下一进程 resume 同一 SwanLab ID，云端仍显示一个 run。ICAE 从 pretrain 持续累计到 QA；动态方法仅累计自身 warmup 与 policy，不把共享预训练 step 加入子 run。最终评估追加到对应方法的最终累计 step。阶段内续训恢复 optimizer，跨阶段则重建 optimizer，训练语义不变。

同一系列的 runs 使用同一 group。默认正式 group 与目录名为 `capacity_<run-id>`，试跑为 `capacity-<mode>_<run-id>`，即 `capacity-smoke_<run-id>` 或 `capacity-pilot_<run-id>`；可用 `--group` 显式指定比较系列。

| 展示位置 | 内容 |
|---|---|
| `train` / `dev` | 累计 optimizer step 的增量曲线；损失按 `ae_lm_loss`、`qa_loss`、`lm_loss` 区分目标，另记录 slots、梯度范数与阶段边界表 |
| `resources` | 每步耗时、CUDA 峰值显存 |
| `evaluation` | 最后一个 checkpoint 的 NLL、EM、F1 汇总图；容量、成本、距离分层与生成样例表 |

完整动作、门控分数和逐题结果保存在本地；SwanLab 中的信息损失结果注明 `offline_oracle`。API key 不写入任务计划或配置快照。

正式产物根目录为 `artifacts/v3/capacity_<run-id>/`，试跑为 `artifacts/v3/capacity-<mode>_<run-id>/`：

```text
plan/<method>/                   本次方法流程的计划、解析配置、日志及 result.json
train/<run-name>/                experiment.json、唯一的 swanlab.json 及 SwanLab 本地日志
train/<run-name>/<stage>/        阶段 config.json、run.json、metrics.jsonl
  checkpoints/step-*.pt          该阶段可续训 checkpoint
eval/<run-name>/<stage>/         最终评估 trajectories.jsonl 与 summary.json
compare/<method>/               同题池质量—容量点 points.json / points.csv
```

这里的目录方法名用连字符，例如 `memory-change`、`shared-pretrain`；`dynamic`、`all` 各有自己的计划目录。同一系列可以分别启动两个动态方法，已有的方法任务和阶段产物不会覆盖。阶段失败后停止后续工作，在该方法的 `plan/<method>/result.json` 记录失败与已完成 checkpoint。中断恢复使用内部训练模块的 `--resume` 和原阶段配置；新独立实验使用新的 `--run-id`。

## 其他常用参数

| 参数 | 作用 |
|---|---|
| `--dry-run` | 打印预算、阶段依赖、解析配置和命令；不加载模型/数据，不连接 SwanLab，不创建实验目录 |
| `--gpus 4,5` | 逗号分隔的非重复物理卡号，训练进程数随卡数变化；默认 `0,1` |
| `--model-path /absolute/model/path` | 覆盖本地模型位置，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--micro-batch-size-per-gpu 2` | 每卡一次并行处理的样本/轨迹数，默认 2 |
| `--gradient-accumulation-steps 2` | 每次参数更新累积的 microbatch 数，默认 2；全局 batch 自动计算 |
| `--qa-batch-size 8` | 每条轨迹单次读取的 QA 数，默认 8 |
| `--init-checkpoint` | 指定动态方法的共享预训练 checkpoint，并从其 `run.json` 继承 `run-id` |
| `--run-id` | 系列标识；默认上海时间 `YYYYMMDD-HHMMSS`，外部初始化时继承来源 |
| `--threshold-i / --threshold-d / --threshold-g / --eta` | 覆盖门控阈值，记录于训练配置 |
| `--max-new-tokens 64` | 最终评估生成上限 |
| `--tracking disabled` | 仅保存本地记录；默认 online |
| `--swanlab-project` / `--group` | 项目与实验系列 |
| `--output-root` | 产物父目录，默认 `artifacts/v3` |

仓库路径由脚本开头的 `LWM_REPO_DIR` 常量统一指定，不接受同名环境变量覆盖，也不根据脚本位置推导。

本地测试验证预算、阶段衔接、数据选择、原生指标和评估图表构造；服务器上的实际显存、训练速度及云端展示需要通过第一轮 `smoke` 任务核验。
