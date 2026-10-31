# 20261031_实验代码模板

创建时间：20261031 17:42:57 Asia/Shanghai

最后修订时间：20261031 20:49:15 Asia/Shanghai

本模板将 v4.1 的计划、训练、恢复和评估流程整理为一套可复制的 Python 代码。执行规则见[主规范](../README.md)，原 reference 已整合到 `execution/`。

默认任务是小型序列回归：用 MLP 根据数值序列预测标量，执行真实前向、反向和 optimizer step。它用于运行骨架验证；新方法替换计算接口后沿用相同流程。

## 1. 运行方式

从项目根目录运行，使用根目录 `.venv/`。默认 CPU、单进程，读取 [config.json](config.json)，数据来自 [datasets/](datasets/)。

```bash
runner=scaffolds/experiment/template/scripts/run.sh

# 完整运行：train、dev、最后一个 checkpoint 的 test
bash "$runner" --run-id cpu-demo

# 仅训练，不创建最终 test 计划
bash "$runner" --train-only --run-id cpu-train

# 原运行恢复；已完成训练时仅继续或复用自动 test
run=artifacts/experiment-template/mean-regression-h8_cpu-demo
bash "$runner" --resume --checkpoint "$run/train/checkpoints/global_step_4"

# 独立评估该 run 保留的全部 checkpoint
bash "$runner" --evaluate-only --eval-run "$run" --eval-id manual \
  --test-data scaffolds/experiment/template/datasets
```

- 默认配置的单进程训练共 4 步；中断恢复时，使用原 run 当前最新完整 checkpoint。
- CPU 多进程添加 `--cpu-processes 2`；CUDA 使用 `CUDA_VISIBLE_DEVICES=0,1 bash "$runner" --device cuda`，进程数由可见 GPU 决定。
- 训练进程全部退出后才开始自动 test；多个 checkpoint 按 step 串行，每个 checkpoint 由全部评估 worker 按样本分工。
- 评估重试沿用同一 `eval-id`；已有计划可省略数据和预算，显式传入时必须一致。worker 数可以改变。

## 2. 代码职责与替换边界

| 位置 | 职责 |
|---|---|
| [run.py](run.py)、[cli.py](cli.py) | 四模式分流、显式参数覆盖、计算进程启动 |
| [config.py](config.py) | 默认值、正式配置字段及边界校验 |
| [execution/plans.py](execution/plans.py) | 创建与沿用计划、实际预算推导、恢复校验 |
| [execution/lifecycle.py](execution/lifecycle.py) | 训练状态、自动 test、checkpoint 评估顺序与失败状态 |
| [execution/processes.py](execution/processes.py)、[runtime.py](execution/runtime.py) | 进程输出与清理、设备、通用 RNG |
| [execution/state.py](execution/state.py) | 原子写入、checkpoint 保留、指标回退、进度复用与最终提交 |
| [training/](training/) | 全局游标、梯度累积与归约、dev、训练保存和恢复、SwanLab |
| [evaluation/](evaluation/) | 缺失样本分片、独立 worker、逐题提交与汇总 |
| [method.py](method.py) | 数据、模型、损失、预测与评分的具体计算 |

### 2.1 方法接口

| 接口 | 返回值与契约 |
|---|---|
| `load_samples` | 样本、内容身份、筛选统计；顺序稳定，身份覆盖实际输入与目标，统计包含原始文档 ID |
| `method_name` | 简洁的方法设置名称，用于构造运行目录 |
| `build_model`、`build_optimizer` | 模型和 optimizer；全部 rank 的可训练参数顺序一致 |
| `compute_losses` | 每个值为 `[batch]` Tensor 的字典，包含 `loss`；由引擎按全局实际样本数平均 |
| `predict` | 单样本计算与评分；返回含 `id` 的 JSON 记录，自行初始化或清理样本临时状态 |
| `summarize`、`evaluation_protocol` | 汇总指标，以及影响输入、生成和评分的协议身份 |
| `trainable_state_dict`、`load_trainable_state_dict` | 全部可训练状态的 CPU 保存与恢复；冻结基座须由方法配置绑定 |

- 示例数据每行是 `{"id": "...", "values": [1, 2], "target": 1.5}`。改为文本任务时，同步替换数据格式、长度统计、编码和身份构造。
- 当前所有 loss／dev 指标均按样本平均；token 加权等其他分母需同步调整方法返回值和归约逻辑。
- 训练引擎显式归约梯度，不使用 DDP 的隐式平均；新增方法保留这一执行约定。
- 方法专用随机流使用独立 CPU `torch.Generator`，与通用 RNG 分别保存。增加新的持久状态时，同步修改其保存、恢复和验证路径。

### 2.2 复制为新方法

1. 将整个 `template/` 复制到新方法包，保留相对导入和子目录。
2. 实现 `method.py`，同步修改配置、数据和方法命名；根据需要增加公开参数。
3. 检查方法专属状态及评估协议绑定，更新 SwanLab 固定标签。
4. 用小数据运行完整流程与中断恢复，再接入正式任务。

启动脚本根据自身位置解析模块名，可用于仓库内普通包或 `src/` 下的包。模板当前提供固定 `test` 任务；添加其他 benchmark 时，在评估层补充其计划、身份和增量规则。

## 3. 产物与恢复

```text
<output-root>/<方法设置>_<run-id>/
├── plan/config.json、train.json、eval-<eval-id>/test.json
├── train/
│   ├── config.json、data.json、metrics.jsonl、status.json、summary.json
│   ├── workers.log、workers.commands.json
│   └── checkpoints/global_step_<step>/state.pt
└── eval/<eval-id>/test/
    ├── status.json
    └── global_step_<step>/
        ├── eval.json、work.json、workers.log、workers.commands.json
        └── progress/worker_<rank>.jsonl、details.jsonl、summary.json
```

| 规则 | 当前实现 |
|---|---|
| 计划与实际配置 | 启动 worker 前保存请求计划；选样后推导实际步数，写入 `train/config.json` |
| 保存边界 | 只在完整 optimizer step 后提交 `state.pt`，成功后清理超出保留数量的旧 checkpoint |
| Checkpoint 内容 | `config`、`model`、`optimizer`、`scheduler`、`cursor`、`rng`、`method_rng`、`binding` |
| 恢复条件 | 原目录、最新完整 checkpoint、原进程数和训练设备类型；配置与数据身份严格一致 |
| 指标恢复 | 保留到 checkpoint step 的连续记录，回退之后的记录及未完成末行 |
| 评估复用 | 固定 checkpoint 内容、实际数据和协议身份；完整结果复用，部分结果只补缺失项 |
| 最终提交 | 所有目标样本齐全后写 details，最后提交 summary；status 仅描述执行状态 |

SwanLab 默认关闭。开启时显式指定已有项目，如 `--tracking online --swanlab-project <project> --group <group>`；train／dev 使用同一 run，恢复沿用原身份。评估只保存本地产物，后续 dashboard 可读取 `summary.json` 和 `details.jsonl`。

## 4. 参数

方法超参数放在 JSON 的 `model` 中。下表列出执行参数；省略 CLI 参数时保留配置值。

### 4.1 运行配置与训练预算

| 参数 | 用途／默认值 |
|---|---|
| **运行方式** | |
| `--train-only` | 只训练及 dev 验证，不创建或执行自动 test |
| `--resume` | 从单个 checkpoint 恢复，沿用原运行计划，须指定 `--checkpoint` |
| `--evaluate-only` | 独立评估已有 checkpoint，须指定 `--checkpoint` 或 `--eval-run` |
| **配置与设备** | |
| `--config` | JSON 配置文件，默认模板 `config.json` |
| `--seed` | 随机种子，默认 42，控制初始化、选样与打乱 |
| `--device` | 计算设备，支持 `cpu`（默认）和 `cuda`；GPU 通过 `CUDA_VISIBLE_DEVICES` 分配 |
| `--cpu-processes` | CPU 训练／评估进程数，默认 1；仅用于 CPU |
| **目录与运行标识** | |
| `--output-root` | 产物父目录，默认 `artifacts/experiment-template` |
| `--run-id` | 训练运行标识，默认 `YYYYMMDD-HHMMSS` |
| **数据集与筛选** | |
| `--train-data` | train／dev 数据目录，默认模板 `datasets/` |
| `--max-train-data-tokens` | train／dev 原始序列长度上限，默认无限制；超限整条排除 |
| `--max-train-samples` | 过滤后的训练样本数上限，默认 `null`（全部） |
| **训练时长** | |
| `--epochs` | 训练轮数，默认 2；`null` 表示只按 step 预算限制 |
| `--max-steps` | optimizer step 上限，默认 `null`（不额外限制） |
| **批处理与优化** | |
| `--micro-batch-size-per-gpu` | 每卡一次并行处理的样本数，默认 2；CPU 下表示每进程的值 |
| `--gradient-accumulation-steps` | 每次 optimizer 更新前累积的 microbatch 数，默认 2 |
| `--learning-rate` | 学习率，默认 0.01 |

- 三种模式参数互斥；都未指定时执行完整运行。
- `epochs` 和 `max-steps` 至少保留一项有效预算，同时设置时取较小预算。
- 全局 batch = 训练进程数 × 每进程 microbatch × 梯度累积次数。

### 4.2 验证、评估与结果记录

| 参数 | 用途／默认值 |
|---|---|
| **Checkpoint** | |
| `--save-every` | 保存频率，支持正整数 optimizer step 间隔或 `after_each_epoch`（默认） |
| `--save-total-limit` | 保留最近成功保存的 checkpoint 数，默认 2 |
| `--checkpoint` | 恢复时指定单个 checkpoint；独立评估支持同一 run 的一个或多个 checkpoint |
| `--eval-run` | 指定运行根目录，评估其 `train/checkpoints/` 下保留的全部完整 checkpoint |
| **Dev 验证** | |
| `--max-dev-samples` | 过滤后的 dev 样本数上限，默认 `null`（全部） |
| `--dev-every` | dev 验证的 optimizer step 间隔，默认 2 |
| **Test 评估** | |
| `--eval-id` | 自动 test 默认为 `auto`，独立评估默认为 `YYYYMMDD-HHMMSS`；重试显式沿用 |
| `--test-data` | test 数据目录；完整运行默认沿用训练数据目录，新建独立评估计划时须指定 |
| `--max-test-samples` | test 样本数上限，默认 `null`（全部） |
| `--max-new-tokens` | 任务生成预算，默认 128；本回归示例无生成过程，不使用该预算 |
| **SwanLab** | |
| `--tracking` | 记录方式，支持 `disabled`（默认）和 `online` |
| `--swanlab-project` | SwanLab 项目，默认未指定；开启 online 时须显式指定 |
| `--group` | SwanLab 实验分组，默认未指定 |

- epoch 末和最终步始终执行 dev 并保存 checkpoint。
- `--checkpoint` 与 `--eval-run` 互斥。

恢复只接受 checkpoint 和设备／CPU 进程设置；独立评估只接受 checkpoint 来源与评估设置。未知字段、不适用当前模式的参数及冲突设置直接报错。

## 5. CPU 验证

```bash
.venv/bin/python -m scaffolds.experiment.verify_cpu
```

- 双 rank 执行四种模式，覆盖尾批及空 rank，验证完整运行只自动评估最后一个 checkpoint。
- 训练保存后发送真实 SIGTERM，恢复时修剪未完成指标行；与连续训练比较模型、optimizer、scheduler、cursor 和各 rank RNG。
- 评估部分提交后中断，改变 worker 数续算；验证已提交记录不变、缺失项补齐、完整缓存复用。
- 默认使用临时目录并清理；添加 `--output-root /tmp/experiment-verification` 可保留产物与验证日志，目录须为空。

验证使用 CPU，不连接 SwanLab，不验证 CUDA 吞吐或显存。
