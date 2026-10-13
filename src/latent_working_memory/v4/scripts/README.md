# 20261013_v4 统一运行入口

创建时间：20261013 10:12:17 Asia/Shanghai（UTC+08:00）

最后修订时间：20261013 17:23:57 Asia/Shanghai（UTC+08:00）

使用 [run.sh](run.sh) 启动新训练、恢复训练或单条 greedy 生成。模型结构与训练目标见 [v4 说明](../README.md)，默认配置见 [experiment.json](../../../../configs/v4/attention_matching/experiment.json)。以下命令从项目根目录执行，使用与 v3 共用的根 `.venv/` 环境。

## 1. 准备数据

- 训练文件统一为文档 QA JSONL：每行 `text` 保存完整正文，`qas` 保存对应的 `question`／`answer` 列表，详见 [v4 数据说明](../README.md#数据)。
- 默认读取 `data/delta-mem-qasper/train.jsonl`、`dev.jsonl`；目录还包含 `test.jsonl`、`preparation.json` 和说明。可从 [ModelScope 仓库](https://modelscope.cn/datasets/percyWeeei/latent-working-memory/files) 的 `delta-mem-qasper/` 获取。
- 从本地 raw 构造数据时，在项目根目录执行：

```bash
uv run --frozen python -m latent_working_memory.data_preparation.delta_mem_qasper \
  --config configs/data_preparation/delta-mem-qasper.json
```

- 构造阶段按论文固定划分，保留全部 QA；加载时将同一正文的不同 QA 展开为独立 episode，分别初始化 memory。
- 统一短答提示词按基座 chat template 分词，关闭 thinking；外层仅计算当前答案 CE。
- `max_seq_length` 仅过滤 train，预设 12288；`--max-seq-length 0`（配置省略或设为 `null`）关闭人为上限。实际模型窗口强制约束 train/dev，完整输入不截断。
- 输入至少为 `pending_size + recent_size + 2` tokens，答案从首次写入后的有效目标位置开始；默认最短输入为 386 tokens。
- 加载前检查原始 train/dev 正文无重叠，分词结果在内存中供各 epoch 复用，各阶段过滤数量单独记录。

## 2. 新训练

默认使用物理 GPU 0、1；可通过 `--gpus 0` 或 `--gpus 1` 使用单卡：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --config configs/v4/attention_matching/experiment.json \
  --gpus 0,1 \
  --run-id am-query_20261013 \
  --run-name qwen3-0.6b_conditioned-m32
```

显式 CLI 参数覆盖配置。省略 `--run-id` 时使用上海时间 `YYYYMMDD-HHMMSS`；`--run-name` 未指定且配置未设置时，根据模型名、query 模式与 slots 数生成，例如 `qwen3-0.6b_conditioned-m32`。新运行使用独立目录，已有产物不覆盖。

如需限制本次执行，在命令后增加 `--stop-after-steps 10`：在全局 optimizer step 达到 10 时停止，不改变配置中的 epoch 预算。停止点会执行开发集评估并保存恢复所需产物。

默认配置关闭在线跟踪，只记录本地结果。启用时设置 `--tracking online --swanlab-project 已有项目名`；`--group` 未配置时采用 `run-id`，可重复传入 `--tag study:名称` 等命名空间标签。数据来源通过配置的 `tags` 标记，QASPER 预设包含 `data:qasper`。凭据来自项目 `.env` 或终端的 `SWANLAB_API_KEY`，入口不会自动创建项目。

## 3. 恢复训练

从 `train/<run-name>/training-result.json` 的 `checkpoint` 字段或实际保存目录取得 checkpoint，将下例的 `global_step_N` 替换为真实目录名：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --resume --gpus 0,1 \
  --checkpoint artifacts/v4/am-query_20261013/train/qwen3-0.6b_conditioned-m32/checkpoints/global_step_N
```

- 恢复沿用原输出目录、模型、数据、训练超参数和 SwanLab run 身份；不接受 `--config`、数据或训练超参数覆盖。
- checkpoint 须属于 `<实验系列>/train/<run-name>/checkpoints/global_step_N/`，入口根据原输出目录定位 `plan/` 与 `eval/`。
- GPU 可以在允许范围内重新选择，但卡数必须与原训练一致。保留原训练目录中的 `run.json`、`tokenizer/`、checkpoint，以及启用跟踪时的 `swanlab.json`。
- 可以用 `--stop-after-steps N` 设置本次停止位置；`N` 是恢复前后统一的绝对 global step，不是新增步数。
- 原配置另存为 `plan/<run-name>/resume-config.json`，日志继续追加到 `train.log`，`job.json` 记录 `action=resume`。

恢复读取新增模块权重、optimizer、数据加载器进度和各 rank RNG。保存点位于完成一个全局 batch 后；每条 QA episode 重新初始化 slots，不恢复文档内部尚未结束的流。

## 4. 单条生成

准备 UTF-8 纯文本 prompt，使用一张 GPU 从 checkpoint 生成：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --generate-only --gpus 0 \
  --checkpoint artifacts/v4/am-query_20261013/train/qwen3-0.6b_conditioned-m32/checkpoints/global_step_N \
  --prompt-file data/attention-matching/prompt.txt \
  --max-new-tokens 128
```

默认写入原实验系列的 `eval/<run-name>/greedy-sample.json`；可用 `--output` 指定文件。结果包含 prompt、生成文本和 token IDs。生成复用原 run 的模型 revision 与 tokenizer，并继续执行局部 TTT；prompt 与新增 tokens 的总长度不能超过基座绝对位置上限。

当前入口提供训练中的开发集 CE 和单条生成，不包含 QA benchmark 或跨运行比较。

## 5. 关键参数

| 参数 | 用途与默认值 |
|---|---|
| `--config` | 新训练配置，默认 `configs/v4/attention_matching/experiment.json` |
| `--gpus` | 默认 `0,1`；其他 GPU 编号须显式指定，生成必须选择一张卡 |
| `--run-id`／`--run-name` | 新训练的实验系列／运行名称，默认规则见第 2 节 |
| `--output-root` | 新训练产物父目录，默认 `artifacts/v4` |
| `--model-path` | 覆盖本地基座模型路径；HF repo ID 在 JSON 配置中设置 |
| `--train-file`／`--dev-file` | 覆盖训练／开发集 canonical JSONL 文件 |
| `--global-batch-size` | 每次 optimizer step 的全局样本数，样例配置为 8 |
| `--micro-batch-size-per-gpu` | 每卡一次并行处理的轨迹数，样例配置为 2 |
| `--max-seq-length` | train 完整输入的人为上限，默认 12288；0 关闭。模型窗口始终过滤 train/dev，不截断 |
| `--stop-after-steps` | 本次执行停止于指定绝对 global step，省略则按 epoch 预算运行 |
| `--tracking` | `disabled`／`online`；未传时沿用配置，默认配置为 disabled |
| `--swanlab-project`／`--group`／`--tag` | 已有项目、实验分组、可重复指定的标签；仅用于新训练 |
| `--resume`／`--checkpoint` | 从 checkpoint 目录恢复原 run；不接受新训练配置覆盖 |
| `--generate-only`／`--prompt-file` | 从 checkpoint 执行单条 greedy 生成，prompt 为纯文本文件 |
| `--output`／`--max-new-tokens` | 生成结果文件／新增 token 上限；上限默认 128 |

CLI 的相对文件路径以**项目根目录**为基准；配置内部的数据路径按**该配置所在目录**解析。入口将解析后的绝对数据路径保存到运行配置快照，并按本次系列与运行名称设置训练输出目录。

全局 batch 与每卡 microbatch 分别控制 optimizer 更新和单次前向的规模。样例双卡设置为全局 batch 8、每卡 microbatch 2，一个完整全局 batch 对应每卡累积两次 microbatch。其余模型和训练设置直接修改 JSON 配置；完整参数见：

```bash
bash src/latent_working_memory/v4/scripts/run.sh --help
```

## 6. 产物与进度

```text
artifacts/v4/<run-id>/
├── plan/<run-name>/
│   ├── config.json                 新训练配置快照
│   ├── resume-config.json          恢复时保存的配置
│   ├── job.json                    本次训练／恢复动作与执行命令
│   ├── train.log                   训练日志；恢复时追加
│   ├── generate.log                单条生成日志
│   └── result.json                 训练／恢复执行结果
├── train/<run-name>/
│   ├── run.json、tokenizer/        配置、运行身份与 tokenizer
│   ├── data-statistics.json        原始划分、人为上限、模型窗口及监督过滤统计
│   ├── metrics.jsonl              train/dev 指标、耗时与显存
│   ├── checkpoints/global_step_N/ 新增权重、optimizer、加载器与 RNG
│   ├── swanlab.json                启用跟踪时保存的云端身份
│   └── training-result.json        训练结果与最后 checkpoint
└── eval/<run-name>/greedy-sample.json
```

终端和 `train.log` 展示训练进度；开发集在配置的间隔、epoch 结束及主动停止点评估。在线跟踪时，训练和验证追加到同一 SwanLab run。训练子进程的异常退出码写入 `result.json`；参数与配置错误直接显示在终端，具体指标和 checkpoint 见原训练目录。
