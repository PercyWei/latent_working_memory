# 20261031_实验执行规范与参考骨架

创建时间：20261031 16:16:25 Asia/Shanghai（UTC+08:00）

最后修订时间：20261031 17:46:40 Asia/Shanghai

本文规定实验从**创建计划、训练与保存、恢复、评估到结果提交**的完整执行规则，以 v4.1 为基准。新方法沿用这些规则，模型、loss、记忆机制、输入编码和评分由方法模块实现。

本文是执行规范；[代码模板](template/README.md)实现通用执行流程，并提供可运行的 CPU 任务，[产物示例](examples/README.md)说明 v4.1 的文件格式。正式方法配置放在 `configs/`，运行产物放在 `artifacts/`。

## 1. 运行模式

四种模式互斥；未指定模式时执行完整运行。

| 模式 | 计划与配置来源 | 执行范围 | 结束条件 |
|---|---|---|---|
| 完整运行 | 输入 JSON＋显式 CLI 覆盖；新建训练和最终 test 计划 | train、dev、最后一个 checkpoint 的 test | 训练及最终 test 均成功 |
| `--train-only` | 输入 JSON＋显式 CLI 覆盖；只建训练计划 | train、dev、checkpoint 保存 | 训练预算完成；不创建或执行最终 test 计划 |
| `--resume` | 原计划＋最新完整 checkpoint | 恢复剩余训练，或继续原计划中的最终 test | 原运行计划完成 |
| `--evaluate-only` | checkpoint＋新建或已有的任务计划 | 指定 checkpoint 的 test／benchmark | 本次选择的评估项均完整提交 |

- 新训练使用新 run；已有非空运行目录拒绝覆盖。
- `resume` 只接受一个 checkpoint，不接受训练配置覆盖，也不改变原 `train-only` 安排。
- 独立评估接受同一 run 的一个或多个 checkpoint；`--eval-run` 可选择当前保留的全部完整 checkpoint。

## 2. 生命周期

### 2.1 正常执行顺序

| 阶段 | 执行规则 | 进入下一阶段的条件 |
|---|---|---|
| ① 参数定案 | 识别模式，合并显式参数，校验配置，解析 run 身份与目录 | 参数合法，目标目录符合当前模式 |
| ② 保存计划 | 保存训练计划；完整运行同时保存最终 test 计划 | 计划已落盘，计算进程尚未启动 |
| ③ 初始化训练 | 读取计划、选择数据、绑定身份、推导实际总步数；需要时恢复状态 | 实际配置与数据身份校验通过 |
| ④ 训练与 dev | 按 optimizer step 更新、记录、验证与保存 | 达到原计划的实际训练预算 |
| ⑤ 最终评估 | 等待全部训练进程退出，核对最终 checkpoint，再复用独立评估流程 | 原计划要求最终 test；`train-only` 在此之前结束 |
| ⑥ 提交结果 | 校验评估样本完整，提交逐题结果与汇总 | 所有请求样本完成，结果身份一致 |

独立评估从确定任务计划开始，直接进入评估流程。恢复先校验原计划与 checkpoint，再进入尚未完成的阶段；训练预算已经完成时不再更新参数。

### 2.2 失败与中断后的行为

| 当前状态 | 再次执行时的处理 |
|---|---|
| 训练失败／中断，存在完整 checkpoint | 显式 `resume`，从该 run 最新完整 checkpoint 恢复 |
| 训练失败／中断，没有完整 checkpoint | 不视为可恢复训练；重新启动须使用新 run |
| 训练完成，最终 test 未完成 | `resume` 跳过训练，继续原计划中的 test |
| 评估部分完成 | 沿用原 `eval-id`，保留已完成样本，只调度缺失项 |
| 评估完整且身份一致 | 复用结果，不重新计算 |

失败阶段不进入后续阶段。中断时停止并等待全部子进程退出，防止旧 worker 继续写入。恢复由用户再次执行入口触发，不设置自动反复重试循环。

## 3. 配置与计划管理

### 3.1 参数处理规则

| 项目 | 规则 |
|---|---|
| 新运行优先级 | 输入 JSON → 合并实际出现的 CLI 参数 → 补默认值 → 校验 → 解析路径与标识 → 保存计划 |
| 省略与 `null` | 未传 CLI 参数不覆盖 JSON；显式 `null` 按字段定义处理。可用 `argparse.SUPPRESS` 区分省略值 |
| 输入校验 | 未知字段、非法数值、冲突模式及不适用当前模式的参数直接报错 |
| 路径 | 启动入口固定项目根目录；支持 `~`，持久化计算路径使用解析后的绝对路径 |
| 运行标识 | 方法设置与 `run-id` 共同标识训练 run；默认 `run-id` 为 `YYYYMMDD-HHMMSS` |
| 进程配置 | worker 只读取已保存计划，不各自重新合并 JSON、CLI 和默认值 |
| 设备 | 使用 `CUDA_VISIBLE_DEVICES` 指定 GPU；本项目默认只分配物理 GPU 0、1 |

### 3.2 计划与实际配置

| 类型 | 规范位置 | 保存内容 | 更新规则 |
|---|---|---|---|
| 训练计划 | `plan/config.json`、`plan/train.json` | seed、模型／方法设置、数据选择、请求预算、batch、保存策略与输出目录 | 计算前保存；恢复沿用原值 |
| 评估计划 | `plan/eval-<eval-id>/<task>.json` | checkpoint 所属 run、任务、数据、选样与生成设置 | 首次须指定数据并在执行前保存；之后省略值继承，显式值须一致 |
| 实际训练配置 | `train/config.json` | 原计划加数据选定后推导的实际总步数等执行信息 | 初始化训练后保存，不反写请求计划 |
| 数据记录 | `train/data.json` | 选样统计、编码协议和数据／缓存身份 | 数据选定后保存，恢复时重新构造并核对 |
| 评估请求 | 每个 checkpoint 的 `eval.json` | 当前计算的 checkpoint、实际样本内容、方法配置与评估协议 | 分配样本前保存，复用结果时严格核对 |

计划保存**请求预算**，实际配置和 checkpoint 保存**推导预算**。恢复时根据原计划、选样数与全局 batch 重新推导，不能用当前默认值替换原设置。

### 3.3 允许改变的设置

| 操作 | 可改变 | 必须保持 |
|---|---|---|
| 创建新 run | 本次实验的配置与数据设置 | 正式配置格式和执行规则 |
| 恢复训练 | 重新启动进程 | 原目录、方法、数据、选样、batch、预算、进程数和缓存身份 |
| 创建独立评估 | 首次须显式指定数据目录；可设置评估预算和并行度 | checkpoint 中的模型与方法设置 |
| 重跑同一评估 | worker 数和 GPU 分配 | 同一 `eval-id/task/checkpoint` 的请求身份；任务明确允许的增量扩展除外 |

## 4. 数据与训练执行

### 4.1 数据规则

| 项目 | 规则 |
|---|---|
| 数据接口 | 明确 train／dev／test 来源、正式格式、tokenizer、模板及正文／问题／答案边界 |
| 长度筛选 | 训练正文上限针对原始正文 token 数；超限整条排除，不截断正文。方法特有下限单独定义 |
| 模型窗口 | 另行检查实际编码输入，不静默截断；评估记录请求／可用生成预算与停止原因 |
| 数量限制 | 过滤后按稳定 ID 和 seed 确定子集；选样不依赖 GPU 数、调度顺序或中断位置 |
| 划分与身份 | 检查原始文档集合重叠；绑定选中顺序、实际 token 内容及必要的方法缓存身份 |
| 数据产物 | 保存筛选统计与身份；仅过滤、抽样或更换 tokenizer 时不保存派生文本副本 |

### 4.2 Batch 与训练预算

设 `W` 为训练进程数、`M` 为每卡 microbatch、`A` 为梯度累积次数、`N` 为实际选中样本数：

```text
B = W × M × A
steps_per_epoch = ceil(N / B)
total_steps = min(epochs × steps_per_epoch, requested_max_steps)
```

至少提供一个有效训练预算；只提供 epochs 或 max steps 时使用对应项。Scheduler、验证、保存和指标横轴均使用真实 optimizer step。

### 4.3 每次更新的执行规则

1. 全局游标取最多 `B` 条样本，再按 `indices[rank::W]` 分片；各 rank 推进同一游标。
2. 本地按 microbatch 累积样本 loss 之和；跨 rank 求和后，除以本次全局实际样本数，再统一裁剪、更新 optimizer 和 scheduler。
3. Epoch 尾批不丢弃、不补样本、不跨 epoch 拼接；空 rank 仍参加相同顺序的 collective。
4. Dev 在配置间隔、epoch 末和最终步执行，按定义分母汇总，不执行参数更新；记录时不为未执行的 dev 补零。
5. Checkpoint 按配置频率保存，epoch 末和最终步也保存；公共训练产物由 rank 0 写入。

方法负责定义每样本 loss、内部归一化及 metric 分母；使用 DDP 默认梯度平均等机制时，须相应推导缩放，保持上述全局实际样本平均语义。

## 5. Checkpoint 与训练恢复

### 5.1 保存规则

| 项目 | 规则 |
|---|---|
| 保存边界 | 仅在完整 optimizer step 结束后保存，不持久化未完成的梯度累积 |
| 完成标记 | 先写临时文件，再原子替换 `state.pt`；空目录和 `state.tmp` 不算完整 checkpoint |
| 保留策略 | 新 checkpoint 成功提交后，才删除超出保留数量的旧 checkpoint |
| 模型状态 | 保存全部可训练状态；冻结基座由配置绑定，样本临时状态由方法确定是否需要持久化 |
| 更新状态 | 保存 optimizer、scheduler、已完成 step；启用 scaler 等机制时一并保存必要状态 |
| 数据与随机状态 | 保存 epoch、offset、样本排列、游标 generator，以及各 rank 的通用和方法专用 RNG |
| 实验绑定 | 保存实际配置、数据身份及必要的方法缓存身份 |

### 5.2 恢复规则

1. 只接受原 run **最新完整 checkpoint**；规范路径、目录 step 与内部 step 必须一致。
2. 核对原计划、实际预算、数据内容、选样顺序与缓存身份；保持原 `world_size`。
3. 恢复模型、optimizer、scheduler、数据游标及各 rank 随机流；校验游标、scheduler 与 optimizer step 一致。
4. 本地指标在 checkpoint step 前须完整且唯一；回退其后未提交记录和末尾未完成行，再继续追加。
5. 达到预算后只继续原计划中的最终评估；原 `train-only` 运行仍不执行 test。

任一校验失败直接报错。改变实验设置应创建新 run，不通过重新初始化缺失状态、覆盖配置或更换目录继续原运行。

## 6. 评估与失败重跑

### 6.1 调度与身份规则

| 项目 | 规则 |
|---|---|
| Checkpoint 选择 | 同一 run 的 checkpoint 按真实 step 升序执行；完整运行只选最后一个 |
| 多卡分工 | 多张 GPU 共同评估一个 checkpoint 的不同样本；当前项汇总后再处理下一项 |
| 样本独立性 | 状态和随机性不依赖 worker 分配或先前样本；局部适应是否启用 autograd 由方法确定 |
| `eval-id` | 自动 test 默认 `auto`；独立评估默认时间戳；重试须显式沿用原标识 |
| 缓存复用 | 请求绑定 checkpoint、模型／方法、seed、实际数据及输入／生成／停止／评分协议；完整一致则复用，变化则报错 |
| 失败续算 | 保留所有已完成样本，重新分配缺失项；允许调整 worker 数和 GPU 分配 |

### 6.2 进度与结果提交顺序

1. 调度器持锁校验请求、合并旧进度，生成本轮 `work.json`；分工时不得有旧 worker 继续运行。
2. 每个样本完成计算与评分后，追加一条完整进度并 flush，再处理下一条。
3. 重试只裁掉末尾未完成行；完整坏行、未知样本、重复 ID 或提交冲突直接报错。
4. 所有目标 ID 齐全后，按原始样本顺序原子写入 `details.jsonl`，最后原子提交 `summary.json`。
5. `summary.json` 只有与请求及完整 details 匹配才表示完成；`status.json` 仅记录本轮调度状态。

v4.1 的 RULER 长度集合只允许保持或扩展，旧长度的数据与协议保持一致，根汇总覆盖本轮全部请求。模板当前实现固定 test 样本集合；增加 benchmark 时，须在任务适配层实现其选样、身份和增量规则。

## 7. 记录、展示与代码索引

### 7.1 记录与展示规则

| 范围 | 规则 |
|---|---|
| 训练与 dev | `metrics.jsonl` 每 optimizer step 一条；SwanLab 使用同一训练 run、真实 step 的增量曲线，恢复沿用原身份 |
| 独立评估与最终 test | 保存本地 summary／details，由 dashboard 展示，不向 SwanLab 追加评估记录 |
| 日志与状态 | 保存命令、设备、起止时间、退出码和错误；日志与 status 不替代 checkpoint 或完整结果 |
| 质量比较 | 核对基座、实际输入和参考答案、输入／生成预算及评分口径；允许方法和运行环境不同 |
| 成本比较 | 另行统一硬件、批量、并行度和测量范围，区分样本累计耗时与本次调度墙钟 |

完整文件布局与字段说明见[产物示例](examples/README.md)。

### 7.2 代码模板与方法边界

| 模块 | 固定职责 | 新方法需要调整 |
|---|---|---|
| [config.py](template/config.py)、[cli.py](template/cli.py)、[run.py](template/run.py) | 配置校验、显式覆盖、模式限制与运行入口 | 方法配置、命名和需要公开的参数 |
| [execution/](template/execution/) | 计划、生命周期、进程清理、原子状态、随机流及评估进度 | 方法特有的身份和计划字段 |
| [training/](template/training/) | 游标、microbatch 累积、梯度归约、dev、保存与恢复 | 方法新增的必要训练状态 |
| [evaluation/](template/evaluation/) | checkpoint 串行、样本并行、缺失项续算、最终提交 | 新任务的调度与增量规则 |
| [method.py](template/method.py) | 可替换的方法接口及 CPU 示例 | 数据与身份、模型、逐样本损失、预测、评分和模型状态 |

现有 reference 已整合到模板的 `execution/`。复制模板后沿用执行流程，在方法边界实现具体算法；模板不导入 v4／v4.1。示例任务使用序列回归，checkpoint 将方法专用随机流命名为 `method_rng`；v4.1 产物示例中的 `query_rng` 属于其方法字段。

[CPU 验证入口](verify_cpu.py)通过真实子进程验证四种模式、双 rank 尾批与空 rank、信号中断恢复及评估缓存续算，运行方式见[模板说明](template/README.md)。

实现依据：[入口与计划](../../src/latent_working_memory/v4_1/training/run.py) · [数据与训练](../../src/latent_working_memory/v4_1/training/train.py) · [保存恢复](../../src/latent_working_memory/v4_1/training/checkpoint.py) · [评估调度](../../src/latent_working_memory/v4_1/evaluation/run.py) · [评估进度](../../src/latent_working_memory/v4_1/evaluation/shards.py)。
