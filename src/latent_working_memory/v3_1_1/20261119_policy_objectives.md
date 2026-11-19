# 20261119_v3.1.1 边界策略奖励与 PPO 修改方案

创建时间：20261119 14:52:11 Asia/Shanghai

最后修订时间：20261119 16:38:33 Asia/Shanghai

## 范围与约束

本次在 dev 分支修改 v3.1.1 的配置、边界策略训练和现行文档。预训练与 compressor_only QA 训练保持原有逻辑；SCST／GRPO 的数据、学习率、batch、记忆成本权重及现有终端损失归约不变。完整方法公式见[方法说明](README.md#qa-与边界后训练)。

## verl 接入执行计划

本轮将 SCST／GRPO／PPO 的通用算法与执行能力接入已安装的 verl 0.8.0。SCST terminal 使用原生 ReMax，GRPO terminal 使用原生 GRPO，PPO 使用原生 GAE 与优势白化；两种 stepwise 相对优势保留任务适配。策略损失采用原生 vanilla，包含 dual-clip 与 log 概率比截断；价值损失采用原生 value clipping。奖励定义、分段、recent／pending、两种 mask、损失权重与归约、数据和其他超参数保持原设置。

PPO 采用原生 actor／critic 独立更新组织：先执行 critic 的全部内轮，再执行 actor 的全部内轮，分别优化和裁剪；同批旧概率、旧价值、优势和价值目标固定。`training.loop.critic_warmup` 默认 0，沿用从 1 开始的 rollout 批次阈值：0／1 从首批更新 actor，11 表示前 10 批只更新 critic。仅 PPO 配置可以包含该字段。恢复保存 rollout、actor 和 critic 的实际更新进度。PPO 以 critic 实际更新数为主步数，其他策略以 actor 更新数为主步数；两个 PPO scheduler 各自推进，温度、dev、保存和图表沿用训练主步数，预热占原预算。

执行顺序为配置与算法适配、原生执行及预热门控接入、文档同步与检查。本轮实测结果记录在“验证与交付”。

## 奖励和优势

固定 recent 先从正文移出，成本使用 `c=((BK+P)/(D−R))^p`；完整保留比例独立记录。terminal 仅在最终事件给出 `Z=−QA_NLL−αc`。stepwise 对各真实事件给出 `−α(c_t−c_{t−1})`，最终再加入 QA 奖励；强制收尾不重复扣完整成本。

| 策略 | terminal | stepwise |
|---|---|---|
| SCST | 原生 ReMax：随机轨迹最终得分减同篇 greedy 最终得分 | 同正文读取位置的折扣 reward-to-go 差 |
| GRPO | 原生 GRPO：同篇采样组最终得分标准化并分配给各自主动作 | 同篇采样组所有真实事件奖励统一标准化，再累计折扣 reward-to-go |
| PPO | 有效 gamma=1 的原生 GAE 与优势白化 | 使用配置 gamma 的原生 GAE 与优势白化 |

奖励结算与优势统计使用 FP32，避免混合精度舍入丢失小幅 QA 或成本差异。所有 terminal 分支的有效 gamma 固定为 1。stepwise 读取配置 gamma，默认 1，范围 [0,1]。PPO 的 gae_lambda 独立配置，范围 [0,1]，必须显式提供；terminal 不覆盖它。

SCST 按正文读取位置对齐，不按自主动作或记忆块序号对齐。greedy 与随机路径可以有不同的 memory 和 pending，因此该参考值不解释为同一状态的反事实动作评价。GRPO 的事件统计使用样本标准差，排除 padding 与初始 c₀ 占位，保留奖励值为零的真实事件及强制事件。PPO 的优势白化在完整 rollout 的全局真实事件上计算，critic 目标使用白化前的 GAE。

## 配置与实现分工

- objective 使用统一 group_size 表示每正文随机轨迹数。SCST／PPO 至少为 1，GRPO 至少为 2；greedy 不计入 group_size。
- strategy 支持 scst、grpo、ppo，reward_mode 为 terminal 或 stepwise，优势算法由二者确定。
- GRPO 的两种奖励均配置 advantage_epsilon，预设沿用 1e-6。
- PPO 仅支持 boundary_only，独立创建 value head，冻结 Writer、memory embeddings 和 Reader。value head 复用 boundary_learning_rate。
- PPO 的 ppo_epochs 默认 4，value_loss_weight 默认 0.5。post_k256_ppo.json 使用 group_size=1、gae_lambda=0.95；0.95 是预设值，不是系统默认。
- training.loop.critic_warmup 默认 0，仅 PPO 配置可以包含该字段。原生批次从 1 开始，0／1 首批更新 actor，11 表示前 10 批只更新 critic。
- optimizer 的学习率字段按 update_mode 区分：compressor_only 仅含 compressor_learning_rate，boundary_only 仅含 boundary_learning_rate，joint 同时包含两者。PPO 的 critic 与边界共用后者的数值，各自独立更新。
- training.backward 仅用于 joint 策略训练，以 bptt_steps 控制压缩器跨段梯度；其他模式省略该分组。training.dev.max_new_tokens 仅用于 QA 生成验证，AE／LM 不配置。
- checkpoint 的 model 状态保存 memory_embeddings、adapter、boundary、value_head；非 PPO 的 value_head 为 null。恢复严格按当前格式读取，无旧格式自动转换。

完整事件序列用于奖励、折扣回报和 GAE，decision_mask 仅控制 actor 损失。pending 不足的强制等待与末尾强制压缩不产生 actor 梯度，自主选择等待仍参与更新。首次读入、各固定读取位置以及收尾的状态特征对应同一事件时钟。

PPO 从完整 rollout 缓存冻结特征，固定旧动作概率、旧价值预测、优势和价值目标，多轮优化只重算边界与 value head。优势停止梯度，critic 固定目标为 `Y=V_old+未白化GAE`，调用原生范围为 0.5 的 value clipping。三种策略的 actor 调用原生 vanilla 策略损失，包含 dual-clip 系数 3 与 log 概率比 `[-20,20]` 截断。原有 policy_weight、value_loss_weight 保持不变。

PPO actor／critic 分别使用 optimizer 和梯度裁剪，先完成 critic 的全部内轮，再完成 actor 的全部内轮；每次实际更新推进相应 scheduler。两种奖励模式均对每条轨迹的自主动作平均后再平均轨迹；critic 对真实事件、轨迹、文档依次平均。零自主动作路径保留在平均分母。完整 rollout 更新批次完成后记录、验证和保存；恢复沿用真实更新计数与 rollout 预热进度。

## 原生接入边界

`BoundaryTrainer` 继承 verl 0.8.0 的 `main_ppo_sync.PPOTrainer`，直接运行未经复制或修改的原生 `step`。同步正文 rollout 与缓存概率／价值通过任务 hooks 接入，原生方法负责 critic → actor 顺序和 `critic_warmup` 门控。`TrainingEngine` 继承 `BaseEngine`，RL 更新调用原生 `train_batch` 的清空梯度 → 前后向 → optimizer 更新生命周期；AdamW builder 和余弦 scheduler helper 同样复用原生实现。

任务微批、分段 autograd、FP32 复制式梯度 all-reduce、外层预算／dev／记录和原子单文件 checkpoint 仍由项目负责。当前不使用 TrainingWorker、FSDP、Ray worker 或 TransferQueue 服务。`fastapi` 和 `cachetools` 为原生 trainer 导入链所需依赖，不增加服务进程。

checkpoint 顶层字段保持 `config/model/optimizer/scheduler/cursor/rng/boundary_rng/binding`。`optimizer` 与 `scheduler` 分别保存 actor、critic；非 PPO 的 critic 为 null。`cursor` 保存主步数 step、rollout_step 和 samples，恢复同时核对两侧 scheduler 的实际进度，保持预热阶段和采样流。

PPO 保留原 `approx_kl=mean(q−1−log q)`，另记原生 `ppo_kl=−mean(clip(log q,−20,20))`；原生策略和价值损失提供 clip_fraction、dual_clip_fraction 和 value_clip_fraction。actor_optimizer_step、critic_optimizer_step 与 rollout_step 分别记录实际执行进度，仅保存在本地辅助指标中。actor_loss 与 critic_loss 记录各角色加权后的损失，loss 为两者之和；预热批仅包含 critic_loss。

## 验证与交付

本轮实测结果：

| 检查 | 结果 |
|---|---|
| 配置契约 | 9 份预设往返解析通过；critic_warmup 类型、范围和策略限制校验通过 |
| 现有测试 | `tests/v3_1_1` 共 31 项通过；参考结果按已采用的 GRPO 样本标准差和 dual-clip 修订，未新增用例 |
| 算法数值 | 原生优势／损失调用及任务适配通过临时数值检查，奖励成本与两种 mask 正确 |
| CPU 小型 Qwen3 | 六种策略／奖励组合通过；覆盖 critic_warmup=0／1／2／11、critic 全部内轮后 actor、Writer 冻结及固定旧价值／优势／目标 |
| 边界情况 | group_size=1、单 EOF 事件、无自主动作批通过；无动作批的 actor 参数、动量和 scheduler 均保持不变 |
| 单 rank 恢复 | 预热中与预热结束后恢复，下一批结果与连续执行逐位一致 |
| 双 rank CPU Gloo | 正文长度和 QA 数不同、空 rank 尾批、critic_warmup=0／2 均通过；各 rank 参数一致 |
| 双 rank 恢复 | checkpoint 后下一批参数、指标、两个 optimizer 和采样 RNG 与连续执行逐位一致 |
| 静态检查 | Ruff、Python 编译与 `git diff --check` 通过 |

新增验证均使用 `/tmp` 下的临时脚本，没有新增持久化测试。未启动 GPU 或服务器训练；CPU 数值和分布式检查不替代 GPU 运行验证。

## 既有产物的手动对齐

内部产物按当前唯一格式读取。使用本次修改之前的 checkpoint 时，先手动对齐其配置：`retention_delta` 改为 `stepwise`，移除 `advantage_mode`，SCST 的 `num_samples` 改为 `group_size`，策略目标补充 `gamma`；GRPO 两种奖励均需 `advantage_epsilon`。配置中的学习率、`backward`、`dev.max_new_tokens` 和 `critic_warmup` 字段须符合上述更新范围与目标限制。模型状态补充 `value_head=null`，包括仅用于初始化压缩器的预训练 checkpoint。用于完整恢复的 optimizer／scheduler 必须分别整理为 `actor`、`critic` 两个状态，非 PPO 的 critic 为 null；cursor 补充真实 rollout_step。旧 PPO 联合优化器需按参数身份拆分对应状态并核对双方步数，不能只增加一个空字段。代码不自动转换旧格式，既有运行结果也不会因配置改名而成为新算法的实验结果。
