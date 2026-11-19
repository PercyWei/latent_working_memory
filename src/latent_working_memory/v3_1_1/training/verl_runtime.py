"""将正文压缩轨迹接入 verl 原生 PPO 单步调度。

模型与数据由同进程的任务 Engine 管理；不启动文本生成服务或 TransferQueue。
PPOTrainer.step 原样负责 critic 更新、critic_warmup 判断及 actor 更新顺序。
"""

from dataclasses import dataclass, field

from omegaconf import OmegaConf
from verl.checkpoint_engine import CheckpointEngineManager
from verl.trainer.main_ppo_sync import PPOTrainer
from verl.workers.config.rollout import CheckpointEngineConfig

from ..data import Document


@dataclass
class BoundaryBatch:
    documents: list[Document]
    groups: list
    extra_info: dict = field(default_factory=dict)


class _DocumentRollout:
    """同步执行本 rank 的真实采样，并将完整批次交给原生 trainer。"""

    # QA 评分在采样进程内执行，无独立 reward-model worker。
    reward_loop_worker_handles = None

    def __init__(self, trainer):
        self.trainer = trainer
        self.batch = None

    def generate_sequences(self, request):
        documents = list(request["raw_prompt"][0])
        self.batch = self.trainer.engine._collect_policy_batch(
            documents, self.trainer.global_steps,
            self.trainer.config.actor_rollout_ref.rollout.temperature,
        )

    def sample(self, partition_id, global_steps):
        # 原生 step 生成与消费之间没有参数更新；消费后不保留上一批轨迹。
        batch, self.batch = self.batch, None
        return batch


class BoundaryTrainer(PPOTrainer):
    """保留原生 step；任务 hooks 使用已采样的事件而非词表 token。"""

    def __init__(self, engine):
        self.engine = engine
        objective = engine.config.objective
        self.config = OmegaConf.create({
            "algorithm": {
                # SCST 的 sampled/greedy 共享正文批次，由任务 rollout 一起采样。
                "adv_estimator": "gae" if objective.strategy == "ppo" else objective.strategy,
            },
            "actor_rollout_ref": {"rollout": {
                "n": objective.group_size,
                "temperature": 1.0,
            }},
            "trainer": {"critic_warmup": engine.config.loop.critic_warmup},
        })
        self.use_critic = objective.strategy == "ppo"
        self.use_reference_policy = False
        self.global_steps = 0
        rollout = _DocumentRollout(self)
        self.async_rollout_manager = rollout
        self.replay_buffer = rollout
        self.reward_loop_manager = rollout
        # Writer 与策略在同进程共享参数，没有需要休眠或传权重的远端副本。
        # 此 manager 仅用于 rollout 权重同步，与训练 checkpoint 保存无关。
        self.checkpoint_manager = CheckpointEngineManager(
            CheckpointEngineConfig(), trainer=None, replicas=[],
        )

    def run(self, documents, rollout_step, temperature):
        self.global_steps = rollout_step
        self.config.actor_rollout_ref.rollout.temperature = temperature
        metrics, timing = {}, {}
        # 一个请求携带本 rank 的完整正文批次，包括空 rank；未增加样本或奖励事件。
        # 原生 uid 标识这次请求，算法分组继续使用真实 Document.id。
        self.step({"raw_prompt": [tuple(documents)]}, metrics, timing)
        return metrics

    def _compute_reward_colocate(self, batch):
        """rollout 已携带 QA NLL 和逐事件占用量，优势入口统一结算奖励。"""
        return batch

    def _balance_batch(self, batch, metrics):
        """保持现有文档分配及完整采样组，不以动作数重排或补齐真实样本。"""
        return batch

    def _compute_old_log_prob(self, batch, metrics):
        """直接使用采样动作时记录的概率，保持原温度和旧策略身份。"""
        return batch

    def _compute_values(self, batch, metrics):
        """采样批次已缓存更新前的 value head 预测，不重复运行 Writer。"""
        return batch

    def _compute_advantage(self, batch, metrics):
        self.engine._prepare_policy_batch(batch)
        return batch

    def _update_critic(self, batch, metrics):
        self.engine._update_policy_critic(batch, metrics)
        return batch

    def _update_actor(self, batch, metrics):
        self.engine._update_policy_actor(batch, metrics)
        return batch
