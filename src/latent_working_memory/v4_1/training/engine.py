"""按完整样本轨迹累积元梯度，统一更新慢参数。"""

import math

import torch


class SampleCursor:
    """保存当前 epoch 的排列与读取位置；epoch 尾部允许不足一批。"""

    def __init__(self, size, seed):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.offset = 0
        self.order = torch.randperm(size, generator=self.generator).tolist()

    def next_batch(self, batch_size):
        if self.offset == self.size:
            self.epoch += 1
            self.offset = 0
            self.order = torch.randperm(self.size, generator=self.generator).tolist()
        indices = self.order[self.offset:self.offset + batch_size]
        self.offset += len(indices)
        return indices

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "offset": self.offset,
            "order": self.order,
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.offset = state["offset"]
        self.order = state["order"]
        self.generator.set_state(state["generator"])


class TrainingEngine:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        groups = {"decoder": [], "memory": [], "lora": []}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.dtype != torch.float32:
                raise ValueError(f"trainable parameter must use FP32: {name}")
            category = (
                "lora" if "lora_" in name
                else "decoder" if "decoder" in name and parameter.ndim >= 2
                else "memory"
            )
            groups[category].append(parameter)
        self.parameters = [parameter for group in groups.values() for parameter in group]
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": parameters,
                    "name": name,
                    "lr": config.lora_learning_rate if name == "lora" else config.learning_rate,
                    "weight_decay": config.weight_decay if name == "decoder" else 0.0,
                }
                for name, parameters in groups.items() if parameters
            ],
            betas=(config.beta1, config.beta2),
            eps=config.adam_epsilon,
        )
        # 第 k 次 optimizer 更新使用 f(k)，最后一次更新恰好到达 min_lr_ratio。
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda completed: self._lr_factor(completed + 1)
        )

    def _lr_factor(self, step):
        if self.config.warmup_steps and step <= self.config.warmup_steps:
            return step / self.config.warmup_steps
        span = max(1, self.config.max_steps - self.config.warmup_steps)
        progress = min(1.0, (step - self.config.warmup_steps) / span)
        cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
        return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

    def train_batch(self, episodes, teacher, generator):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {name: 0.0 for name in ("loss", "qa_loss", "outer_am", "inner_before", "inner_after")}
        target_tokens = write_events = 0
        learning_rates = {group["name"]: group["lr"] for group in self.optimizer.param_groups}
        for episode in episodes:
            with teacher.episode(episode) as trajectory:
                output = self.model(
                    episode,
                    teacher=trajectory,
                    lambda_am=self.config.lambda_am,
                    generator=generator,
                    differentiable=True,
                )
            # 反向只需要固定参考张量，不再占用完整原文的 teacher KV。
            (output["loss"] / len(episodes)).backward()
            for name in totals:
                totals[name] += float(output[name]) / len(episodes)
            target_tokens += output["target_tokens"]
            write_events += output["write_events"]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.config.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.scheduler.step()
        return {
            **totals,
            "grad_norm": float(grad_norm),
            "samples": len(episodes),
            "target_tokens": target_tokens,
            "write_events": write_events,
            **{f"lr_{name}": value for name, value in learning_rates.items()},
        }


def evaluate_dev(model, episodes, teacher, lambda_am, generator):
    """验证不保留元梯度；inner 的局部求导由模型内部显式启用。"""
    model.eval()
    totals = {name: 0.0 for name in ("loss", "qa_loss", "outer_am", "inner_before", "inner_after")}
    target_tokens = write_events = 0
    with torch.no_grad():
        for episode in episodes:
            with teacher.episode(episode) as trajectory:
                output = model(
                    episode, teacher=trajectory, lambda_am=lambda_am,
                    generator=generator, differentiable=False,
                )
            for name in totals:
                totals[name] += float(output[name]) / len(episodes)
            target_tokens += output["target_tokens"]
            write_events += output["write_events"]
    return {**totals, "samples": len(episodes), "target_tokens": target_tokens, "write_events": write_events}
