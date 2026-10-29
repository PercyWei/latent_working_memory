"""生成时采集紧凑统计；按原提示顺序维护固定候选池，不保存逐位置向量。"""

import torch


class GenerationStatistics:
    """直接消费 attention 的局部输出，CPU 保留候选池，GPU 仅保留本批三标量。"""

    def __init__(self, model, pool_size, rng):
        self.pool_size = pool_size
        self.rng = rng
        self.device = next(model.parameters()).device
        self.layers = model.config.num_hidden_layers
        self.heads = model.config.num_attention_heads
        self.pool = torch.empty(self.layers, pool_size, self.heads, 3, dtype=torch.float64)
        self.positions = []
        self.count = 0
        self.batch = None
        self.destinations = None

    def start_batch(self, prompt_indices, capacities):
        self.prompt_indices = tuple(prompt_indices)
        self.offsets = {}
        size = 0
        for index, capacity in zip(prompt_indices, capacities):
            self.offsets[index] = size
            size += capacity
        self.batch = torch.empty(
            self.layers, size, self.heads, 3, device=self.device, dtype=torch.float64,
        )

    def select_positions(self, prompt_indices, sequence_positions):
        destinations = [self.offsets[index] + position
                        for index, position in zip(prompt_indices, sequence_positions)]
        self.destinations = torch.tensor(destinations, device=self.device, dtype=torch.long)

    def capture(self, layer_index, outputs, log_mass):
        variance, mean = torch.var_mean(outputs.double(), dim=-1, correction=0)
        moments = torch.stack((mean, variance, log_mass.double()), dim=-1)
        self.batch[layer_index].index_copy_(0, self.destinations, moments)

    def finish_batch(self, lengths):
        # 按原提示／位置顺序采样；一批内同槽被多次替换时只拷贝最终值。
        replacements = {}
        for prompt_index, length in zip(self.prompt_indices, lengths):
            for position in range(length):
                self.count += 1
                if len(self.positions) < self.pool_size:
                    slot = len(self.positions)
                    self.positions.append((prompt_index, position))
                else:
                    slot = self.rng.randrange(self.count)
                    if slot >= self.pool_size:
                        continue
                    self.positions[slot] = (prompt_index, position)
                replacements[slot] = self.offsets[prompt_index] + position
        if replacements:
            slots = torch.tensor(list(replacements), dtype=torch.long)
            sources = torch.tensor(list(replacements.values()), dtype=torch.long, device=self.device)
            selected = self.batch.index_select(1, sources).cpu()
            self.pool.index_copy_(1, slots, selected)
        self.batch = self.destinations = None

    def finish(self, floor):
        if self.count < self.pool_size:
            raise ValueError(
                f"teacher generated {self.count} candidate positions, below required "
                f"pool_size={self.pool_size}; increase prompt/response coverage rather than duplicating positions"
            )
        # 全方差公式：跨位置/通道方差 = 通道内方差均值 + 通道均值的跨位置方差。
        output_variance = self.pool[..., 1].mean(dim=1) + self.pool[..., 0].var(dim=1, correction=0)
        mass_variance = self.pool[..., 2].var(dim=1, correction=0)
        scales = {
            "scales_o": output_variance.sqrt().float().clamp_min(floor),
            "scales_z": mass_variance.sqrt().float().clamp_min(floor),
        }
        return torch.tensor(self.positions, dtype=torch.int64), self.count, scales
