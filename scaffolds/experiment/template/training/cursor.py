"""各 rank 共用同一全局样本顺序，尾批不补齐。"""

import math

import torch


class SampleCursor:
    def __init__(self, size, seed):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.offset = 0
        self.epoch_end = False
        self.order = torch.randperm(size, generator=self.generator).tolist()

    def next_batch(self, batch_size):
        indices = self.order[self.offset : self.offset + batch_size]
        self.offset += len(indices)
        self.epoch_end = self.offset == self.size
        if self.epoch_end:
            self.epoch += 1
            self.offset = 0
            self.order = torch.randperm(self.size, generator=self.generator).tolist()
        return indices

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "offset": self.offset,
            "order": self.order,
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state, step, batch_size):
        epoch, offset, order = state["epoch"], state["offset"], state["order"]
        if (
            type(epoch) is not int
            or epoch < 0
            or type(offset) is not int
            or not 0 <= offset < self.size
            or offset % batch_size
            or type(step) is not int
            or step != epoch * math.ceil(self.size / batch_size) + math.ceil(offset / batch_size)
        ):
            raise ValueError("checkpoint cursor differs from its optimizer step and global batch")
        if (
            not isinstance(order, list)
            or any(type(index) is not int for index in order)
            or sorted(order) != list(range(self.size))
        ):
            raise ValueError(
                "checkpoint sample order must be a permutation of the training samples"
            )
        self.epoch, self.offset, self.order = epoch, offset, list(order)
        self.generator.set_state(state["generator"])
        self.epoch_end = epoch > 0 and offset == 0
