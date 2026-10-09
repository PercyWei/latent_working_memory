"""用 verl 的 collator 与 StatefulDataLoader 调度完整训练和验证样本。"""

import random

import torch
from torch.utils.data import Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import SFTTensorCollator


class EpochSampler(Sampler):
    """所有 rank 使用相同全局顺序，设备内的样本分配由训练引擎处理。"""

    def __init__(self, rows, seed, shuffle):
        self.rows, self.seed, self.epoch = rows, seed, 0
        self.shuffle = shuffle

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.rows)))
        if self.shuffle:
            random.Random(f"{self.seed}:v3:{self.epoch}").shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.rows)


_COLLATOR = SFTTensorCollator("no_padding")


def _collate_examples(rows):
    data = tu.get_tensordict(_COLLATOR([{"examples": row} for row in rows]))
    return tu.get(data, "examples")


class TrainingDataLoader(StatefulDataLoader):
    def state_dict(self):
        state = super().state_dict()
        # 最后一批已交付时，原生 iterator 要等下次 next 才标记结束。
        # 保存完整 epoch 的真实状态，让原生恢复直接创建下一 epoch 的 iterator。
        if state["_num_yielded"] == len(self):
            state["_iterator_finished"] = True
        return state


def make_data_loader(rows, batch_size, seed, shuffle=True):
    """返回完整 global batch；续训先恢复状态，再按游标设置 sampler.epoch。"""
    return TrainingDataLoader(
        rows,
        batch_size=batch_size,
        sampler=EpochSampler(rows, seed, shuffle),
        collate_fn=_collate_examples,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
