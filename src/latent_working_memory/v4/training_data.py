"""使用 verl collator 和 StatefulDataLoader 调度完整的全局 episode batch。"""

import random

import torch
from torch.utils.data import Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import SFTTensorCollator


class EpochSampler(Sampler):
    """各 rank 共享全局顺序，局部 microbatch 由 engine 分配。"""

    def __init__(self, episodes, seed, shuffle):
        self.episodes, self.seed, self.shuffle = episodes, seed, shuffle
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.episodes)))
        if self.shuffle:
            random.Random(f"{self.seed}:v4:{self.epoch}").shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.episodes)


_COLLATOR = SFTTensorCollator("no_padding")


def _collate_episodes(episodes):
    data = tu.get_tensordict(_COLLATOR([{"episodes": episode} for episode in episodes]))
    return tu.get(data, "episodes")


class TrainingDataLoader(StatefulDataLoader):
    def state_dict(self):
        state = super().state_dict()
        # 最后一批交付后立即保存时，原生 iterator 尚未标记 epoch 结束。
        if state["_num_yielded"] == len(self):
            state["_iterator_finished"] = True
        return state


def make_data_loader(episodes, batch_size, seed, shuffle=True):
    return TrainingDataLoader(
        episodes,
        batch_size=batch_size,
        sampler=EpochSampler(episodes, seed, shuffle),
        collate_fn=_collate_episodes,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
