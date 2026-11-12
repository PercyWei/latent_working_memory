"""采样与回放共用的在线缓冲区；提交后旧记忆块保持不变。"""

import torch


class SourceStream:
    def __init__(self, model, first_unit_ids):
        if not first_unit_ids:
            raise ValueError("a stream must start with a nonempty first unit")
        self.model = model
        self.blocks = []
        self.pending_ids = tuple(first_unit_ids)
        self._cached = not torch.is_grad_enabled()
        self._cache = model.prefill_source(self.blocks, self.pending_ids) if self._cached else None
        self._observed_unit = None
        self._pending_prefix_length = 0

    def _check_cache_mode(self):
        if self._cached and torch.is_grad_enabled():
            raise RuntimeError("a cached source stream cannot be reused for gradient replay")

    def logit(self, new_unit_ids):
        self._check_cache_mode()
        if self._cached:
            if self._observed_unit is not None:
                raise RuntimeError("advance the previous observed unit before observing another")
            self._pending_prefix_length = self._cache.get_seq_length()
            logit, self._cache = self.model.observe_source(self._cache, new_unit_ids)
            self._observed_unit = tuple(new_unit_ids)
            return logit
        return self.model.boundary_logit(self.blocks, self.pending_ids, new_unit_ids)

    def advance(self, new_unit_ids, commit):
        self._check_cache_mode()
        if self._cached and self._observed_unit != tuple(new_unit_ids):
            raise RuntimeError("advance must use the unit from the preceding boundary decision")
        if commit:
            if self._cached:
                history_length = sum(map(len, self.blocks))
                # 新单元仅用于判断，提交的是此前的 pending；不能将它混入压缩。
                self._cache.crop(self._pending_prefix_length)
                self.blocks.append(self.model.compress_source(self._cache))
                # 历史 memory 的内容和位置不变；丢弃原文及压缩查询的 KV。
                self._cache.crop(history_length)
            else:
                self.blocks.append(self.model.compress(self.pending_ids, self.blocks))
            self.pending_ids = tuple(new_unit_ids)
            if self._cached:
                # 仅追加新 memory 和新单元，沿用同一缓存，避免整套新旧 KV 并存。
                self._cache = self.model.prefill_source(self.blocks[-1:], self.pending_ids, self._cache)
        else:
            self.pending_ids += tuple(new_unit_ids)
        self._observed_unit = None

    def finish(self):
        self._check_cache_mode()
        if self._observed_unit is not None:
            raise RuntimeError("advance the observed unit before finishing the stream")
        if self.pending_ids:
            self.blocks.append(
                self.model.compress_source(self._cache)
                if self._cached
                else self.model.compress(self.pending_ids, self.blocks)
            )
            self.pending_ids = ()
            self._cache = None
        return tuple(self.blocks)
