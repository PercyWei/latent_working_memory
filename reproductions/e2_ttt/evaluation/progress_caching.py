"""观察原生生成与缓存事件，不改变请求、缓存操作或返回顺序。"""

from collections import Counter

from lm_eval.api.model import CacheHook, CachingLM, hash_args


class ProgressCacheHook(CacheHook):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner

    def add_partial(self, attr, req, res):
        # 原生每次写入仍先原样执行；HFLM 在每个生成 batch 后逐样本调用。
        super().add_partial(attr, req, res)
        if attr == "generate_until":
            self.owner.generated(req)


class ProgressCachingLM(CachingLM):
    def __init__(self, lm, cache_db, progress):
        self.progress = progress
        self.pending = Counter()
        super().__init__(lm, cache_db)

    def get_cache_hook(self):
        return ProgressCacheHook(self)

    @staticmethod
    def request_key(args):
        # 同一原生 kwargs 组可能复用首行的字典；只规范观察用的键顺序。
        context, kwargs = args
        return hash_args("generate_until", (context, dict(sorted(kwargs.items()))))

    def generated(self, args):
        key = self.request_key(args)
        if self.pending[key]:
            self.pending[key] -= 1
            self.progress.advance_generated(1)

    def generate_until(self, requests):
        self.pending.clear()
        hits = 0
        for request in requests:
            # 缓存命中仍按真实请求逐条计数，重复 prompt 不折成一个缓存键。
            if (not request.args[1].get("do_sample", False)
                    and hash_args("generate_until", request.args) in self.dbdict):
                hits += 1
            else:
                self.pending[self.request_key(request.args)] += 1
        self.progress.set_stage("generating")
        if hits:
            self.progress.advance_generated(hits)
        try:
            # 保留原生缓存检查、未命中请求、模型调用、二次写入和结果还原。
            result = super().__getattr__("generate_until")(requests)
            self.progress.set_stage("scoring")
            return result
        finally:
            self.pending.clear()
