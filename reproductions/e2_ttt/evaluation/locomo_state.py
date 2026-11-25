"""E²-TTT 的原生递归状态与局部 attention 状态按问题分支。"""

from copy import deepcopy


def fork_cache(prefix, batch_size):
    cache = deepcopy(prefix)
    for index in range(len(cache)):
        state = cache[index]["padding_state"]
        state["rows"] = [deepcopy(state["rows"][0]) for _ in range(batch_size)]
        state["seen"] = state["seen"] * batch_size
    return cache


def state_tensors(cache):
    for index in range(len(cache)):
        for row in cache[index]["padding_state"]["rows"]:
            yield from row["attn_state"]
            yield from row["recurrent_state"]
