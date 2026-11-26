"""从各正文选取独立的 E²-TTT 递归状态与局部 attention 状态。"""

def fork_cache(sources):
    prefix, _ = sources[0]
    lengths = [cache[0]["padding_state"]["seen"][row] for cache, row in sources]
    branch = type(prefix)(seen_tokens=max(lengths))
    for index in range(len(prefix)):
        state = branch.update(layer_idx=index, offset=0)
        # padding_forward 和 TTT 内核用新张量替换各行状态，分支可共享初始张量。
        state["padding_state"] = {
            "rows": [cache[index]["padding_state"]["rows"][row].copy() for cache, row in sources],
            "seen": tuple(lengths),
        }
    return branch


def state_tensors(cache):
    for index in range(len(cache)):
        for row in cache[index]["padding_state"]["rows"]:
            yield from row["attn_state"]
            yield from row["recurrent_state"]
