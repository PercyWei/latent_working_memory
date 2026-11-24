"""冻结 Reader 的等长问题批量生成；传入的 cache 已由方法适配器分支。"""

from copy import copy
import torch


def fork_dynamic_cache(prefix, batch_size):
    cache = copy(prefix)
    cache.layers = [copy(layer) for layer in prefix.layers]
    for layer in cache.layers:
        layer.keys = layer.keys.expand(batch_size, -1, -1, -1)
        layer.values = layer.values.expand(batch_size, -1, -1, -1)
    return cache


@torch.no_grad()
def greedy_read(model, cache, embeddings, budgets, eos_id, complete, vocab_size=None):
    """问题等长且位置一致；结束的行停止记分，整批完成后释放分支 cache。"""
    tokens = [[] for _ in budgets]
    active = set(range(len(budgets)))
    for _ in range(max(budgets)):
        output = model.model(inputs_embeds=embeddings, past_key_values=cache,
                             use_cache=True, return_dict=True)
        hidden = output.last_hidden_state[:, -1]
        cache = output.past_key_values
        del output
        logits = model.get_output_embeddings()(hidden)
        if vocab_size is not None:
            logits = logits[:, :vocab_size]
        predicted = logits.argmax(dim=-1)
        for index in sorted(active):
            token = predicted[index].item()
            tokens[index].append(token)
            reason = ("eos" if token == eos_id else
                      "max_new_tokens" if len(tokens[index]) == budgets[index] else None)
            if reason:
                complete(index, {"generated_token_ids": tokens[index], "stop_reason": reason})
                active.remove(index)
        if not active:
            break
        embeddings = model.get_input_embeddings()(predicted[:, None])
