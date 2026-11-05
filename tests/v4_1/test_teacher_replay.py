"""批量截短回放必须等价于各分支独立的完整原文前向。"""

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import rotate_half

from latent_working_memory.v4_1.modeling.am import local_read
from latent_working_memory.v4_1.teacher.replay import TeacherReplay


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_replay_matches_native_full_context(tiny_setup, dtype):
    model = tiny_setup.teacher.to(dtype)
    context = tiny_setup.episode.context_ids[:12]
    sequences = ((6, 7, 8, 9, 10, 11), (12, 13, 14), (15, 16, 17, 18, 19))
    # 不同分支长度、非排序位置，以及最后一个响应 token 的 query。
    positions = torch.tensor([[2, 4], [0, 1], [1, 2], [0, 3], [2, 0], [1, 0]])
    compressed_end = 8
    shape = (model.config.num_hidden_layers, len(positions), model.config.num_attention_heads,
             model.config.head_dim)
    expected = {"queries": torch.empty(shape), "outputs": torch.empty(shape),
                "log_mass": torch.empty(shape[:-1])}
    queries, rotary = {}, {}

    def capture_rotary(module, args, kwargs):
        rotary["cos"], rotary["sin"] = kwargs["position_embeddings"]

    def capture_query(index):
        def capture(module, args, output):
            cosine, sine = rotary["cos"][:, :, None, :], rotary["sin"][:, :, None, :]
            queries[index] = (output * cosine + rotate_half(output) * sine)[0].float()
        return capture

    handles = []
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(capture_rotary, with_kwargs=True))
        handles.append(layer.self_attn.q_norm.register_forward_hook(capture_query(index)))
    try:
        with torch.no_grad():
            for branch, sequence in enumerate(sequences):
                output = model.model(input_ids=torch.tensor([(*context, *sequence)]), use_cache=True)
                pool_indices = (positions[:, 0] == branch).nonzero().flatten()
                columns = len(context) + positions[pool_indices, 1]
                for layer_index, layer_cache in enumerate(output.past_key_values.layers):
                    query = queries[layer_index][columns]
                    key = layer_cache.keys[0, :, :compressed_end]
                    value = layer_cache.values[0, :, :compressed_end]
                    local_output, mass = local_read(query, key, value, key.new_zeros(key.shape[:2]))
                    expected["queries"][layer_index, pool_indices] = query
                    expected["outputs"][layer_index, pool_indices] = local_output
                    expected["log_mass"][layer_index, pool_indices] = mass
    finally:
        for handle in handles:
            handle.remove()

    replay = TeacherReplay(model, prefill_chunk_size=5, batch_size=1)
    prefix = DynamicCache(config=model.config)
    replay.advance(context, prefix, len(context))
    original_prefix = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.layers]
    original_attention = [layer.self_attn for layer in model.model.layers]
    for batch_size in (1, 3):
        replay.batch_size = batch_size
        actual = {name: torch.empty_like(value) for name, value in expected.items()}
        branch_lengths = []

        def capture_branch(module, args, output):
            branch_lengths.append(output.past_key_values.get_seq_length())

        def collect(layer_index, pool_indices, query, outputs, log_mass):
            for name, value in (("queries", query), ("outputs", outputs), ("log_mass", log_mass)):
                actual[name][layer_index, pool_indices] = value

        handle = model.model.register_forward_hook(capture_branch)
        try:
            replay.replay(prefix, sequences, positions, compressed_end, collect)
        finally:
            handle.remove()
        assert max(branch_lengths) <= max(map(len, sequences))
        assert [layer.self_attn for layer in model.model.layers] == original_attention
        for name in actual:
            tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-6
            torch.testing.assert_close(actual[name], expected[name], atol=tolerance, rtol=tolerance)
            assert not actual[name].requires_grad and not actual[name].is_inference()
        assert prefix.get_seq_length() == len(context)
        for layer, (keys, values) in zip(prefix.layers, original_prefix):
            torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
            torch.testing.assert_close(layer.values, values, rtol=0, atol=0)


def test_distill_branches_preserve_device_and_native_log_probabilities(tiny_setup):
    model = tiny_setup.teacher
    context = tiny_setup.episode.context_ids[:12]
    branches = (((6, 7, 8, 9), 1), ((12, 13, 14), 0))
    replay = TeacherReplay(model, prefill_chunk_size=5, batch_size=2)
    prefix = DynamicCache(config=model.config)
    replay.advance(context, prefix, len(context))
    original_attention = [layer.self_attn for layer in model.model.layers]
    actual = replay.distill_branches(prefix, branches)
    assert [layer.self_attn for layer in model.model.layers] == original_attention
    assert prefix.get_seq_length() == len(context)
    for (ids, first, log_probs), (expected_ids, expected_first) in zip(actual, branches):
        assert ids == expected_ids and first == expected_first
        assert log_probs.device == replay.device and log_probs.dtype == torch.float32
        assert not log_probs.requires_grad and not log_probs.is_inference()
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([(*context, *ids)]), use_cache=False).logits
            expected = logits[0, len(context) + first:].float().log_softmax(-1)
        torch.testing.assert_close(log_probs, expected, rtol=2e-6, atol=2e-6)
