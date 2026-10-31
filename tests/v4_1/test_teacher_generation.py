"""融合生成对照完整原文响应与独立回放统计，包含 EOS、末位置和批量变化。"""

import random

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from latent_working_memory.v4_1.modeling.am import teacher_scales
from latent_working_memory.v4_1.teacher.attention import SharedPrefixAttention
from latent_working_memory.v4_1.teacher.generation import generate_reference_pool
from latent_working_memory.v4_1.teacher.replay import TeacherReplay


@torch.no_grad()
def test_batched_generation_matches_full_context(tiny_setup):
    model = tiny_setup.teacher
    context = tiny_setup.episode.context_ids[:12]
    prompts = ((6, 7, 8, 9, 10, 11), (12, 13), (15, 16, 17), (18,))
    budget, compressed_end, seed, floor = 6, 8, 101, 1e-6
    cache = model(input_ids=torch.tensor([context]), use_cache=True).past_key_values
    prefix = tuple((layer.keys, layer.values) for layer in cache.layers)
    saved_prefix = [(keys.clone(), values.clone()) for keys, values in prefix]
    original_attention = [layer.self_attn for layer in model.model.layers]

    def reference(eos_ids):
        responses = []
        for prompt in prompts:
            response = []
            for _ in range(budget):
                output = model(input_ids=torch.tensor([(*context, *prompt, *response)]), use_cache=False)
                token = int(output.logits[0, -1].argmax())
                response.append(token)
                if token in eos_ids:
                    break
            responses.append(tuple(response))
        return responses

    unlimited = reference(set())
    eos = unlimited[0][0]
    expected_stops = reference({eos})
    assert min(map(len, expected_stops)) == 1
    assert max(map(len, expected_stops)) > 1
    for eos_ids in (set(), {eos}):
        expected = reference(eos_ids)
        sequences = [(*prompt, *response) for prompt, response in zip(prompts, expected)]
        all_positions = [(index, position) for index, sequence in enumerate(sequences)
                         for position in range(len(sequence))]
        # 全池覆盖最后生成的 token；小池覆盖 reservoir 多次替换。
        for pool_size in (5, len(all_positions)):
            positions, rng = [], random.Random(seed)
            for count, position in enumerate(all_positions, 1):
                if len(positions) < pool_size:
                    positions.append(position)
                else:
                    replacement = rng.randrange(count)
                    if replacement < pool_size:
                        positions[replacement] = position
            positions = torch.tensor(positions)
            base = model.config
            shape = (base.num_hidden_layers, pool_size, base.num_attention_heads, base.head_dim)
            outputs, masses = torch.empty(shape), torch.empty(shape[:-1])

            def collect(layer, indices, queries, values, mass):
                outputs[layer, indices] = values
                masses[layer, indices] = mass

            TeacherReplay(model, 5, 2).replay(cache, sequences, positions, compressed_end, collect)
            scales = [teacher_scales(outputs[layer], masses[layer], floor)
                      for layer in range(base.num_hidden_layers)]
            for batch_size in (1, 3, 4):
                calls = []
                suffix_lengths = []

                def capture(module, args, kwargs):
                    calls.append(tuple(kwargs['input_ids'].shape))
                    suffix_lengths.append(kwargs['past_key_values'].get_seq_length())

                handle = model.model.register_forward_pre_hook(capture, with_kwargs=True)
                try:
                    tensors, count = generate_reference_pool(
                        model, prefix, prompts, eos_ids, budget, batch_size,
                        compressed_end, pool_size, random.Random(seed), floor,
                    )
                finally:
                    handle.remove()
                offsets = tensors['response_offsets'].tolist()
                ids = tensors['response_token_ids'].tolist()
                assert [tuple(ids[left:right]) for left, right in zip(offsets, offsets[1:])] == expected
                assert count == len(all_positions)
                torch.testing.assert_close(tensors['positions'], positions, rtol=0, atol=0)
                for layer, (expected_o, expected_z) in enumerate(scales):
                    torch.testing.assert_close(tensors['scales_o'][layer], expected_o, rtol=1e-5, atol=1e-7)
                    torch.testing.assert_close(tensors['scales_z'][layer], expected_z, rtol=1e-5, atol=1e-7)
                # 每批仅一次提示 prefill，之后各响应 token（包括最后一个）仅前向一次。
                expected_calls = []
                expected_suffix_lengths = []
                for start in range(0, len(prompts), batch_size):
                    batch = list(range(start, min(start + batch_size, len(prompts))))
                    width = max(len(prompts[index]) for index in batch)
                    expected_calls.append((len(batch), width))
                    expected_suffix_lengths.append(0)
                    for step in range(max(len(expected[index]) for index in batch)):
                        expected_calls.append((sum(len(expected[index]) > step for index in batch), 1))
                        expected_suffix_lengths.append(width + step)
                assert calls == expected_calls
                assert suffix_lengths == expected_suffix_lengths
                assert [layer.self_attn for layer in model.model.layers] == original_attention
                for (keys, values), (saved_keys, saved_values) in zip(prefix, saved_prefix):
                    torch.testing.assert_close(keys, saved_keys, rtol=0, atol=0)
                    torch.testing.assert_close(values, saved_values, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="interrupted"):
        with SharedPrefixAttention(model, prefix, compressed_end, lambda *args: None):
            raise RuntimeError("interrupted")
    assert [layer.self_attn for layer in model.model.layers] == original_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="实际 FlashInfer 对照需要 CUDA")
@pytest.mark.parametrize("compressed_end", [7, 11])
@torch.no_grad()
def test_flashinfer_shared_attention_matches_sdpa(compressed_end):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 对照需要 SM80+")
    torch.manual_seed(42)
    device, dtype = torch.device("cuda"), torch.bfloat16
    config = Qwen3Config(
        vocab_size=32, hidden_size=256, intermediate_size=512, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
    )
    model = Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()
    prefix = ((torch.randn(1, 2, 11, 64, device=device, dtype=dtype),
               torch.randn(1, 2, 11, 64, device=device, dtype=dtype)),)
    captured = []
    attention = SharedPrefixAttention(
        model, prefix, compressed_end,
        lambda layer, query, output, mass: captured.append((output, mass)),
    )
    # 左填充的提示 prefill，以及已有不同长度 suffix 后的单 token decode。
    for query_lengths, suffix_lengths in (([3, 1], [3, 1]), ([1, 1], [4, 2])):
        width, suffix_width = max(query_lengths), max(suffix_lengths)
        queries = torch.randn(2, 4, width, 64, device=device, dtype=dtype)
        keys = torch.randn(2, 2, suffix_width, 64, device=device, dtype=dtype)
        values = torch.randn_like(keys)
        attention.plan(query_lengths, width, suffix_lengths, suffix_width)
        actual = attention.read(0, queries, keys, values, 64 ** -0.5)
        expected = torch.zeros_like(actual)
        local_outputs, local_masses = [], []
        for row, (q_len, kv_len) in enumerate(zip(query_lengths, suffix_lengths)):
            query = queries[row, :, -q_len:]
            key = torch.cat((prefix[0][0][0], keys[row, :, -kv_len:]), dim=1)
            value = torch.cat((prefix[0][1][0], values[row, :, -kv_len:]), dim=1)
            key, value = (tensor.repeat_interleave(2, dim=0) for tensor in (key, value))
            visible = torch.arange(11 + kv_len, device=device)[None] <= (
                11 + kv_len - q_len + torch.arange(q_len, device=device)[:, None]
            )
            expected[row, -q_len:] = F.scaled_dot_product_attention(
                query, key, value, attn_mask=visible,
            ).transpose(0, 1)
            scores = query.float() @ key[:, :compressed_end].float().transpose(-1, -2) / 8
            local_outputs.append((scores.softmax(-1) @ value[:, :compressed_end].float()).transpose(0, 1))
            local_masses.append(scores.logsumexp(-1).transpose(0, 1))
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
        local_output, local_mass = captured[-1]
        torch.testing.assert_close(local_output.float(), torch.cat(local_outputs), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(local_mass, torch.cat(local_masses), rtol=2e-3, atol=2e-3)
