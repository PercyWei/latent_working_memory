"""融合生成对照完整原文响应与独立回放统计，包含 EOS、末位置和批量变化。"""

import random

import torch

from latent_working_memory.v4_1.modeling.am import teacher_scales
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

                def capture(module, args, kwargs):
                    calls.append(tuple(kwargs['input_ids'].shape))

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
                for start in range(0, len(prompts), batch_size):
                    batch = list(range(start, min(start + batch_size, len(prompts))))
                    expected_calls.append((len(batch), max(len(prompts[index]) for index in batch)))
                    for step in range(max(len(expected[index]) for index in batch)):
                        expected_calls.append((sum(len(expected[index]) > step for index in batch), 1))
                assert calls == expected_calls
                for (keys, values), (saved_keys, saved_values) in zip(prefix, saved_prefix):
                    torch.testing.assert_close(keys, saved_keys, rtol=0, atol=0)
                    torch.testing.assert_close(values, saved_values, rtol=0, atol=0)
