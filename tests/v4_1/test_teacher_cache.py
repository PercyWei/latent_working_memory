"""响应缓存保留固定池语义；融合尺度与独立回放统计一致。"""

import torch
from transformers import GenerationConfig
from transformers.cache_utils import DynamicCache

from latent_working_memory.v4_1.modeling.am import teacher_scales
from latent_working_memory.v4_1.teacher.cache import build_cache_identity
from latent_working_memory.v4_1.teacher.replay import TeacherReplay


def test_compact_cache_scales_equal_full_pool_statistics(tiny_setup):
    setup = tiny_setup
    record, cached = setup.cache.read(setup.episode.document_id, 1)
    assert set(cached) == {
        "response_token_ids", "response_offsets", "positions", "scales_o", "scales_z",
    }
    for name, value in cached.items():
        assert value.dtype == (torch.float32 if name.startswith("scales_") else torch.int64)
    assert (record["processed_end"], record["compressed_end"]) == (12, 8)
    offsets = cached["response_offsets"].tolist()
    tokens = cached["response_token_ids"].tolist()
    prompts = setup.cache.identity["prompt_token_ids"]
    assert offsets[0] == 0 and offsets[-1] == len(tokens)
    sequences = [
        (*prompt, *tokens[offsets[index]:offsets[index + 1]])
        for index, prompt in enumerate(prompts)
    ]
    assert record["candidate_count"] == sum(map(len, sequences))
    replay = TeacherReplay(setup.teacher, 5, 2)
    prefix = DynamicCache(config=setup.teacher.config)
    replay.advance(setup.episode.context_ids, prefix, record["processed_end"])
    base = setup.teacher.config
    shape = (base.num_hidden_layers, setup.config.teacher.pool_size,
             base.num_attention_heads, base.head_dim)
    outputs, log_mass = torch.empty(shape), torch.empty(shape[:-1])

    def collect(layer, indices, queries, values, mass):
        outputs[layer, indices] = values
        log_mass[layer, indices] = mass

    replay.replay(prefix, sequences, cached["positions"], record["compressed_end"], collect)
    for layer in range(base.num_hidden_layers):
        expected_o, expected_z = teacher_scales(
            outputs[layer], log_mass[layer], setup.config.model.am_std_floor
        )
        torch.testing.assert_close(cached["scales_o"][layer], expected_o, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(cached["scales_z"][layer], expected_z, rtol=1e-6, atol=1e-7)
    # 生成停止条件属于缓存身份；tokenizer 的单个 EOS 不能代替模型的 EOS 集合。
    generation = GenerationConfig.from_pretrained(setup.config.model.model_name_or_path)
    generation.eos_token_id = [2, 1, 2]
    generation.save_pretrained(setup.config.model.model_name_or_path)
    changed = build_cache_identity(setup.config, setup.tokenizer)
    assert changed["eos_token_ids"] == [1, 2]
    assert changed != setup.cache.identity
