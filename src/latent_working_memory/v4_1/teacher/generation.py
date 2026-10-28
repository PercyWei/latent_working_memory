"""共享原文 prefix KV，在批量生成响应的同一次前向中收集尺度统计。"""

import torch
from transformers.cache_utils import DynamicCache

from .statistics import GenerationStatistics


@torch.no_grad()
def generate_reference_pool(model, prefix, prompts, eos_ids, max_new_tokens, batch_size,
                            compressed_end, pool_size, rng, floor, on_batch=None):
    device = next(model.parameters()).device
    prefix_length = prefix[0][0].shape[-2]
    responses = [[] for _ in prompts]
    statistics = GenerationStatistics(model, prefix, compressed_end, pool_size, rng)
    with statistics:
        # 保持原提示顺序，批次大小不改变 reservoir 遍历顺序。
        for start in range(0, len(prompts), batch_size):
            batch_indices = list(range(start, min(start + batch_size, len(prompts))))
            active = batch_indices.copy()
            statistics.start_batch(
                batch_indices, [len(prompts[index]) + max_new_tokens for index in batch_indices],
            )
            lengths = torch.tensor([len(prompts[index]) for index in active], device=device)
            width = max(len(prompts[index]) for index in active)
            ids = torch.zeros((len(active), width), dtype=torch.long)
            rows, columns, prompt_indices, sequence_positions = [], [], [], []
            for row, index in enumerate(active):
                length = len(prompts[index])
                ids[row, -length:] = torch.tensor(prompts[index])
                rows.extend([row] * length)
                columns.extend(range(width - length, width))
                prompt_indices.extend([index] * length)
                sequence_positions.extend(range(length))
            statistics.select_positions(rows, columns, prompt_indices, sequence_positions)
            # 在原文 prefix 之后左填充，全部真实提示均结束在最后一列。
            columns_tensor = torch.arange(width, device=device)[None]
            prompt_mask = columns_tensor >= width - lengths[:, None]
            attention_mask = torch.cat((
                torch.ones((len(active), prefix_length), dtype=torch.bool, device=device),
                prompt_mask,
            ), dim=1)
            position_ids = prefix_length + (columns_tensor - width + lengths[:, None]).clamp_min(0)
            branch = DynamicCache(ddp_cache_data=tuple(
                (keys.expand(len(active), -1, -1, -1), values.expand(len(active), -1, -1, -1))
                for keys, values in prefix
            ))
            logits = model(
                input_ids=ids.to(device), past_key_values=branch, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=True, logits_to_keep=1,
            ).logits[:, -1]
            for step in range(max_new_tokens):
                tokens = logits.argmax(dim=-1)
                live_rows = []
                for row, token in enumerate(tokens.tolist()):
                    responses[active[row]].append(token)
                    if token not in eos_ids:
                        live_rows.append(row)
                attention_mask = torch.cat((
                    attention_mask, attention_mask.new_ones((len(active), 1)),
                ), dim=1)
                position_ids = (prefix_length + lengths + step)[:, None]
                statistics.select_positions(
                    list(range(len(active))), [0] * len(active), active,
                    [len(prompts[index]) + step for index in active],
                )
                # EOS 或长度触顶的最后 token 也属于候选池，必须前向取得其 query。
                if not live_rows or step + 1 == max_new_tokens:
                    model.model(
                        input_ids=tokens[:, None], past_key_values=branch,
                        attention_mask=attention_mask, position_ids=position_ids, use_cache=True,
                    )
                    break
                logits = model(
                    input_ids=tokens[:, None], past_key_values=branch,
                    attention_mask=attention_mask, position_ids=position_ids,
                    use_cache=True, logits_to_keep=1,
                ).logits[:, -1]
                # 结束行完成最后 token 的统计后退出，不生成新的响应 token。
                if len(live_rows) != len(active):
                    rows_tensor = torch.tensor(live_rows, dtype=torch.long, device=device)
                    branch.batch_select_indices(rows_tensor)
                    logits = logits.index_select(0, rows_tensor)
                    attention_mask = attention_mask.index_select(0, rows_tensor)
                    lengths = lengths.index_select(0, rows_tensor)
                    active = [active[row] for row in live_rows]
            # 先释放本批分支 KV，再把入池位置的紧凑统计搬到 CPU。
            del logits, branch
            statistics.finish_batch([
                len(prompts[index]) + len(responses[index]) for index in batch_indices
            ])
            if on_batch is not None:
                on_batch(start + len(batch_indices))
    positions, candidate_count, scales = statistics.finish(floor)
    lengths = torch.tensor([len(response) for response in responses], dtype=torch.int64)
    tensors = {
        "response_token_ids": torch.tensor(
            [token for response in responses for token in response], dtype=torch.int64,
        ),
        "response_offsets": torch.cat((lengths.new_zeros(1), lengths.cumsum(0))),
        "positions": positions,
        **scales,
    }
    return tensors, candidate_count
