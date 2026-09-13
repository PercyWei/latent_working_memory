"""Group original messages around complete evidence without tokenizing or writing data."""


def evidence_blocks(history, qas):
    messages = history["messages"]
    ranges = sorted((q["evidence_message_start"], q["evidence_message_end_exclusive"]) for q in qas)
    merged = []
    for start, end in ranges:
        if not 0 <= start < end <= len(messages):
            raise ValueError("evidence message range out of bounds")
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    starts = dict(merged)
    blocks = []
    i = 0
    while i < len(messages):
        end = starts.get(i, i + 1)
        subset = messages[i:end]
        context, offsets = "", {}
        for message in subset:
            if context:
                context += "\n"
            context += message["role"] + ": "
            offsets[message["message_id"]] = len(context)
            context += message["content"]
        block_qas = []
        for qa in qas:
            if i <= qa["evidence_message_start"] and qa["evidence_message_end_exclusive"] <= end:
                original = messages[
                    next(
                        j
                        for j in range(i, end)
                        if messages[j]["message_id"] == qa["answer_message_id"]
                    )
                ]
                a, b = qa["answer_char_start"], qa["answer_char_end_exclusive"]
                if original["content"][a:b] != qa["answer"]:
                    raise ValueError("answer does not reconstruct from history")
                for evidence in qa["evidence_messages"]:
                    actual = next(m for m in subset if m["message_id"] == evidence["message_id"])
                    if actual != evidence:
                        raise ValueError("saved evidence does not match original history")
                answer_start = offsets[qa["answer_message_id"]] + a
                if context[answer_start : answer_start + len(qa["answer"])] != qa["answer"]:
                    raise ValueError("serialized answer offsets disagree")
                block_qas.append(
                    dict(
                        id=qa["qa_id"],
                        question=qa["question"],
                        answers=[dict(text=qa["answer"], answer_start=answer_start)],
                        subject_type=qa["subject_type"],
                        temporal_scope=qa["temporal_scope"],
                    )
                )
        blocks.append(
            dict(message_start=i, message_end_exclusive=end, context=context, qas=block_qas)
        )
        i = end
    if sum(len(b["qas"]) for b in blocks) != len(qas):
        raise ValueError("not all QA were assigned to exactly one dialogue block")
    return blocks
