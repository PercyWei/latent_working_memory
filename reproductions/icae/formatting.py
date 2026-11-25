"""ICAE Reader 的模板解析，供评估及指令微调的数据构造共用。"""

from latent_working_memory.prompts import QAInputBuilder, qa_input_format


def icae_input_format(tokenizer, stage, ft_token_id, system_prompt):
    if stage not in {"pretrained", "finetuned"}:
        raise ValueError(f"unsupported ICAE stage: {stage}")
    settings = qa_input_format(tokenizer)
    parts = QAInputBuilder(tokenizer, settings, system_prompt).parts()
    # ICAE 按模板、系统提示、记忆、问题分段编码。去除模板末尾的空格，
    # 避免单独编码空格改变官方 Mistral 输入；Qwen 角色标记后的换行保留。
    prefix_ids = tokenizer.encode(parts.template_prefix.rstrip(" "), add_special_tokens=False)
    prefix_ids += tokenizer.encode(parts.system_text, add_special_tokens=False)
    return {
        **settings,
        "reader_prefix_ids": prefix_ids,
        "question_prefix_ids": [ft_token_id] if stage == "finetuned" else [],
        "answer_prefix_ids": tokenizer.encode(parts.suffix, add_special_tokens=False),
    }
