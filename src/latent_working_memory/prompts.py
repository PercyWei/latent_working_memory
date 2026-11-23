"""文档问答共用的 system 消息与聊天角色。"""

QA_SYSTEM_PROMPT = (
    "Answer questions using the provided document. Use exact details whenever possible. "
    "Answer with a short span, yes, no, or a concise phrase. "
    "If the document does not contain the answer, reply exactly: No information available."
)


def qa_messages(user_content):
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
