"""问答共用的系统提示、输入配置与模板包装。"""

from dataclasses import dataclass

QA_SYSTEM_PROMPT = (
    "Answer questions using the provided document. Use exact details whenever possible. "
    "Answer with a short span, yes, no, or a concise phrase. "
    "If the document does not contain the answer, reply exactly: No information available."
)


def qa_messages(user_content, system_prompt=QA_SYSTEM_PROMPT):
    messages = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
    return [*messages, {"role": "user", "content": user_content}]


def qa_input_format(tokenizer=None, use_chat_template=True, answer_prefix="Answer:", system_separator="\n\n"):
    """方法显式选择是否包装聊天模板；保存实际模板或文本分隔设置。"""
    settings = {"use_chat_template": use_chat_template}
    if use_chat_template:
        template = tokenizer.chat_template
        if not isinstance(template, str) or not template.strip():
            raise ValueError("QA input requires one explicit tokenizer chat template")
        settings.update(chat_template=template,
                        add_generation_prompt=True, enable_thinking=False)
    else:
        settings.update(system_separator=system_separator, answer_prefix=answer_prefix)
    return settings


@dataclass(frozen=True)
class QAPromptParts:
    template_prefix: str
    system_text: str
    suffix: str

    @property
    def prefix(self):
        return self.template_prefix + self.system_text


class QAInputBuilder:
    """包装方法提供的内容；记忆嵌入及其分词拼接边界由方法维护。"""

    def __init__(self, tokenizer, input_format, system_prompt=QA_SYSTEM_PROMPT):
        self.tokenizer = tokenizer
        self.input_format = dict(input_format)
        self.system_prompt = system_prompt

    def render(self, content, answer=None):
        settings = self.input_format
        if settings["use_chat_template"]:
            messages = qa_messages(content, self.system_prompt)
            if answer is not None:
                messages.append({"role": "assistant", "content": answer})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, chat_template=settings["chat_template"],
                add_generation_prompt=settings["add_generation_prompt"] if answer is None else False,
                enable_thinking=settings["enable_thinking"],
            )
        prefix = "" if self.system_prompt is None else self.system_prompt + settings["system_separator"]
        return prefix + content + settings["answer_prefix"] + ("" if answer is None else answer)

    def encode(self, content, add_special_tokens=True):
        # 聊天模板已包含特殊标记，整段分词与 apply_chat_template(tokenize=True) 一致。
        return self.tokenizer.encode(
            self.render(content),
            add_special_tokens=False if self.input_format["use_chat_template"] else add_special_tokens,
        )

    def parts(self):
        """返回内容前后的包装；方法可在中间插入 soft tokens。"""
        if not self.input_format["use_chat_template"]:
            system = "" if self.system_prompt is None else self.system_prompt + self.input_format["system_separator"]
            return QAPromptParts("", system, self.input_format["answer_prefix"])
        content_marker = "<|qa_content_placeholder|>"
        system_marker = None if self.system_prompt is None else "<|qa_system_placeholder|>"
        rendered = QAInputBuilder(self.tokenizer, self.input_format, system_marker).render(content_marker)
        if rendered.count(content_marker) != 1:
            raise ValueError("chat template must preserve one content placeholder")
        prefix, suffix = rendered.split(content_marker)
        if system_marker is None:
            return QAPromptParts(prefix, "", suffix)
        if rendered.count(system_marker) != 1 or system_marker not in prefix:
            raise ValueError("chat template must preserve one system message before the user content")
        template_prefix, separator = prefix.split(system_marker)
        return QAPromptParts(template_prefix, self.system_prompt + separator, suffix)

    def answer_suffix(self):
        """训练目标沿用模板的 assistant 结束符，并核对生成起始边界。"""
        content_marker = "__LWM_DOCUMENT_PLACEHOLDER__"
        answer_marker = "__LWM_ANSWER_PLACEHOLDER__"
        rendered = self.render(content_marker)
        answer_prefix, suffix = self.render(content_marker, answer_marker).split(answer_marker)
        if answer_prefix != rendered:
            raise ValueError("chat template must expose the same assistant generation prefix")
        return suffix
