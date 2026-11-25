"""TTT 方法共用的 LoCoMo 输入格式。"""

from latent_working_memory.prompts import qa_input_format


INPUT_FORMAT = {
    **qa_input_format(use_chat_template=False, answer_prefix="\nAnswer:"),
    "add_special_tokens": True,
}
