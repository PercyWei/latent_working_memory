"""完整上下文 LongMemEval 沿用公共正文预填，长历史按块控制激活显存。"""

from .locomo import load_adapter as load_document_adapter


PREFILL_CHUNK_SIZE = 4096


def load_adapter(settings):
    return load_document_adapter(settings, prefill_chunk_size=settings["prefill_chunk_size"])
