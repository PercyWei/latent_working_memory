# 固定提示来自 Latent Compaction 官方实现；保留原文和顺序。
# Source: https://github.com/THUNLP/latent-compaction/blob/81cfd4b3f30f820fb5f76451cb0f6385e0ac9510/latent_compaction/prompts.py
# MIT License
#
# Copyright (c) 2026 Latent Compaction contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Fixed, task-independent source prompts for document-level teacher rollouts."""

REQUESTS = (
    "Summarize the document's main subject and structure.",
    "Describe the writing style used in the document.",
    "What broad themes appear in the document?",
    "Give a concise overview of the document.",
    "Explain how the document is organized.",
    "Describe the kinds of statements found in the document.",
    "What is the general tone of the document?",
    "Paraphrase the document's introductory idea.",
    "What recurring language patterns appear in the document?",
    "Describe the document without quoting it directly.",
    "What would be a suitable title for this document?",
    "Give a high-level outline of the document.",
    "What kind of reader is this document written for?",
    "Explain the document's overall purpose.",
    "What stylistic features are most noticeable?",
    "Write one sentence characterizing the document.",
    "Identify the document's main topics.",
    "What factual details are emphasized in the document?",
    "Which parts of the document are most distinctive?",
    "Describe how ideas progress through the document.",
    "What concepts recur throughout the text?",
    "State the document's central claims.",
    "What examples does the document use?",
    "Describe the document's vocabulary and sentence patterns.",
)

INSTRUCTIONS = (
    "Answer briefly.",
    "Use one sentence.",
    "Be precise and concise.",
    "Focus only on information stated in the text.",
    "Do not introduce outside information.",
    "Use at most two sentences.",
    "Give a compact answer.",
    "Emphasize concrete textual evidence.",
    "Preserve important proper nouns.",
    "Avoid speculation.",
    "Answer in plain language.",
    "Prioritize the most important point.",
    "Mention one supporting detail.",
    "Use a short noun phrase when possible.",
    "Distinguish content from writing style.",
    "State only what can be inferred directly.",
    "Keep the answer self-contained.",
    "Use a neutral tone.",
    "Retain specific terminology from the text.",
    "Answer as accurately as possible in a compact form.",
)

SOURCE_PROMPTS = tuple(
    f"{request} {instruction}\nAnswer:"
    for instruction in INSTRUCTIONS
    for request in REQUESTS
)

assert len(SOURCE_PROMPTS) == len(set(SOURCE_PROMPTS)) == 480

PROMPT_SOURCE = "https://github.com/THUNLP/latent-compaction/blob/81cfd4b3f30f820fb5f76451cb0f6385e0ac9510/latent_compaction/prompts.py"
