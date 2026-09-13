from __future__ import annotations

from dataclasses import dataclass

import pysbd


@dataclass(frozen=True, slots=True)
class Sentence:
    start: int
    end: int


def sentence_spans(text: str) -> list[Sentence]:
    """Conservative sentence units with offsets into unchanged source text.

    Line breaks are layout hints, not sentence boundaries. Uncertain proposed cuts
    are merged, and an unfinished document tail is not a semantic endpoint.
    """
    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    boundary_text = text.translate(str.maketrans({"\r": " ", "\n": " "}))
    sentences = []
    start = len(text) - len(text.lstrip())
    for span in segmenter.segment(boundary_text):
        end = span.end
        while end > start and text[end - 1].isspace():
            end -= 1
        terminal = text[start:end].rstrip("\"'”’)]}")
        if terminal.endswith(("..", "…", "….")):
            continue
        if not terminal.endswith((".", "?", "!")):
            continue
        after = end
        while after < len(text) and text[after].isspace():
            after += 1
        following = text[after:].lstrip("\"'“‘([{")
        # A lowercase continuation often marks quoted speech attribution or a false cut.
        if following and following[0].islower():
            continue
        opening = text[start:end].lstrip("\"'“‘([{")
        if opening and not opening[0].islower():
            sentences.append(Sentence(start, end))
        start = after
    return sentences
