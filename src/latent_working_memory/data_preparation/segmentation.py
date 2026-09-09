from __future__ import annotations

import re
from dataclasses import dataclass

import pysbd


@dataclass(frozen=True, slots=True)
class Sentence:
    start: int
    end: int
    paragraph: int


def sentence_spans(text: str) -> list[Sentence]:
    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    sentences = []
    for paragraph, match in enumerate(re.finditer(r"[^\r\n]+", text)):
        # Quotes keep their original offsets but do not hide internal sentence boundaries.
        boundary_text = match.group().translate(str.maketrans({'"': " ", "“": " ", "”": " "}))
        for span in segmenter.segment(boundary_text):
            start, end = match.start() + span.start, match.start() + span.end
            while start > match.start() and text[start - 1] in '"“':
                start -= 1
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start + 1 and (
                text[end - 1] == "“" or (text[end - 1] == '"' and text[end - 2].isspace())
            ):
                end -= 1
                while end > start and text[end - 1].isspace():
                    end -= 1
            if start < end:
                sentences.append(Sentence(start, end, paragraph))
    return sentences
