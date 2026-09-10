import pytest

from latent_working_memory.data_preparation.segmentation import sentence_spans
from latent_working_memory.data_preparation.inspection import judgment_statistics


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Dr. Smith paid 3.14 dollars. He left.", ["Dr. Smith paid 3.14 dollars.", "He left."]),
        (
            "A sentence crosses\na line break. Another begins.",
            ["A sentence crosses\na line break.", "Another begins."],
        ),
        ('"Stop!" he shouted. She left.', ['"Stop!" he shouted.', "She left."]),
        (
            'He said, "Go now. Come back soon." She stayed.',
            ['He said, "Go now. Come back soon."', "She stayed."],
        ),
        ("This is complete. An unfinished tail", ["This is complete."]),
        ("without a beginning. This sentence is complete.", ["This sentence is complete."]),
        (
            "First complete sentence.\r\nSecond complete sentence.",
            ["First complete sentence.", "Second complete sentence."],
        ),
    ],
)
def test_sentence_offsets_and_ambiguous_boundaries(text, expected):
    spans = sentence_spans(text)
    assert [text[s.start : s.end] for s in spans] == expected
    assert all(a.end <= b.start for a, b in zip(spans, spans[1:]))


def test_truncated_web_summary_is_not_a_sentence_endpoint():
    text = "A complete sentence. The state is soon to...\nA new article begins here."
    ends = {s.end for s in sentence_spans(text)}
    assert text.index("...") + 3 not in ends
    assert len(text) in ends


def test_threshold_needs_completed_independent_judgments():
    passed = {"judgment": "pass", "reviewer": "test", "review_reason": "all boundaries valid"}
    stats = judgment_statistics([passed] * 200)
    assert stats["success_rate"] == 1
    assert stats["lower_bound_above_threshold"]
    assert not judgment_statistics([passed] * 20)["lower_bound_above_threshold"]
    pending = judgment_statistics([passed, {"judgment": None}])
    assert pending["success_rate"] is None
    assert pending["above_threshold"] is None
