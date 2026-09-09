from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping
from urllib.parse import urlsplit


QUALITY_FILTER = {
    "language": "en",
    "min_language_score": 0.8,
    "min_sentence_words": 1,
    "min_sentence_letter_fraction": 0.5,
    "max_repeated_fivegram_fraction": 0.2,
    "max_sentence_occurrences": 2,
    "require_sentence_termination": True,
    "preserve_contiguous_source": True,
    "split_quoted_sentences": True,
    "dedup_input_min_tokens": 32,
    "exclude_trailing_ellipsis": True,
    "template_match": "sentence_start_or_copyright",
    "max_boilerplate_sentence_words": 120,
    "max_spam_phrase_occurrences": 2,
}

BOILERPLATE = re.compile(
    r"\b(?:all rights reserved|privacy policy|terms (?:of (?:use|service)|and conditions)|"
    r"(?:log|sign)[ -]?in|subscribe to|accept (?:all )?cookies|enable javascript|"
    r"click here|read more|leave a (?:reply|comment)|add to cart|your shopping cart|"
    r"forgot(?:ten)? (?:your )?password|file history|mime type|"
    r"(?:follow|like) (?:us|me|@\w+) on|click (?:any|the|a) (?:picture|image|photo)|"
    r"(?:type|enter) your email|prices shown in|(?:book now|save as) (?:button|link)|"
    r"click (?:the )?(?:next|following) (?:website|web) (?:page|link)|"
    r"contact us (?:below|to)|(?:get the whole|custom) (?:paper|essay)|"
    r"(?:respective )?copyright holders|not reviewed in advance|"
    r"we (?:respect your privacy|will not (?:publish|share))|"
    r"(?:available to (?:watch|stream|rent)|not available to stream)|"
    r"(?:full refund|guarantee policy)|no data so far|"
    r"(?:see|read) our (?:returns|refund|shipping) policy|"
    r"(?:buy|grab|order) (?:one|yours) now|added to your cart|"
    r"sign up for|hold harmless)\b",
    re.IGNORECASE,
)
METADATA = re.compile(
    r"^(?:subject|to|from|date|posted by|last updated|published on|category|tags?|"
    r"share|related posts?|comments?)\s*[:|]",
    re.IGNORECASE,
)
TEMPLATE_PATH = re.compile(
    r"/(?:privacy(?:[-_]policy)?|terms[-_](?:of[-_](?:use|service)|and[-_]conditions)|"
    r"contact[-_]?us|login|register|cart)/?$",
    re.IGNORECASE,
)
BIBLIOGRAPHY = re.compile(
    r"\bRetrieved\s+(?:\d|[A-Z][a-z]+ \d)|\bISBN[- :\d]|\bdoi:\s*10\."
    r"|^[A-Z][\w-]+, [^\n]{2,80}, [\"“][^\n]+[\"”] \((?:19|20)\d{2}\)\."
)
MARKUP = re.compile(
    r"<[/!a-zA-Z][^>]*>|\b(?:var|const|let)\s+\w+\s*=|function\s*\("
    r"|\b(?:nbsp|quot);|&#\d{2,5};"
)
SPAM_PHRASES = re.compile(
    r"\b(?:male enhancement|cheap jerseys|jerseys (?:for sale|cheap)|"
    r"casino (?:slots|bonus)|free (?:casino|spins)|custom essay)\b",
    re.IGNORECASE,
)
WORDS = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


def document_rejection_reason(record: Mapping[str, Any]) -> str | None:
    for name in ("text", "id", "url", "language"):
        if not isinstance(record[name], str) or not record[name]:
            raise ValueError(f"FineWeb {name} must be a non-empty string")
    if record["language"] != QUALITY_FILTER["language"] or (
        record["language_score"] < QUALITY_FILTER["min_language_score"]
    ):
        return "language"
    if TEMPLATE_PATH.search(urlsplit(record["url"]).path):
        return "template_page"
    if len(SPAM_PHRASES.findall(record["text"])) > QUALITY_FILTER["max_spam_phrase_occurrences"]:
        return "keyword_spam"
    return None


def sentence_quality_flags(text: str) -> list[str]:
    flags = []
    if "..." in text or "…" in text:
        flags.append("ellipsis")
    if BOILERPLATE.search(text):
        flags.append("template_phrase")
    return flags


def sentence_rejection_reason(text: str) -> str | None:
    stripped = text.strip()
    words = WORDS.findall(stripped)
    if len(words) < QUALITY_FILTER["min_sentence_words"] or (len(words) == 1 and len(words[0]) < 2):
        return "too_few_words"
    if re.fullmatch(
        r"(?:home|menu|previous|next|login|logout|subscribe|search)[.!?]", stripped, re.I
    ):
        return "boilerplate"
    if METADATA.search(stripped) or (
        len(words) <= QUALITY_FILTER["max_boilerplate_sentence_words"]
        and (
            BOILERPLATE.match(stripped)
            or re.search(r"\b(?:copyright holders|all rights reserved)\b", stripped, re.I)
        )
    ):
        return "boilerplate"
    if MARKUP.search(stripped):
        return "markup_or_code"
    ending = stripped.rstrip("\"'”’»)]} ")
    if ending.endswith(("...", "…")):
        return "ellipsis_fragment"
    if not ending.endswith((".", "!", "?")) or ending.endswith("..."):
        return "unfinished_sentence"
    visible = [c for c in stripped if not c.isspace() and c not in ".,!?;:'\"“”‘’()[]{}-"]
    if (
        sum(c.isalpha() for c in visible) / len(visible)
        < QUALITY_FILTER["min_sentence_letter_fraction"]
    ):
        return "low_letter_fraction"
    if len(words) >= 25:
        fivegrams = list(zip(*(tuple(w.lower() for w in words)[i:] for i in range(5))))
        repeated = sum(count - 1 for count in Counter(fivegrams).values()) / len(fivegrams)
        if repeated > QUALITY_FILTER["max_repeated_fivegram_fraction"]:
            return "repeated_text"
    return None
