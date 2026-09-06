"""Repair ligatures that the prospectus font loses during PDF extraction.

The embedded font maps its ti/tt/ft/tti ligature glyphs to U+0000, so pypdf
yields "cri\x00cal" for "critical" and "so\x00ware" for "software" - about 1000
broken words in the 2026-27 prospectus alone. A blind "ti" substitution fixes
most but corrupts often/software/committed/settings.

We instead try every plausible expansion and keep the one the Qwen tokenizer
considers a real word: a genuine word costs fewer BPE tokens than a misspelling
("software" is 1 token, "sotiware" is 4). This needs no wordlist and keeps
working for prospectuses uploaded later.
"""

import re
from functools import lru_cache
from itertools import product

from backend.config import LLM_PATH

PLACEHOLDER = "\x00"
# Ordered by frequency; "ti" dominates, the rest are the real exceptions.
CANDIDATES = ("ti", "tt", "ft", "tti", "fi", "fl", "tf")

# Word-final placeholders are ambiguous ("shi\x00"->shift but "Gujara\x00"->Gujarati)
# and proper nouns aren't in the tokenizer as single pieces. Pin those here.
OVERRIDES = {
    "gujara\x00": "Gujarati",
    "gujara\x00,": "Gujarati,",
}

_WORD_RE = re.compile(r"\S*\x00\S*")


@lru_cache(maxsize=1)
def _tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(LLM_PATH)
    except Exception:
        return None  # fall back to plain "ti" substitution


def _score(tokenizer, word: str, added: int) -> tuple:
    ids = tokenizer.encode(" " + word)
    longest = max(len(tokenizer.decode([i])) for i in ids)
    # Fewest tokens wins; then the most word-like split; then the shortest fill.
    return (len(ids), -longest, added)


@lru_cache(maxsize=8192)
def _repair_word(word: str) -> str:
    override = OVERRIDES.get(word.lower())
    if override is not None:
        return override

    count = word.count(PLACEHOLDER)
    tokenizer = _tokenizer()
    if tokenizer is None or count == 0 or count > 4:
        return word.replace(PLACEHOLDER, "ti")

    best_word, best_score = None, None
    for combo in product(CANDIDATES, repeat=count):
        candidate = word
        for fill in combo:
            candidate = candidate.replace(PLACEHOLDER, fill, 1)
        score = _score(tokenizer, candidate, sum(len(f) for f in combo))
        if best_score is None or score < best_score:
            best_word, best_score = candidate, score
    return best_word


def repair_ligatures(text: str) -> str:
    if PLACEHOLDER not in text:
        return text
    return _WORD_RE.sub(lambda m: _repair_word(m.group(0)), text)
