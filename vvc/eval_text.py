"""Word error rate for scoring transcripts (Unicode-aware, so Danish æ/ø/å count)."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np


def words(s: str) -> list[str]:
    return re.sub(r"[^\w' ]", " ", s.lower()).split()


def _same(a: str, b: str, lenient: bool) -> bool:
    if a == b:
        return True
    # lenient: forgive typos/spelling variants in longer words ("kafe" for "kaffe")
    return lenient and min(len(a), len(b)) >= 4 and SequenceMatcher(None, a, b).ratio() >= 0.8


def wer(ref: str, hyp: str, lenient: bool = False) -> float:
    r, h = words(ref), words(hyp)
    d = np.arange(len(h) + 1)
    for i, rw in enumerate(r, 1):
        prev, d[0] = d.copy(), i
        for j, hw in enumerate(h, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (not _same(rw, hw, lenient)))
    return d[len(h)] / max(1, len(r))


def words_correct(ref: str, hyp: str, lenient: bool = True) -> float:
    """Fraction of reference words recovered (1 - WER, floored at 0): the usual
    'percent words correct' of listening tests."""
    return max(0.0, 1.0 - wer(ref, hyp, lenient))
