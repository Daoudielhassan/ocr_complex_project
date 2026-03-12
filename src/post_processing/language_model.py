"""Rule-based language-model post-correction."""
from __future__ import annotations

import re

from src.utils.logger import get_logger

log = get_logger(__name__)

# Each entry is (compiled pattern, replacement string).
# Rules are applied in order on the joined token string.
_RULES: list[tuple[re.Pattern, str]] = [
    # Isolated lowercase "l" is almost always capital "I"
    (re.compile(r"(?<!\w)l(?!\w)"), "I"),
    # Isolated zero is often capital "O" when surrounded by letters
    (re.compile(r"(?<=[A-Za-z])0(?=[A-Za-z])"), "O"),
    # Broken hyphenation: "some- thing" → "something"
    (re.compile(r"(\w)-\s+(\w)"), r"\1\2"),
    # Multiple spaces → single space
    (re.compile(r" {2,}"), " "),
]


def apply_lm(tokens: list[str]) -> list[str]:
    """Apply lightweight rule-based corrections to a token list.

    This is a deterministic post-corrector.  It does **not** use a neural
    language model.

    Parameters
    ----------
    tokens:
        List of word / punctuation strings (output of the OCR pipeline).

    Returns
    -------
    list[str]
        Corrected token list.
    """
    text = " ".join(tokens)
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    result = text.split()
    log.debug("apply_lm: %d tokens → %d tokens", len(tokens), len(result))
    return result
