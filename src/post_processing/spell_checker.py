"""Dictionary-based spell correction via pyspellchecker."""
from __future__ import annotations

from functools import lru_cache

from src.utils.logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=4)
def _get_spell(language: str):
    """Lazily create and cache a SpellChecker for *language*."""
    try:
        from spellchecker import SpellChecker
        return SpellChecker(language=language)
    except ImportError as exc:
        raise ImportError(
            "pyspellchecker is required for spell correction. "
            "Install it with:  pip install pyspellchecker"
        ) from exc


def correct_word(word: str, language: str = "en") -> str:
    """Return the most likely correct spelling for a single *word*.

    Non-alphabetic tokens and single characters are returned unchanged.

    Parameters
    ----------
    word:
        Input token.
    language:
        Language code accepted by pyspellchecker (e.g. ``"en"``, ``"fr"``).

    Returns
    -------
    str
        Corrected token (or *word* unchanged if no correction is found).
    """
    if not word.isalpha() or len(word) < 2:
        return word

    spell = _get_spell(language)
    corrected = spell.correction(word)
    return corrected if corrected is not None else word


def correct_tokens(tokens: list[str], language: str = "en") -> list[str]:
    """Apply :func:`correct_word` to every token in *tokens*."""
    return [correct_word(t, language) for t in tokens]
