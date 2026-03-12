"""Reconstruct readable text from nested character predictions."""
from __future__ import annotations

from src.utils.logger import get_logger

log = get_logger(__name__)


def rebuild_text(
    all_char_predictions: list[list[list[str]]],
) -> str:
    """Reconstruct plain text from the output of the inference loop.

    Parameters
    ----------
    all_char_predictions:
        3-level nested structure:
        ``blocks → lines → chars`` where each char is a single string
        (including ``" "`` space tokens inserted between words).

    Returns
    -------
    str
        Full reconstructed text.  Blocks are separated by a blank line;
        lines within a block are separated by ``\\n``.
    """
    block_texts: list[str] = []
    for block_lines in all_char_predictions:
        line_texts: list[str] = []
        for line_chars in block_lines:
            line = "".join(line_chars).strip()
            if line:
                line_texts.append(line)
        if line_texts:
            block_texts.append("\n".join(line_texts))

    text = "\n\n".join(block_texts)
    log.debug("rebuild_text: %d chars reconstructed", len(text))
    return text
