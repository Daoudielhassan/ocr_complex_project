"""Regex-based field validators for common business document fields."""
from __future__ import annotations

import re

from src.utils.logger import get_logger

log = get_logger(__name__)

_VALIDATORS: dict[str, re.Pattern] = {
    "iban": re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{4,30}$"),
    "date_iso": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "date_eu": re.compile(r"^\d{2}[./]\d{2}[./]\d{4}$"),
    "amount": re.compile(r"^\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?$"),
    "email": re.compile(r"^[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"),
    "phone": re.compile(r"^\+?[\d\s\-().]{7,20}$"),
}


def validate_field(value: str, field_type: str) -> bool:
    """Return *True* if *value* matches the pattern for *field_type*.

    Supported types: ``iban``, ``date_iso``, ``date_eu``, ``amount``,
    ``email``, ``phone``.

    Leading / trailing whitespace and internal spaces are removed before
    matching.
    """
    value = value.strip().replace(" ", "")
    pattern = _VALIDATORS.get(field_type.lower())
    if pattern is None:
        log.warning("Unknown field type: %r", field_type)
        return False
    result = bool(pattern.match(value))
    log.debug("validate_field(%r, %r) → %s", value, field_type, result)
    return result


def validate_fields(
    data: dict[str, str],
    schema: dict[str, str],
) -> dict[str, bool]:
    """Validate multiple named fields at once.

    Parameters
    ----------
    data:
        ``{field_name: raw_value}``
    schema:
        ``{field_name: field_type}``

    Returns
    -------
    dict[str, bool]
        Validation result per field.
    """
    return {k: validate_field(data.get(k, ""), schema[k]) for k in schema}
