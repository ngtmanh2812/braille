"""Explicit Latin text normalization for OCR output reviewed by the user."""

from __future__ import annotations

import string

from decoder import LETTER_TO_BITS, bits_to_unicode


# ICEB Rules of UEB, section 7: comma dot 2, apostrophe dot 3, period dots 256.
# ASCII ' is interpreted as an apostrophe, not a contextual quotation mark.
PUNCTUATION_TO_BITS = {",": "010000", "'": "001000", ".": "010011"}


def normalize_latin_line(text: str) -> str:
    """Accept ASCII letters, spaces, comma, apostrophe, and period.

    Unsupported symbols are rejected rather than silently removed. Uppercase is
    converted to lowercase; this function does not encode a capital indicator.
    """
    if not isinstance(text, str):
        raise ValueError("Vui lòng nhập một dòng chữ Latin.")
    if len(text) > 200:
        raise ValueError("Chỉ hỗ trợ tối đa 200 ký tự trong một dòng.")
    invalid = sorted(set(text) - set(string.ascii_letters + " ") - set(PUNCTUATION_TO_BITS))
    if invalid:
        symbols = ", ".join(repr(char) for char in invalid)
        raise ValueError(f"Ký tự chưa hỗ trợ: {symbols}. Hỗ trợ chữ a–z, dấu cách và các dấu , . ' (apostrophe).")
    normalized = " ".join(part for part in text.lower().split(" ") if part)
    if not normalized:
        raise ValueError("Vui lòng nhập ít nhất một chữ cái Latin.")
    return normalized


def encode_latin_line(text: str) -> str:
    """Encode a reviewed Latin line, keeping word gaps as Unicode blank cells."""
    return "".join(bits_to_unicode(bits) for bits in line_to_patterns(text))


def line_to_patterns(text: str) -> list[str]:
    """Create cells for supported letters, spaces, and three punctuation signs."""
    normalized = normalize_latin_line(text)
    mapping = {**LETTER_TO_BITS, **PUNCTUATION_TO_BITS, " ": "000000"}
    return [mapping[char] for char in normalized]
