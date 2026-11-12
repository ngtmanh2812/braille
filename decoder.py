"""Braille Grade 1 Latin letter patterns and Unicode braille encoding."""

from __future__ import annotations


_LETTER_DOTS: dict[str, tuple[int, ...]] = {
    "a": (1,), "b": (1, 2), "c": (1, 4), "d": (1, 4, 5), "e": (1, 5),
    "f": (1, 2, 4), "g": (1, 2, 4, 5), "h": (1, 2, 5), "i": (2, 4), "j": (2, 4, 5),
    "k": (1, 3), "l": (1, 2, 3), "m": (1, 3, 4), "n": (1, 3, 4, 5), "o": (1, 3, 5),
    "p": (1, 2, 3, 4), "q": (1, 2, 3, 4, 5), "r": (1, 2, 3, 5), "s": (2, 3, 4),
    "t": (2, 3, 4, 5), "u": (1, 3, 6), "v": (1, 2, 3, 6), "w": (2, 4, 5, 6),
    "x": (1, 3, 4, 6), "y": (1, 3, 4, 5, 6), "z": (1, 3, 5, 6),
}


def dots_to_bits(dots: tuple[int, ...] | list[int]) -> str:
    """Return a six-character 1–6 occupancy string from dot numbers."""
    active = set(dots)
    if any(dot < 1 or dot > 6 for dot in active):
        raise ValueError("Braille dot numbers must be between 1 and 6.")
    return "".join("1" if dot in active else "0" for dot in range(1, 7))


LETTER_TO_BITS: dict[str, str] = {letter: dots_to_bits(dots) for letter, dots in _LETTER_DOTS.items()}
BITS_TO_LETTER: dict[str, str] = {bits: letter for letter, bits in LETTER_TO_BITS.items()}


def text_to_patterns(text: str) -> list[str]:
    """Encode lowercase a-z as six-bit cells; reject unsupported characters."""
    if not isinstance(text, str) or any(char not in LETTER_TO_BITS for char in text):
        raise ValueError("Chỉ hỗ trợ chữ Latin thường a–z, không có dấu cách, số hoặc dấu câu.")
    return [LETTER_TO_BITS[char] for char in text]


def encode_text(text: str) -> str:
    """Convert lowercase Latin text to Unicode Grade 1 Braille."""
    return "".join(bits_to_unicode(pattern) for pattern in text_to_patterns(text))


def normalize_bits(pattern: str | tuple[bool, ...] | list[bool]) -> str:
    if isinstance(pattern, str):
        bits = pattern.replace(" ", "")
    else:
        bits = "".join("1" if bit else "0" for bit in pattern)
    if len(bits) != 6 or any(bit not in "01" for bit in bits):
        raise ValueError("A Braille pattern must contain exactly six binary values.")
    return bits


def bits_to_unicode(pattern: str | tuple[bool, ...] | list[bool]) -> str:
    """Encode a six-bit dot 1–6 pattern as its Unicode Braille cell."""
    bits = normalize_bits(pattern)
    value = sum((bit == "1") << index for index, bit in enumerate(bits))
    return chr(0x2800 + value)


def decode_pattern(pattern: str | tuple[bool, ...] | list[bool]) -> str:
    """Decode one Grade 1 lowercase Latin letter, or '?' for unsupported cells."""
    return BITS_TO_LETTER.get(normalize_bits(pattern), "?")


def decode_patterns(patterns: list[str]) -> tuple[str, str]:
    """Decode a list of cells into plain text and Unicode braille strings."""
    return "".join(decode_pattern(pattern) for pattern in patterns), "".join(
        bits_to_unicode(pattern) for pattern in patterns
    )
