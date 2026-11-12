from decoder import BITS_TO_LETTER, LETTER_TO_BITS, bits_to_unicode, decode_pattern


EXPECTED_GRADE_1_BITS = {
    "a": "100000", "b": "110000", "c": "100100", "d": "100110", "e": "100010",
    "f": "110100", "g": "110110", "h": "110010", "i": "010100", "j": "010110",
    "k": "101000", "l": "111000", "m": "101100", "n": "101110", "o": "101010",
    "p": "111100", "q": "111110", "r": "111010", "s": "011100", "t": "011110",
    "u": "101001", "v": "111001", "w": "010111", "x": "101101", "y": "101111",
    "z": "101011",
}


def test_all_26_grade_1_letters_have_unique_patterns():
    assert LETTER_TO_BITS == EXPECTED_GRADE_1_BITS
    assert len(LETTER_TO_BITS) == 26
    assert len(BITS_TO_LETTER) == 26
    for letter, bits in LETTER_TO_BITS.items():
        assert len(bits) == 6
        assert decode_pattern(bits) == letter


def test_unicode_uses_standard_braille_block():
    assert bits_to_unicode("100000") == "\u2801"
    assert bits_to_unicode("000000") == "\u2800"


def test_unsupported_pattern_returns_question_mark():
    assert decode_pattern("000000") == "?"
