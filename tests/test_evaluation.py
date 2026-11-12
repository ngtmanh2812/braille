from evaluation import character_error_rate, levenshtein_distance


def test_levenshtein_and_cer():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert character_error_rate("abc", "axc") == 1 / 3
    assert character_error_rate("", "") == 0
