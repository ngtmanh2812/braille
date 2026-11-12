from grid_alignment import GridConfig, align_dots
from decoder import LETTER_TO_BITS


def test_fixed_grid_keeps_absent_positions_and_dot_numbering():
    # Pattern 100101 means dots 1, 4, and 6 are present.
    centers = [(10, 10), (30, 10), (30, 50)]
    cells = align_dots(centers, config=GridConfig(
        mode="fixed", dot_spacing=20, cell_stride=60, origin_x=10, origin_y=10,
    ))
    assert len(cells) == 1
    assert cells[0].bits == "100101"
    assert cells[0].positions[1] is None
    assert cells[0].positions[3] == (30.0, 10.0)
    assert cells[0].positions[4] is None


def test_fixed_grid_aligns_multiple_cells():
    centers = [(10, 10), (70, 10), (90, 10)]
    cells = align_dots(centers, config=GridConfig(
        mode="fixed", dot_spacing=20, cell_stride=60, origin_x=10, origin_y=10,
    ))
    assert [cell.bits for cell in cells] == ["100000", "100100"]


def test_empty_input_returns_no_cells():
    assert align_dots([]) == []


def test_estimated_grid_handles_repeated_single_dot_cells():
    # A repeated "a" has no within-cell pair, so every observed dot is one cell stride apart.
    for count in (2, 4):
        centers = [(24 + i * 54, 24) for i in range(count)]
        cells = align_dots(centers)
        assert [cell.bits for cell in cells] == ["100000"] * count


def test_estimated_grid_compensates_for_mild_line_skew():
    centers = []
    text = "abcxyz"
    for cell_index, character in enumerate(text):
        for bit_index, bit in enumerate(LETTER_TO_BITS[character]):
            if bit == "1":
                x = 24 + cell_index * 54 + (bit_index // 3) * 18
                y = 24 + (bit_index % 3) * 18 + 0.04 * (x - 24)
                centers.append((x, y))
    cells = align_dots(centers)
    assert [cell.bits for cell in cells] == [LETTER_TO_BITS[char] for char in text]
