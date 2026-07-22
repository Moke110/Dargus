"""Tests for dargus.tui._logo module."""

from rich.text import Text

from dargus.tui._logo import TAGLINE, A, D, G, R, S, U, build_logo


def test_d_letter_is_8_lines():
    assert len(D) == 8


def test_d_letter_consistent_width():
    """D lines are all 10 chars wide."""
    widths = {len(line) for line in D}
    assert widths == {10}, f"D has unexpected widths: {widths}"


def test_argus_letters_are_6_lines():
    for name, lines in [("A", A), ("R", R), ("G", G), ("U", U), ("S", S)]:
        assert len(lines) == 6, f"{name} should have 6 lines"


def test_build_logo_returns_8_text_lines():
    lines = build_logo()
    assert len(lines) == 8
    for line in lines:
        assert isinstance(line, Text)


def test_tagline_is_non_empty():
    assert "Data-driven Analysis" in TAGLINE
    assert "Grounded in Unified Science" in TAGLINE


def test_argus_letters_consistent_width():
    """Each ARGUS letter must have lines of uniform length."""
    for name, lines in [("A", A), ("R", R), ("G", G), ("U", U), ("S", S)]:
        widths = {len(line) for line in lines}
        assert len(widths) == 1, f"'{name}' letter has inconsistent line widths: {widths}"


def test_build_logo_d_plus_argus_rows_uniform_width():
    """Rows with D+ARGUS (bottom 6) all have the same width."""
    result = build_logo()
    # First 2 rows are D-only (10 cols), bottom 6 are D+ARGUS
    argus_rows = result[2:]  # rows where ARGUS appears
    widths = [len(line.plain) for line in argus_rows]
    assert len(set(widths)) == 1, f"D+ARGUS rows have varying widths: {sorted(set(widths))}"


def test_a_letter_has_pupil():
    """Every letter in DARGUS must contain the ○ (hollow ring pupil) character."""
    for name, lines in [("D", D), ("A", A), ("R", R), ("G", G), ("U", U), ("S", S)]:
        has_pupil = any("○" in line for line in lines)
        assert has_pupil, f"'{name}' letter should contain ○ (hollow ring pupil)"
