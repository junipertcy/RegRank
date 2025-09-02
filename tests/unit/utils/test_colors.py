# tests/unit/utils/test_colors.py

from unittest.mock import patch

import pytest

# Assuming your project structure is src/regrank, this absolute import is standard
from regrank.utils.colors import (
    generate_adjacent_colors,
    generate_complementary_colors,
    hex_to_rgb,
    reverse_dict,
    rgb_to_hex,
)

# -- Tests for reverse_dict ----------------------------------------------------


@pytest.mark.parametrize(
    ("input_dict", "expected_output"),
    [
        ({"a": 1, "b": 2, "c": 1}, {1: ["a", "c"], 2: ["b"]}),
        ({1: "x", 2: "y", 3: "z"}, {"x": [1], "y": [2], "z": [3]}),
        ({}, {}),
        ({"a": 1, "b": 1, "c": 1}, {1: ["a", "b", "c"]}),
    ],
)
def test_reverse_dict(input_dict, expected_output):
    """Tests that reverse_dict correctly inverts dictionary mappings."""
    # The output order within the lists can vary, so we sort for a stable comparison.
    result = reverse_dict(input_dict)
    for key in result:
        result[key].sort()
    assert result == expected_output


# -- Tests for color conversions ---------------------------------------------


@pytest.mark.parametrize(
    ("rgb_tuple", "hex_string"),
    [
        ((255, 0, 0), "#ff0000"),  # Red
        ((0, 255, 0), "#00ff00"),  # Green
        ((0, 0, 255), "#0000ff"),  # Blue
        ((0, 0, 0), "#000000"),  # Black
        ((255, 255, 255), "#ffffff"),  # White
        ((16, 32, 64), "#102040"),  # A custom color
    ],
)
def test_rgb_hex_conversions(rgb_tuple, hex_string):
    """Tests the round-trip conversion between RGB and hex color formats."""
    # Test rgb_to_hex
    assert rgb_to_hex(rgb_tuple) == hex_string
    # Test hex_to_rgb
    assert hex_to_rgb(hex_string) == rgb_tuple


# -- Tests for color generation ----------------------------------------------


def test_generate_complementary_colors():
    """
    Tests the properties of the output from generate_complementary_colors.
    It checks for the correct number of colors and valid hex format.
    """
    k = 5
    colors = generate_complementary_colors(k)

    # 1. Check if the correct number of colors are generated
    assert len(colors) == k

    # 2. Check if all generated items are valid hex color strings
    for color in colors:
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7
        # Ensure all characters are valid hex digits
        int(color[1:], 16)


@patch("random.randint", return_value=10)
def test_generate_adjacent_colors_deterministic(mock_randint):
    """
    Tests generate_adjacent_colors with a mocked random number generator
    to ensure the output is predictable and correct.
    """
    base_color_hex = "#808080"  # RGB (128, 128, 128)
    k = 3

    # Since randint is mocked to always return 10, the new RGB should be (138, 138, 138)
    expected_adjacent_hex = "#8a8a8a"  # 138 in hex is 8a

    colors = generate_adjacent_colors(base_color_hex, k)

    # 1. Check if the correct number of colors are generated
    assert len(colors) == k

    # 2. Check if all generated colors match the expected deterministic output
    for color in colors:
        assert color == expected_adjacent_hex
