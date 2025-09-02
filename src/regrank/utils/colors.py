# regrank/utils/colors.py
import random
from collections import defaultdict

import distinctipy


def reverse_dict(input_dict):
    reversed_dict = defaultdict(list)
    for key, value in input_dict.items():
        reversed_dict[value].append(key)
    return dict(reversed_dict)


def generate_complementary_colors(k):
    """
    Generate a list of k visually distinct complementary colors.

    Parameters:
    k (int): Number of complementary colors to generate.

    Returns:
    List[str]: List of hex color codes.
    """
    # Generate k visually distinct colors
    colors = distinctipy.get_colors(k, pastel_factor=0.9)

    # Convert the RGB tuples to hex color codes
    hex_colors = [rgb_to_hex(color) for color in colors]

    return hex_colors


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(
    rgb_color: tuple[int, int, int] | tuple[float, float, float],
) -> str:
    """
    Converts an RGB color tuple to a hex string.

    This function is robust and accepts two types of input:
    1. A tuple of floats, where each value is in the range [0.0, 1.0].
    2. A tuple of integers, where each value is in the range [0, 255].

    Args:
        rgb_color: A tuple containing the three RGB color components.

    Returns:
        A hex color string in the format '#RRGGBB'.

    Raises:
        ValueError: If the input values are outside their expected ranges.
        TypeError: If the tuple contains a mix of floats and ints.
    """
    # Case 1: Input is floats (e.g., from distinctipy)
    if all(isinstance(c, float) for c in rgb_color):
        if not all(0.0 <= c <= 1.0 for c in rgb_color):
            raise ValueError("Float RGB components must be between 0.0 and 1.0")
        # Scale floats [0,1] to integers [0,255]
        scaled_color = tuple(int(round(c * 255)) for c in rgb_color)

    # Case 2: Input is integers
    elif all(isinstance(c, int) for c in rgb_color):
        if not all(0 <= c <= 255 for c in rgb_color):
            raise ValueError("Integer RGB components must be between 0 and 255")
        scaled_color = tuple(int(c) for c in rgb_color)

    else:
        raise TypeError("RGB tuple must contain either all floats or all integers.")

    return "#{:02x}{:02x}{:02x}".format(*scaled_color)


def generate_adjacent_colors(hex_color: str, k: int) -> list[str]:
    """Generate a list of k adjacent hex colors near the given hex color."""
    base_rgb = hex_to_rgb(hex_color)
    adjacent_colors: list[str] = []

    for _ in range(k):
        new_rgb = tuple(
            max(0, min(255, base_rgb[i] + random.randint(-20, 20))) for i in range(3)
        )
        adjacent_colors.append(rgb_to_hex(new_rgb))

    return adjacent_colors
