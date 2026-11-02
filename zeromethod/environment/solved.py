"""Observation-level solved-state semantics for GymRubiksCube.

The upstream environment defines solved as six uniform 3x3 faces. The color
assigned to each face may change after middle-slice turns, so comparing to one
fixed 54-sticker color ordering is incorrect.
"""

from operator import index


def is_solved_observation(observation):
    """Return whether each consecutive nine-sticker face is monochromatic."""
    try:
        values = tuple(index(value) for value in observation)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation must contain integer sticker colors") from exc

    if len(values) != 54:
        raise ValueError("observation must contain exactly 54 stickers")
    if any(value < 0 or value > 5 for value in values):
        raise ValueError("sticker colors must be in the range 0 through 5")

    return all(
        len(set(values[offset:offset + 9])) == 1
        for offset in range(0, 54, 9)
    )
