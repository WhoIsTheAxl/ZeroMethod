"""Tests for independent 3x3x3 cube-state legality validation."""

import unittest

from zeromethod.environment.state_legality import (
    is_legal_cube_state,
    validate_cube_state,
)


SOLVED = tuple(color for color in range(6) for _ in range(9))


class TestCubeStateLegality(unittest.TestCase):
    def test_solved_state_is_legal(self):
        validate_cube_state(SOLVED)
        self.assertTrue(is_legal_cube_state(SOLVED))

    def test_rejects_wrong_sticker_count(self):
        with self.assertRaisesRegex(ValueError, "54 stickers"):
            validate_cube_state(SOLVED[:-1])

    def test_rejects_wrong_color_counts(self):
        state = list(SOLVED)
        state[0] = 1
        with self.assertRaisesRegex(ValueError, "exactly nine"):
            validate_cube_state(state)

    def test_rejects_wrong_centers(self):
        state = list(SOLVED)
        state[4], state[13] = state[13], state[4]
        self.assertFalse(is_legal_cube_state(state))
        with self.assertRaisesRegex(ValueError, "center colors"):
            validate_cube_state(state)

    def test_rejects_single_edge_flip(self):
        state = list(SOLVED)
        state[7], state[10] = state[10], state[7]
        with self.assertRaisesRegex(ValueError, "edge flip sum"):
            validate_cube_state(state)

    def test_rejects_single_corner_twist(self):
        state = list(SOLVED)
        a, b, c = 8, 18, 11
        state[a], state[b], state[c] = state[c], state[a], state[b]
        with self.assertRaisesRegex(ValueError, "corner twist sum"):
            validate_cube_state(state)

    def test_rejects_two_edge_swap_without_corner_swap(self):
        state = list(SOLVED)
        first = (5, 19)  # UR and UF facelets
        second = (10, 7)
        state[first[0]], state[second[0]] = state[second[0]], state[first[0]]
        state[first[1]], state[second[1]] = state[second[1]], state[first[1]]
        with self.assertRaisesRegex(ValueError, "permutation parity"):
            validate_cube_state(state)

    def test_rejects_duplicate_or_impossible_cubie(self):
        state = list(SOLVED)
        state[8], state[18] = state[18], state[8]
        self.assertFalse(is_legal_cube_state(state))


if __name__ == "__main__":
    unittest.main()
