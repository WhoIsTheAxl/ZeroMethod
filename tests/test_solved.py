"""Tests for the upstream-compatible solved-state predicate."""
import unittest

from zeromethod.environment.solved import is_solved_observation


SOLVED = tuple(color for color in range(6) for _ in range(9))


class TestSolvedObservation(unittest.TestCase):
    def test_canonical_solved_state(self):
        self.assertTrue(is_solved_observation(SOLVED))

    def test_face_uniform_state_with_relocated_centers_is_solved(self):
        # Middle-slice turns can permute which color occupies each face.
        relocated = tuple(color for color in (2, 5, 1, 4, 0, 3) for _ in range(9))
        self.assertTrue(is_solved_observation(relocated))

    def test_non_uniform_face_is_not_solved(self):
        state = list(SOLVED)
        state[0], state[9] = state[9], state[0]
        self.assertFalse(is_solved_observation(state))

    def test_invalid_observations_are_rejected(self):
        with self.assertRaises(ValueError):
            is_solved_observation([0] * 53)
        with self.assertRaises(ValueError):
            is_solved_observation([6] * 54)


if __name__ == "__main__":
    unittest.main()
