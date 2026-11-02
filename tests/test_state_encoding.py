"""Tests for the dependency-free neural state encoder."""
import unittest

from zeromethod.agents.state_encoding import ENCODED_SIZE, encode_observation

SOLVED = tuple(color for color in range(6) for _ in range(9))


class TestStateEncoding(unittest.TestCase):
    def test_one_hot_shape_and_values(self):
        encoded = encode_observation(SOLVED)
        self.assertEqual(len(encoded), ENCODED_SIZE)
        self.assertEqual(ENCODED_SIZE, 324)
        self.assertEqual(set(encoded), {0.0, 1.0})
        for sticker in range(54):
            segment = encoded[sticker * 6:(sticker + 1) * 6]
            self.assertEqual(sum(segment), 1.0)
            self.assertEqual(segment[SOLVED[sticker]], 1.0)

    def test_rejects_invalid_observation(self):
        with self.assertRaises(ValueError):
            encode_observation((0,) * 54)


if __name__ == "__main__":
    unittest.main()
