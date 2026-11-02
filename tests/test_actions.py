"""Dependency-free tests for ZeroMethod's stable action interface."""

import unittest

from zeromethod.environment.actions import (
    ACTION_COUNT,
    INVERSE_ACTIONS,
    Action,
    inverse_action,
)


class TestActionInterface(unittest.TestCase):
    def test_action_values_match_the_upstream_18_action_contract(self):
        self.assertEqual(ACTION_COUNT, 18)
        self.assertEqual(len(Action), ACTION_COUNT)
        self.assertEqual([int(action) for action in Action], list(range(18)))

    def test_inverse_mapping_is_involutive(self):
        self.assertEqual(len(INVERSE_ACTIONS), ACTION_COUNT)
        for action in Action:
            with self.subTest(action=action):
                inverse = inverse_action(action)
                self.assertEqual(inverse_action(inverse), action)
                self.assertNotEqual(inverse, action)

    def test_inverse_mapping_matches_adjacent_upstream_pairs(self):
        for index in range(ACTION_COUNT):
            expected = index + 1 if index % 2 == 0 else index - 1
            self.assertEqual(int(inverse_action(index)), expected)

    def test_accepts_integer_like_indices(self):
        self.assertEqual(inverse_action(Action.U_COUNTERCLOCKWISE), Action.U_CLOCKWISE)

    def test_rejects_boolean_and_non_integer_values(self):
        for value in (True, False, 1.5, "1", None):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    inverse_action(value)

    def test_rejects_out_of_range_indices(self):
        for value in (-1, ACTION_COUNT):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    inverse_action(value)


if __name__ == "__main__":
    unittest.main()
