"""Integration tests for seeded episode initialization against upstream GymRubiksCube.

Run with the pinned upstream dependencies and:
    PYTHONPATH=third_party/GymRubiksCube python -m unittest discover -s tests -p "test_episode_integration.py" -v
"""

import unittest

from zeromethod.environment.episode import initialize_episode
from zeromethod.environment.factory import make_environment
from zeromethod.environment.state_legality import validate_observation


class TestUpstreamEpisodeInitialization(unittest.TestCase):
    def test_seeded_state_is_reproducible_and_has_exact_requested_depth(self):
        env = make_environment(max_steps=None)
        try:
            depth = 23
            first, first_info = initialize_episode(env, depth=depth, seed=20261104)
            self.assertEqual(first_info, {})
            validate_observation(first)
            self.assertEqual(env.env.steps_since_reset, depth)

            second, second_info = initialize_episode(env, depth=depth, seed=20261104)
            self.assertEqual(second_info, {})
            validate_observation(second)
            self.assertEqual(second, first)
            self.assertEqual(env.env.steps_since_reset, depth)
        finally:
            env.close()

    def test_independent_environments_reproduce_the_same_start_state(self):
        left = make_environment(max_steps=None)
        right = make_environment(max_steps=None)
        try:
            left_state, _ = initialize_episode(left, depth=17, seed=8675309)
            right_state, _ = initialize_episode(right, depth=17, seed=8675309)
            self.assertEqual(left_state, right_state)
            validate_observation(left_state)
            validate_observation(right_state)
            self.assertEqual(left.env.steps_since_reset, 17)
            self.assertEqual(right.env.steps_since_reset, 17)
        finally:
            left.close()
            right.close()


if __name__ == "__main__":
    unittest.main()
