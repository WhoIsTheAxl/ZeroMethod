"""Tests for the lazy upstream environment factory."""
import types
import unittest
from unittest.mock import patch

from zeromethod.environment.factory import make_environment

class FakeActionSpace:
    n = 18

class FakeRawEnvironment:
    def __init__(self):
        self.action_space = FakeActionSpace()
        self.scramble_params = None
        self.max_steps = None
        self.closed = False
    def reset(self):
        return [color for color in range(6) for _ in range(9)]
    def step(self, action):
        return self.reset(), 0.0, False, {}
    def isSolved(self):
        return True
    def close(self):
        self.closed = True

class TestEnvironmentFactory(unittest.TestCase):
    def setUp(self):
        self.raw = FakeRawEnvironment()
        self.module = types.ModuleType("gym_rubiks_cube.envs.rubiksCubeEnv")
        self.module.RubiksCubeEnv = lambda: self.raw
        self.patcher = patch("zeromethod.environment.factory.import_module", return_value=self.module)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_creates_configured_wrapped_environment(self):
        env = make_environment(scramble_params=0, max_steps=250)
        self.assertEqual(env.action_count, 18)
        self.assertEqual(self.raw.scramble_params, 0)
        self.assertEqual(self.raw.max_steps, 250)
        observation, info = env.reset()
        self.assertEqual(len(observation), 54)
        self.assertEqual(info, {})
        env.close()
        self.assertTrue(self.raw.closed)

    def test_defaults_to_solved_start_and_no_step_limit(self):
        make_environment()
        self.assertEqual(self.raw.scramble_params, 0)
        self.assertIsNone(self.raw.max_steps)

    def test_rejects_invalid_scramble_depth(self):
        for value in (-1, True, 1.5, "3"):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    make_environment(scramble_params=value)

    def test_rejects_invalid_episode_limit(self):
        for value in (0, -1, True, 1.5, "10"):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    make_environment(max_steps=value)

    def test_missing_upstream_dependency_has_helpful_error(self):
        with patch("zeromethod.environment.factory.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(ImportError, "GymRubiksCube is unavailable"):
                make_environment()

if __name__ == "__main__":
    unittest.main()
