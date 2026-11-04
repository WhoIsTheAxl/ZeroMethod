"""Standard-library tests for the dependency-free environment adapter."""

import unittest

from zeromethod.environment import GymRubiksCubeAdapter


class FakeActionSpace:
    n = 18


class FakeEnvironment:
    def __init__(self):
        self.action_space = FakeActionSpace()
        self.max_steps = None
        self.state = [color for color in range(6) for _ in range(9)]
        self.done = False
        self.reward = 0.0
        self.closed = False

    def reset(self):
        self.state = [color for color in range(6) for _ in range(9)]
        self.done = False
        self.reward = 0.0
        return list(self.state)

    def step(self, action):
        self.state = list(self.state)
        if action == 0:
            self.done = True
            self.reward = 1.0
        return list(self.state), self.reward, self.done, {"action": action}

    def isSolved(self):
        return self.done and self.reward > 0

    def close(self):
        self.closed = True


class TestGymRubiksCubeAdapter(unittest.TestCase):
    def setUp(self):
        self.raw_env = FakeEnvironment()
        self.env = GymRubiksCubeAdapter(self.raw_env)

    def test_reset_returns_immutable_observation_and_info(self):
        observation, info = self.env.reset()
        self.assertIsInstance(observation, tuple)
        self.assertEqual(len(observation), 54)
        self.assertEqual(info, {})

    def test_max_steps_is_exposed_without_reaching_through_adapter(self):
        self.assertIsNone(self.env.max_steps)
        self.raw_env.max_steps = 25
        self.assertEqual(self.env.max_steps, 25)

    def test_step_returns_normalized_result(self):
        self.env.reset()
        result = self.env.step(3)
        self.assertEqual(len(result.observation), 54)
        self.assertEqual(result.reward, 0.0)
        self.assertFalse(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.info, {"action": 3})

    def test_solved_state_is_termination(self):
        self.env.reset()
        result = self.env.step(0)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.reward, 1.0)

    def test_unsolved_done_state_is_truncation(self):
        self.env.reset()
        self.raw_env.done = True
        self.raw_env.reward = 0.0
        result = self.env.step(1)
        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)

    def test_rejects_non_integer_actions(self):
        for action in (True, 1.5, "2", None):
            with self.subTest(action=action):
                with self.assertRaises(TypeError):
                    self.env.step(action)

    def test_rejects_out_of_range_actions(self):
        for action in (-1, 18):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    self.env.step(action)

    def test_rejects_malformed_observations(self):
        for observation in (
            [0] * 53,
            [0] * 54,
            [color for color in range(6) for _ in range(9)][:-1] + [6],
            [0.5] + [color for color in range(6) for _ in range(9)][1:],
            ["0"] + [color for color in range(6) for _ in range(9)][1:],
        ):
            with self.subTest(length=len(observation)):
                with self.assertRaises(ValueError):
                    self.env._normalize_observation(observation)

    def test_close_delegates_to_wrapped_environment(self):
        self.env.close()
        self.assertTrue(self.raw_env.closed)


if __name__ == "__main__":
    unittest.main()
