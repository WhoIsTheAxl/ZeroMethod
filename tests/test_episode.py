"""Tests for seeded episode initialization."""
import unittest

from zeromethod.environment.adapter import GymRubiksCubeAdapter
from zeromethod.environment.episode import initialize_episode

SOLVED = tuple(color for color in range(6) for _ in range(9))


class ActionSpace:
    n = 18


class FakeEnvironment:
    def __init__(self):
        self.action_space = ActionSpace()
        self.max_steps = None
        self.state = SOLVED
        self.actions = []

    def reset(self):
        self.state = SOLVED
        self.actions = []
        return self.state

    def step(self, action):
        self.actions.append(action)
        amount = (1 if action % 2 == 0 else -1) % len(self.state)
        self.state = self.state[amount:] + self.state[:amount]
        return self.state, 0.0, False, {}

    def isSolved(self):
        return self.state == SOLVED

    def close(self):
        pass


class TestEpisodeInitialization(unittest.TestCase):
    def make_env(self):
        raw = FakeEnvironment()
        return raw, GymRubiksCubeAdapter(raw)

    def test_same_seed_reproduces_actions_and_state(self):
        raw_a, env_a = self.make_env()
        raw_b, env_b = self.make_env()
        a, info_a = initialize_episode(env_a, depth=20, seed=123)
        b, info_b = initialize_episode(env_b, depth=20, seed=123)
        self.assertEqual(a, b)
        self.assertEqual(raw_a.actions, raw_b.actions)
        self.assertEqual(len(raw_a.actions), 20)
        self.assertEqual(info_a, {})
        self.assertEqual(info_b, {})

    def test_different_seed_changes_walk(self):
        raw_a, env_a = self.make_env()
        raw_b, env_b = self.make_env()
        initialize_episode(env_a, depth=20, seed=1)
        initialize_episode(env_b, depth=20, seed=2)
        self.assertNotEqual(raw_a.actions, raw_b.actions)

    def test_zero_depth_starts_solved(self):
        raw, env = self.make_env()
        state, info = initialize_episode(env, depth=0, seed=7)
        self.assertEqual(state, SOLVED)
        self.assertEqual(info, {})
        self.assertEqual(raw.actions, [])

    def test_history_is_not_returned(self):
        _, env = self.make_env()
        state, info = initialize_episode(env, depth=8, seed=5)
        self.assertEqual(len(state), 54)
        self.assertEqual(info, {})

    def test_requires_no_upstream_step_limit(self):
        raw, env = self.make_env()
        raw.max_steps = 100
        with self.assertRaisesRegex(ValueError, "max_steps=None"):
            initialize_episode(env, depth=5, seed=1)

    def test_validates_depth_and_seed(self):
        _, env = self.make_env()
        for depth in (-1, True, 1.5):
            with self.subTest(depth=depth):
                with self.assertRaises((TypeError, ValueError)):
                    initialize_episode(env, depth=depth, seed=1)
        for seed in (True, 1.5, "1"):
            with self.subTest(seed=seed):
                with self.assertRaises(TypeError):
                    initialize_episode(env, depth=1, seed=seed)


if __name__ == "__main__":
    unittest.main()
