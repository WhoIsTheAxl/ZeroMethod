"""Tests for the uniform-random baseline experiment."""
import unittest

from zeromethod.environment.adapter import GymRubiksCubeAdapter
from zeromethod.experiments.random_baseline import (
    RandomPolicy,
    evaluate_random_baseline,
)

SOLVED = tuple(color for color in range(6) for _ in range(9))


class ActionSpace:
    n = 18


class FakeEnvironment:
    def __init__(self):
        self.action_space = ActionSpace()
        self.max_steps = None
        self.state = SOLVED
        self.solved = True
        self.actions = []

    def reset(self):
        self.state = SOLVED
        self.solved = True
        self.actions = []
        return self.state

    def step(self, action):
        self.actions.append(action)
        # A deliberately simple non-solving transition for deterministic tests.
        self.state = self.state[1:] + self.state[:1]
        self.solved = False
        return self.state, 0.0, False, {}

    def isSolved(self):
        return self.solved

    def close(self):
        pass


class TestRandomBaseline(unittest.TestCase):
    def make_env(self):
        return GymRubiksCubeAdapter(FakeEnvironment())

    def test_random_policy_is_seeded_and_uses_full_action_space(self):
        first = RandomPolicy(123)
        second = RandomPolicy(123)
        observations = [SOLVED] * 50
        actions_a = [first(obs) for obs in observations]
        actions_b = [second(obs) for obs in observations]
        self.assertEqual(actions_a, actions_b)
        self.assertTrue(all(0 <= action < 18 for action in actions_a))
        self.assertEqual(first.policy_calls, 50)
        self.assertGreaterEqual(first.policy_seconds, 0.0)

    def test_evaluation_is_reproducible_except_timing(self):
        first = evaluate_random_baseline(
            self.make_env(), episodes=4, scramble_depth=5,
            max_agent_steps=7, master_seed=99,
        )
        second = evaluate_random_baseline(
            self.make_env(), episodes=4, scramble_depth=5,
            max_agent_steps=7, master_seed=99,
        )
        self.assertEqual(first.successes, second.successes)
        self.assertEqual(first.policy_calls, second.policy_calls)
        self.assertEqual(
            [(r.agent_steps, r.success, r.total_reward) for r in first.episode_results],
            [(r.agent_steps, r.success, r.total_reward) for r in second.episode_results],
        )
        self.assertEqual(first.success_rate, 0.0)
        self.assertEqual(first.truncations, 4)
        self.assertEqual(first.to_dict()["policy"], "uniform_random_18_actions")

    def test_rejects_invalid_experiment_dimensions(self):
        for kwargs, error in (
            ({"episodes": 0}, ValueError),
            ({"scramble_depth": 0}, ValueError),
            ({"max_agent_steps": -1}, ValueError),
            ({"episodes": True}, TypeError),
            ({"master_seed": 1.5}, TypeError),
        ):
            values = dict(
                episodes=2, scramble_depth=3, max_agent_steps=5, master_seed=7
            )
            values.update(kwargs)
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(error):
                    evaluate_random_baseline(self.make_env(), **values)


if __name__ == "__main__":
    unittest.main()
