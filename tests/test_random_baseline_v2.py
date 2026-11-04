"""Tests for the no-immediate-inverse random baseline."""
import unittest

from zeromethod.environment.actions import inverse_action
from zeromethod.environment.adapter import GymRubiksCubeAdapter
from zeromethod.experiments.random_baseline import (
    NoImmediateInverseRandomPolicy,
    evaluate_no_inverse_baseline,
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

    def reset(self):
        self.state = SOLVED
        self.solved = True
        return self.state

    def step(self, action):
        self.state = self.state[1:] + self.state[:1]
        self.solved = False
        return self.state, 0.0, False, {}

    def isSolved(self):
        return self.solved

    def close(self):
        pass


class TestNoImmediateInverseBaseline(unittest.TestCase):
    def make_env(self):
        return GymRubiksCubeAdapter(FakeEnvironment())

    def test_policy_never_selects_previous_actions_inverse(self):
        policy = NoImmediateInverseRandomPolicy(1234)
        actions = [policy(SOLVED) for _ in range(1000)]
        self.assertTrue(all(0 <= action < 18 for action in actions))
        for previous, current in zip(actions, actions[1:]):
            self.assertNotEqual(current, inverse_action(previous))
        self.assertEqual(policy.policy_calls, 1000)

    def test_policy_is_reproducible(self):
        first = NoImmediateInverseRandomPolicy(99)
        second = NoImmediateInverseRandomPolicy(99)
        actions_a = [first(SOLVED) for _ in range(100)]
        actions_b = [second(SOLVED) for _ in range(100)]
        self.assertEqual(actions_a, actions_b)

    def test_evaluation_reports_distinct_policy_name_and_metrics(self):
        report = evaluate_no_inverse_baseline(
            self.make_env(), episodes=4, scramble_depth=5,
            max_agent_steps=7, master_seed=99,
        )
        self.assertEqual(report.policy, "uniform_random_no_immediate_inverse")
        self.assertEqual(report.episodes, 4)
        self.assertEqual(report.policy_calls, 28)
        self.assertEqual(report.truncations, 4)
        self.assertEqual(report.success_rate, 0.0)
        self.assertEqual(report.to_dict()["policy"], report.policy)

    def test_rejects_invalid_experiment_dimensions(self):
        with self.assertRaises(ValueError):
            evaluate_no_inverse_baseline(
                self.make_env(), episodes=0, scramble_depth=5,
                max_agent_steps=7, master_seed=99,
            )


if __name__ == "__main__":
    unittest.main()
