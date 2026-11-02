"""Tests for the final greedy one-step-lookahead baseline."""
import unittest

from zeromethod.environment.adapter import GymRubiksCubeAdapter
from zeromethod.experiments.greedy_policy_v3 import (
    GreedyOneStepPolicy,
    SOLVED_STATE,
    count_correct_stickers,
)
from zeromethod.experiments.greedy_baseline_v3 import (
    evaluate_greedy_baseline_v3,
)


class ActionSpace:
    n = 18


class ReversibleFakeEnvironment:
    """Small reversible transition model with the project's inverse pairing."""

    def __init__(self):
        self.action_space = ActionSpace()
        self.max_steps = None
        self.state = SOLVED_STATE
        self.solved = True
        self.transition_calls = 0

    def reset(self):
        self.state = SOLVED_STATE
        self.solved = True
        return self.state

    def step(self, action):
        self.transition_calls += 1
        offset = action // 2 + 1
        if action % 2 == 0:
            self.state = self.state[offset:] + self.state[:offset]
        else:
            self.state = self.state[-offset:] + self.state[:-offset]
        self.solved = self.state == SOLVED_STATE
        return self.state, 0.0, False, {}

    def isSolved(self):
        return self.solved

    def close(self):
        pass


class TestGreedyBaselineV3(unittest.TestCase):
    def make_env(self):
        raw = ReversibleFakeEnvironment()
        return GymRubiksCubeAdapter(raw), raw

    def test_sticker_score(self):
        self.assertEqual(count_correct_stickers(SOLVED_STATE), 54)
        changed = (1,) + SOLVED_STATE[1:]
        self.assertEqual(count_correct_stickers(changed), 53)

    def test_policy_probes_all_actions_and_restores_state(self):
        env, raw = self.make_env()
        # Use a valid, non-solved state with the same color counts.
        raw.state = SOLVED_STATE[1:] + SOLVED_STATE[:1]
        raw.solved = False
        observation = raw.state
        policy = GreedyOneStepPolicy(env, seed=17)

        action = policy(observation)

        self.assertIn(action, range(18))
        self.assertEqual(raw.state, observation)
        self.assertEqual(raw.transition_calls, 36)
        self.assertEqual(policy.policy_calls, 1)

    def test_policy_is_reproducible(self):
        env_a, raw_a = self.make_env()
        env_b, raw_b = self.make_env()
        state = SOLVED_STATE[1:] + SOLVED_STATE[:1]
        raw_a.state = state
        raw_b.state = state
        raw_a.solved = raw_b.solved = False
        first = GreedyOneStepPolicy(env_a, seed=123)
        second = GreedyOneStepPolicy(env_b, seed=123)
        self.assertEqual(first(state), second(state))

    def test_evaluator_uses_shared_report_schema(self):
        env, _ = self.make_env()
        report = evaluate_greedy_baseline_v3(
            env, episodes=3, scramble_depth=4,
            max_agent_steps=5, master_seed=42,
        )
        self.assertEqual(report.policy, GreedyOneStepPolicy.name)
        self.assertEqual(report.episodes, 3)
        self.assertEqual(report.policy_calls, 15)
        self.assertEqual(report.to_dict()["policy"], report.policy)

    def test_policy_rejects_wrong_action_space(self):
        class WrongEnvironment:
            action_count = 17
        with self.assertRaises(ValueError):
            GreedyOneStepPolicy(WrongEnvironment(), seed=1)


if __name__ == "__main__":
    unittest.main()
