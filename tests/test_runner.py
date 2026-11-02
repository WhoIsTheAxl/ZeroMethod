"""Tests for the bounded episode runner."""
import unittest

from zeromethod.environment.adapter import GymRubiksCubeAdapter
from zeromethod.environment.runner import EpisodeRunner

SOLVED = tuple(color for color in range(6) for _ in range(9))


class ActionSpace:
    n = 18


class RunnerFakeEnvironment:
    def __init__(self):
        self.action_space = ActionSpace()
        self.max_steps = None
        self.state = SOLVED
        self.solved = False
        self.actions = []
        self.solve_action = 5
        self.reward_per_step = 0.25

    def reset(self):
        self.state = SOLVED
        self.solved = False
        self.actions = []
        return self.state

    def step(self, action):
        self.actions.append(action)
        if action == self.solve_action:
            self.solved = True
            self.state = SOLVED
            return self.state, 1.0, True, {}
        # Preserve the observation contract while remaining unsolved.
        self.state = self.state[1:] + self.state[:1]
        return self.state, self.reward_per_step, False, {}

    def isSolved(self):
        return self.solved

    def close(self):
        pass


class TestEpisodeRunner(unittest.TestCase):
    def make_runner(self):
        raw = RunnerFakeEnvironment()
        return raw, EpisodeRunner(GymRubiksCubeAdapter(raw))

    def run_args(self, **overrides):
        args = {
            "depth": 0,
            "seed": 42,
            "max_agent_steps": 10,
        }
        args.update(overrides)
        return args

    def test_policy_solves_and_result_counts_agent_actions_only(self):
        raw, runner = self.make_runner()
        seen = []
        def policy(observation):
            seen.append(observation)
            return 5

        result = runner.run(policy, **self.run_args())
        self.assertTrue(result.success)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.agent_steps, 1)
        self.assertEqual(result.total_reward, 1.0)
        self.assertEqual(raw.actions, [5])
        self.assertEqual(len(seen), 1)

    def test_step_limit_truncates_without_extra_policy_call(self):
        raw, runner = self.make_runner()
        calls = []
        def policy(observation):
            calls.append(observation)
            return 1

        result = runner.run(
            policy, **self.run_args(max_agent_steps=3)
        )
        self.assertFalse(result.success)
        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertEqual(result.agent_steps, 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(raw.actions), 3)
        self.assertEqual(result.total_reward, 0.75)

    def test_zero_step_budget_returns_truncated(self):
        _, runner = self.make_runner()
        result = runner.run(
            lambda observation: 1,
            **self.run_args(max_agent_steps=0),
        )
        self.assertEqual(result.agent_steps, 0)
        self.assertTrue(result.truncated)
        self.assertFalse(result.success)

    def test_solved_initial_state_ends_without_policy_call(self):
        _, runner = self.make_runner()
        def forbidden_policy(observation):
            self.fail("policy must not run for an already-solved state")

        result = runner.run(
            forbidden_policy, **self.run_args(depth=0)
        )
        self.assertTrue(result.success)
        self.assertTrue(result.terminated)
        self.assertEqual(result.agent_steps, 0)

    def test_rejects_non_callable_policy(self):
        _, runner = self.make_runner()
        with self.assertRaisesRegex(TypeError, "policy must be callable"):
            runner.run(None, **self.run_args())

    def test_rejects_invalid_step_budgets(self):
        _, runner = self.make_runner()
        for value in (-1,):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    runner.run(lambda obs: 1, **self.run_args(max_agent_steps=value))
        for value in (True, 1.5, "3"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    runner.run(lambda obs: 1, **self.run_args(max_agent_steps=value))

    def test_rejects_invalid_policy_actions(self):
        _, runner = self.make_runner()
        for action, error in ((True, TypeError), (1.5, TypeError), (99, ValueError)):
            with self.subTest(action=action):
                with self.assertRaises(error):
                    runner.run(
                        lambda observation, action=action: action,
                        **self.run_args(),
                    )


if __name__ == "__main__":
    unittest.main()
