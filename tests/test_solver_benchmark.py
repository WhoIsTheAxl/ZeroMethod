"""Tests for solve timing, action recording, and state replay."""
import json
import unittest

from zeromethod.environment.adapter import GymRubiksCubeAdapter
from zeromethod.evaluation.solver_benchmark import (
    SolveRecorder,
    SolverBenchmark,
)

SOLVED = tuple(color for color in range(6) for _ in range(9))
SCRAMBLED = SOLVED[1:] + SOLVED[:1]


class ActionSpace:
    n = 18


class FakeEnvironment:
    def __init__(self, solve_on_action=True):
        self.action_space = ActionSpace()
        self.max_steps = None
        self.state = SOLVED
        self.solve_on_action = solve_on_action

    def reset(self):
        self.state = SOLVED
        return self.state

    def step(self, action):
        if self.state == SOLVED:
            self.state = SCRAMBLED
        elif self.solve_on_action:
            self.state = SOLVED
        return self.state, (1.0 if self.state == SOLVED else 0.0), self.state == SOLVED, {}

    def isSolved(self):
        return self.state == SOLVED

    def close(self):
        pass


class TestSolverBenchmark(unittest.TestCase):
    def make_benchmark(self, solve_on_action=True):
        env = GymRubiksCubeAdapter(FakeEnvironment(solve_on_action))
        return env, SolverBenchmark(env)

    def test_records_move_states_and_verified_success(self):
        env, benchmark = self.make_benchmark()
        report = benchmark.run(
            lambda observation: 1,
            depth=1, seed=7, max_agent_steps=4, policy_name="test-policy",
        )
        self.assertTrue(report.success)
        self.assertTrue(report.final_state_verified)
        self.assertEqual(report.agent_steps, 1)
        self.assertEqual(len(report.moves), 1)
        self.assertEqual(report.moves[0].action_name, "U_CLOCKWISE")
        self.assertEqual(report.moves[0].action, 1)
        self.assertEqual(report.states, (SCRAMBLED, SOLVED))
        self.assertEqual(report.moves[0].observation, SOLVED)
        self.assertGreaterEqual(report.solving_time_ns, 0)
        self.assertGreaterEqual(report.total_runtime_ns, report.solving_time_ns)
        self.assertEqual(report.moves[0].step, 1)
        payload = report.to_dict()
        self.assertEqual(payload["moves"][0]["action_name"], "U_CLOCKWISE")
        self.assertEqual(len(payload["states"]), 2)
        json.loads(report.to_json())
        env.close()

    def test_unsolved_run_stops_at_step_limit(self):
        env, benchmark = self.make_benchmark(solve_on_action=False)
        report = benchmark.run(
            lambda observation: 0, depth=1, seed=7, max_agent_steps=3,
        )
        self.assertFalse(report.success)
        self.assertFalse(report.final_state_verified)
        self.assertEqual(report.agent_steps, 3)
        self.assertEqual(len(report.states), 4)
        env.close()

    def test_solved_initial_state_uses_zero_moves(self):
        env, benchmark = self.make_benchmark()
        report = benchmark.run(
            lambda observation: 0, depth=0, seed=7, max_agent_steps=3,
        )
        self.assertTrue(report.success)
        self.assertEqual(report.agent_steps, 0)
        self.assertEqual(report.states, (SOLVED,))
        self.assertEqual(report.mean_policy_time_ns, 0.0)
        env.close()

    def test_rejects_invalid_arguments_and_actions(self):
        env, benchmark = self.make_benchmark()
        with self.assertRaises(ValueError):
            benchmark.run(lambda observation: 0, depth=-1, seed=1, max_agent_steps=3)
        with self.assertRaises(TypeError):
            benchmark.run(lambda observation: 0, depth=1, seed=1, max_agent_steps=True)
        with self.assertRaises(TypeError):
            benchmark.run(lambda observation: True, depth=1, seed=1, max_agent_steps=3)
        env.close()

    def test_recorder_rejects_invalid_action(self):
        recorder = SolveRecorder(SOLVED)
        with self.assertRaises(ValueError):
            recorder.record(
                action=18, policy_time_ns=0, transition_time_ns=0,
                elapsed_time_ns=0, observation=SOLVED,
            )


if __name__ == "__main__":
    unittest.main()
