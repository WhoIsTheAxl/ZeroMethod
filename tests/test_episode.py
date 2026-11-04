"""Tests for deterministic, history-free episode initialization."""

import unittest

from zeromethod.environment.adapter import GymRubiksCubeAdapter
from zeromethod.environment.episode import initialize_episode
from zeromethod.environment.state_generation import RandomWalkStateGenerator
from zeromethod.environment.actions import INVERSE_ACTIONS

SOLVED = tuple(color for color in range(6) for _ in range(9))


class ActionSpace:
    n = 18


def fake_transform(state, action):
    # A reversible toy permutation; this tests the generator contract, not cube math.
    amount = (1 if action % 2 == 0 else -1) % len(state)
    return tuple(state[amount:]) + tuple(state[:amount])


def face_blocks_are_solved(state):
    return all(
        len(set(state[offset:offset + 9])) == 1
        for offset in range(0, 54, 9)
    )


class FakeEnvironment:
    def __init__(self):
        self.action_space = ActionSpace()
        self.max_steps = None
        self.state = SOLVED
        self.done = False
        self.solved = True
        self.actions = []
        self.reset_count = 0
        self.total_step_calls = 0

    def reset(self):
        self.reset_count += 1
        self.state = SOLVED
        self.done = False
        self.solved = True
        self.actions = []
        return self.state

    def step(self, action):
        if self.done:
            raise AssertionError("step() must not be called after termination")
        self.total_step_calls += 1
        self.actions.append(action)
        self.state = fake_transform(self.state, action)
        self.solved = face_blocks_are_solved(self.state)
        self.done = self.solved
        return self.state, float(self.solved), self.done, {}

    def isSolved(self):
        return self.solved

    def close(self):
        pass


class EarlyTerminalEnvironment(FakeEnvironment):
    """Force the first candidate to terminate after its first action."""

    def step(self, action):
        if self.done:
            raise AssertionError("step() must not be called after termination")
        self.total_step_calls += 1
        self.actions.append(action)
        if self.reset_count == 1:
            self.state = SOLVED
            self.solved = True
            self.done = True
            return self.state, 1.0, True, {}
        return self.step_without_count(action)

    def step_without_count(self, action):
        self.state = fake_transform(self.state, action)
        self.solved = face_blocks_are_solved(self.state)
        self.done = self.solved
        return self.state, float(self.solved), self.done, {}


def make_generator(seed):
    return RandomWalkStateGenerator(
        solved_state=SOLVED,
        transform=fake_transform,
        action_count=18,
        inverse_actions=INVERSE_ACTIONS,
        seed=seed,
    )


class TestEpisodeInitialization(unittest.TestCase):
    def make_env(self, raw_type=FakeEnvironment):
        raw = raw_type()
        return raw, GymRubiksCubeAdapter(raw)

    def test_same_seed_reproduces_actions_and_state(self):
        raw_a, env_a = self.make_env()
        raw_b, env_b = self.make_env()
        a, info_a = initialize_episode(env_a, depth=20, seed=123)
        b, info_b = initialize_episode(env_b, depth=20, seed=123)
        self.assertEqual(a, b)
        self.assertEqual(raw_a.actions, raw_b.actions)
        self.assertEqual(len(raw_a.actions), 20)
        for previous, current in zip(raw_a.actions, raw_a.actions[1:]):
            self.assertNotEqual(current, int(INVERSE_ACTIONS[previous]))
        self.assertEqual(info_a, {})
        self.assertEqual(info_b, {})

    def test_online_initialization_matches_offline_state_generation(self):
        seed = 731
        depth = 19
        expected = make_generator(seed).generate(depth)
        raw, env = self.make_env()
        actual, info = initialize_episode(env, depth=depth, seed=seed)
        self.assertEqual(actual, expected)
        self.assertEqual(info, {})
        self.assertEqual(len(raw.actions), depth)
        self.assertEqual([actual.count(color) for color in range(6)], [9] * 6)

    def test_different_seed_changes_walk(self):
        raw_a, env_a = self.make_env()
        raw_b, env_b = self.make_env()
        initialize_episode(env_a, depth=20, seed=1)
        initialize_episode(env_b, depth=20, seed=2)
        self.assertNotEqual(raw_a.actions, raw_b.actions)

    def test_reinitialization_resets_before_replaying_same_seed(self):
        raw, env = self.make_env()
        first, _ = initialize_episode(env, depth=12, seed=99)
        raw.state = fake_transform(raw.state, 0)
        raw.solved = face_blocks_are_solved(raw.state)
        raw.done = raw.solved
        second, info = initialize_episode(env, depth=12, seed=99)
        self.assertEqual(first, second)
        self.assertEqual(info, {})
        self.assertEqual(raw.reset_count, 2)
        self.assertEqual(len(raw.actions), 12)

    def test_zero_depth_starts_solved(self):
        raw, env = self.make_env()
        state, info = initialize_episode(env, depth=0, seed=7)
        self.assertEqual(state, SOLVED)
        self.assertEqual(info, {})
        self.assertEqual(raw.actions, [])

    def test_history_is_not_returned(self):
        _, env = self.make_env()
        state, info = initialize_episode(env, depth=8, seed=5)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 54)
        self.assertEqual(info, {})
        self.assertFalse(hasattr(state, "actions"))
        self.assertFalse(hasattr(info, "actions"))

    def test_retries_if_candidate_solves_before_requested_final_move(self):
        raw, env = self.make_env(EarlyTerminalEnvironment)
        state, info = initialize_episode(env, depth=4, seed=3)
        self.assertEqual(len(state), 54)
        self.assertEqual(info, {})
        self.assertEqual(raw.reset_count, 2)
        self.assertEqual(raw.total_step_calls, 5)
        self.assertEqual(len(raw.actions), 4)

    def test_requires_solved_reset_state(self):
        class UnsolvedResetEnvironment(FakeEnvironment):
            def reset(self):
                super().reset()
                self.state = fake_transform(self.state, 0)
                self.solved = False
                return self.state

        _, env = self.make_env(UnsolvedResetEnvironment)
        with self.assertRaisesRegex(RuntimeError, "reset must produce a solved cube"):
            initialize_episode(env, depth=3, seed=1)

    def test_requires_no_upstream_step_limit(self):
        raw, env = self.make_env()
        raw.max_steps = 100
        with self.assertRaisesRegex(ValueError, "max_steps=None"):
            initialize_episode(env, depth=5, seed=1)

    def test_requires_all_eighteen_actions(self):
        class ReducedActionSpace:
            n = 17

        raw, _ = self.make_env()
        raw.action_space = ReducedActionSpace()
        env = GymRubiksCubeAdapter(raw)
        with self.assertRaisesRegex(ValueError, "exactly 18"):
            initialize_episode(env, depth=1, seed=1)

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
