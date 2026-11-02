"""Tests for deterministic, history-free state generation."""

import unittest

from zeromethod.environment.state_generation import (
    RandomWalkStateGenerator,
    derive_split_seed,
)


SOLVED = tuple(color for color in range(6) for _ in range(9))
INVERSES = tuple(action ^ 1 for action in range(18))


def fake_transform(state, action):
    # A reversible toy permutation; this tests the generator contract, not cube math.
    amount = 1 if action % 2 == 0 else -1
    amount %= len(state)
    return tuple(state[amount:]) + tuple(state[:amount])


def make_generator(seed):
    return RandomWalkStateGenerator(
        solved_state=SOLVED,
        transform=fake_transform,
        action_count=18,
        inverse_actions=INVERSES,
        seed=seed,
    )


class TestStateGeneration(unittest.TestCase):
    def test_same_seed_reproduces_same_state_sequence(self):
        left = make_generator(12345)
        right = make_generator(12345)

        self.assertEqual(
            [left.generate(30) for _ in range(5)],
            [right.generate(30) for _ in range(5)],
        )

    def test_different_split_seeds_are_independent(self):
        training = derive_split_seed(2026, "training")
        validation = derive_split_seed(2026, "validation")
        evaluation = derive_split_seed(2026, "evaluation")

        self.assertEqual(training, derive_split_seed(2026, "TRAINING"))
        self.assertEqual(len({training, validation, evaluation}), 3)

    def test_generate_returns_only_an_observation(self):
        observation = make_generator(7).generate(12)

        self.assertIsInstance(observation, tuple)
        self.assertEqual(len(observation), 54)
        self.assertEqual(
            [observation.count(color) for color in range(6)],
            [9] * 6,
        )
        self.assertFalse(hasattr(observation, "scramble"))
        self.assertFalse(hasattr(observation, "actions"))

    def test_zero_depth_returns_solved_state(self):
        self.assertEqual(make_generator(9).generate(0), SOLVED)

    def test_rejects_invalid_depth(self):
        generator = make_generator(1)
        for depth in (-1, 1.5, True):
            with self.subTest(depth=depth):
                with self.assertRaises((TypeError, ValueError)):
                    generator.generate(depth)

    def test_rejects_invalid_inverse_map(self):
        with self.assertRaises(ValueError):
            RandomWalkStateGenerator(
                SOLVED, fake_transform, 18, tuple(range(18)), 1
            )


if __name__ == "__main__":
    unittest.main()
