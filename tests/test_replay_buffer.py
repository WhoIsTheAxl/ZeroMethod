"""Tests for the dependency-free replay buffer."""
import unittest

from zeromethod.agents.replay_buffer import ReplayBuffer, Transition

STATE = tuple([0] * 54)
NEXT = tuple([1] * 9 + [0] * 45)


class TestReplayBuffer(unittest.TestCase):
    def test_add_and_sample_are_reproducible(self):
        first = ReplayBuffer(10, seed=7)
        second = ReplayBuffer(10, seed=7)
        for buffer in (first, second):
            for action in range(5):
                buffer.add(STATE, action, 0.0, NEXT, False)
        self.assertEqual(first.sample(3), second.sample(3))
        self.assertEqual(len(first), 5)
        self.assertIsInstance(first.sample(1)[0], Transition)

    def test_capacity_is_bounded(self):
        buffer = ReplayBuffer(2, seed=1)
        for action in range(4):
            buffer.add(STATE, action, 0.0, NEXT, False)
        self.assertEqual(len(buffer), 2)
        self.assertEqual([item.action for item in buffer._items], [2, 3])

    def test_rejects_invalid_samples_and_actions(self):
        buffer = ReplayBuffer(2, seed=1)
        with self.assertRaises(ValueError):
            buffer.sample(1)
        with self.assertRaises(ValueError):
            buffer.add(STATE, 18, 0.0, NEXT, False)
        with self.assertRaises(TypeError):
            buffer.add(STATE, 1, 0.0, NEXT, 0)


if __name__ == "__main__":
    unittest.main()
