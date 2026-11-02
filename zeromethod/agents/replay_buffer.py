"""Small dependency-free experience replay buffer for off-policy learning."""

from collections import deque
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Transition:
    state: tuple
    action: int
    reward: float
    next_state: tuple
    done: bool


class ReplayBuffer:
    """Bounded FIFO replay buffer with an independent deterministic sampler."""

    def __init__(self, capacity, seed):
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.capacity = capacity
        self._items = deque(maxlen=capacity)
        self._rng = Random(seed)

    def __len__(self):
        return len(self._items)

    def add(self, state, action, reward, next_state, done):
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("action must be an integer")
        if not 0 <= action < 18:
            raise ValueError("action must be in [0, 18)")
        if isinstance(done, bool) is False:
            raise TypeError("done must be a boolean")
        self._items.append(Transition(
            tuple(state), action, float(reward), tuple(next_state), done
        ))

    def sample(self, batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._items):
            raise ValueError("batch_size cannot exceed the current buffer size")
        return tuple(self._rng.sample(tuple(self._items), batch_size))
