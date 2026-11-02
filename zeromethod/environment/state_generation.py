"""Seeded random-walk state generation without exposing scramble history.

This is a reproducible curriculum/evaluation state source, not a uniform
sampler over all legal Rubik's Cube states. The transform callable is injected
so the module stays independent of Gym and NumPy.
"""

from hashlib import sha256
from random import Random
from typing import Callable, Iterable, Sequence, Tuple


def derive_split_seed(master_seed: int, split: str) -> int:
    """Derive a stable, split-specific 64-bit seed from a master seed."""
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")

    payload = "{}:{}".format(master_seed, split.strip().lower()).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


class RandomWalkStateGenerator:
    """Generate cube observations by applying seeded legal action walks.

    Only the resulting observation is returned. The action sequence remains
    internal to the generator and is never attached to the generated state.
    """

    def __init__(
        self,
        solved_state: Sequence[int],
        transform: Callable[[Sequence[int], int], Iterable[int]],
        action_count: int,
        inverse_actions: Sequence[int],
        seed: int,
    ) -> None:
        if isinstance(action_count, bool) or not isinstance(action_count, int):
            raise TypeError("action_count must be an integer")
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if len(inverse_actions) != action_count:
            raise ValueError("inverse_actions must define every action")
        if any(
            isinstance(action, bool)
            or not isinstance(action, int)
            or action < 0
            or action >= action_count
            for action in inverse_actions
        ):
            raise ValueError("inverse action indices must be valid actions")
        for action, inverse in enumerate(inverse_actions):
            if inverse_actions[inverse] != action:
                raise ValueError("inverse_actions must be an involution")

        self._solved_state = tuple(int(value) for value in solved_state)
        if len(self._solved_state) != 54:
            raise ValueError("solved_state must contain exactly 54 stickers")
        self._transform = transform
        self.action_count = action_count
        self._inverse_actions = tuple(inverse_actions)
        self.seed = seed
        self._rng = Random(seed)

    def generate(self, depth: int) -> Tuple[int, ...]:
        """Return one observation produced by exactly depth random moves."""
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise TypeError("depth must be an integer")
        if depth < 0:
            raise ValueError("depth must not be negative")

        state = self._solved_state
        previous_action = None

        for _ in range(depth):
            choices = list(range(self.action_count))
            if previous_action is not None:
                inverse = self._inverse_actions[previous_action]
                choices.remove(inverse)
            action = self._rng.choice(choices)
            state = tuple(int(value) for value in self._transform(state, action))
            previous_action = action

        if len(state) != 54:
            raise ValueError("transform returned an observation that is not 54 stickers")
        return state
