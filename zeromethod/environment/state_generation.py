"""Seeded random-walk state generation without exposing scramble history.

This is a reproducible curriculum/evaluation state source, not a uniform
sampler over all legal Rubik's Cube states. The transform callable is injected
so the module stays independent of Gym and NumPy.
"""

from hashlib import sha256
from operator import index
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


def _normalize_state(state: Iterable[int]) -> Tuple[int, ...]:
    try:
        values = tuple(index(value) for value in state)
    except (TypeError, ValueError) as exc:
        raise ValueError("state must be a sequence of integer sticker colors") from exc

    if len(values) != 54:
        raise ValueError("state must contain exactly 54 stickers")
    if any(value < 0 or value > 5 for value in values):
        raise ValueError("sticker colors must be in the range 0 through 5")
    if [values.count(color) for color in range(6)] != [9] * 6:
        raise ValueError("each sticker color must occur exactly nine times")
    return values


def _is_solved_observation(state: Sequence[int]) -> bool:
    """Match GymRubiksCube's solved predicate without importing NumPy."""
    return all(
        len(set(state[offset:offset + 9])) == 1
        for offset in range(0, 54, 9)
    )


def _draw_random_walk_actions(rng, depth, action_count, inverse_actions):
    """Draw one private walk, excluding immediate inverse pairs."""
    actions = []
    previous_action = None
    for _ in range(depth):
        choices = list(range(action_count))
        if previous_action is not None:
            choices.remove(int(inverse_actions[previous_action]))
        action = rng.choice(choices)
        actions.append(action)
        previous_action = action
    return tuple(actions)


class RandomWalkStateGenerator:
    """Generate cube observations by applying seeded legal action walks.

    Only the resulting observation is returned. The action sequence remains
    internal to the generator and is never attached to the generated state.
    Candidate walks that reach a solved state before their final action are
    discarded, matching the upstream environment's terminal-state behavior.
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
            if inverse == action or inverse_actions[inverse] != action:
                raise ValueError("inverse_actions must contain distinct inverse pairs")

        self._solved_state = _normalize_state(solved_state)
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

        # Upstream marks a solved state terminal. Reject a candidate if it
        # reaches solved before its final requested move; otherwise continuing
        # that walk would require stepping an already-terminated environment.
        for _ in range(10_000):
            actions = _draw_random_walk_actions(
                self._rng, depth, self.action_count, self._inverse_actions
            )
            state = self._solved_state
            retry = False
            for position, action in enumerate(actions):
                state = _normalize_state(self._transform(state, action))
                if position < depth - 1 and _is_solved_observation(state):
                    retry = True
                    break
            if not retry:
                return state

        raise RuntimeError("could not generate a non-terminal random walk prefix")
