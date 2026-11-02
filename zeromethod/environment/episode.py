"""Seeded episode initialization through real environment transitions."""
from operator import index
from random import Random

from .actions import ACTION_COUNT, INVERSE_ACTIONS
from .adapter import GymRubiksCubeAdapter


def initialize_episode(env: GymRubiksCubeAdapter, *, depth: int, seed: int):
    """Reset and replay a seeded scramble; return only the final observation.

    Upstream max_steps must be disabled because its counter includes scramble
    construction. The training runner should enforce its limit afterward.
    """
    if isinstance(depth, bool):
        raise TypeError("depth must be an integer")
    try:
        depth = index(depth)
    except TypeError as exc:
        raise TypeError("depth must be an integer") from exc
    if depth < 0:
        raise ValueError("depth must not be negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if getattr(env.env, "max_steps", None) is not None:
        raise ValueError("initialize_episode requires max_steps=None")

    observation, _ = env.reset()
    rng = Random(seed)
    previous_action = None
    for _ in range(depth):
        choices = list(range(ACTION_COUNT))
        if previous_action is not None:
            choices.remove(INVERSE_ACTIONS[previous_action])
        action = rng.choice(choices)
        observation = env.step(action).observation
        previous_action = action

    return observation, {}
