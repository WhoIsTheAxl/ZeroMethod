"""Seeded episode initialization through real environment transitions."""

from operator import index
from random import Random

from .actions import ACTION_COUNT, INVERSE_ACTIONS
from .adapter import GymRubiksCubeAdapter
from .state_generation import (
    _draw_random_walk_actions,
    _is_solved_observation,
)


def initialize_episode(env: GymRubiksCubeAdapter, *, depth: int, seed: int):
    """Reset and replay a seeded scramble; return only the final observation.

    GymRubiksCube has no supported arbitrary-state injection API. Starting
    states are therefore constructed only through reset() and normal step()
    calls. If a candidate walk reaches solved before its final requested move,
    the partial attempt is discarded and a new deterministic candidate is
    replayed from reset. The action history is never returned.

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
    if env.action_count != ACTION_COUNT:
        raise ValueError(
            "episode initialization requires exactly {} actions".format(ACTION_COUNT)
        )
    if env.max_steps is not None:
        raise ValueError("initialize_episode requires max_steps=None")

    rng = Random(seed)
    for _ in range(10_000):
        actions = _draw_random_walk_actions(
            rng, depth, ACTION_COUNT, INVERSE_ACTIONS
        )
        observation, _ = env.reset()
        if not env.is_solved() or not _is_solved_observation(observation):
            raise RuntimeError(
                "environment reset must produce a solved cube; disable implicit scrambling"
            )

        retry = False
        for position, action in enumerate(actions):
            result = env.step(action)
            observation = result.observation
            if result.truncated:
                raise RuntimeError(
                    "environment truncated during scramble; create it with max_steps=None"
                )
            if result.terminated and position < depth - 1:
                retry = True
                break

        if not retry:
            return observation, {}

    raise RuntimeError("could not initialize a non-terminal random walk prefix")
