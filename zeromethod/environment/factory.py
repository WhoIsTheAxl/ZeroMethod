"""Construct the pinned upstream environment behind the project adapter.

The upstream dependency is imported lazily so importing ZeroMethod core
interfaces remains dependency-free.
"""

from importlib import import_module
from operator import index
from typing import Optional

from .actions import ACTION_COUNT
from .adapter import GymRubiksCubeAdapter


def make_environment(*, max_steps: Optional[int] = None) -> GymRubiksCubeAdapter:
    """Create a solved-start GymRubiksCube environment and wrap it.

    The upstream environment's own random scramble is disabled. Randomized
    starting states must be generated through RandomWalkStateGenerator so
    their seeds and action histories remain under project control.
    """
    if max_steps is not None:
        if isinstance(max_steps, bool):
            raise TypeError("max_steps must be an integer or None")
        try:
            max_steps = index(max_steps)
        except TypeError as exc:
            raise TypeError("max_steps must be an integer or None") from exc
        if max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")

    try:
        module = import_module("gym_rubiks_cube.envs.rubiksCubeEnv")
    except ImportError as exc:
        raise ImportError(
            "GymRubiksCube is unavailable. Initialize the submodule and "
            "install its pinned environment dependencies before creating "
            "an environment."
        ) from exc

    raw_env = module.RubiksCubeEnv()
    raw_env.scramble_params = 0
    raw_env.max_steps = max_steps

    try:
        actual_action_count = raw_env.action_space.n
        if actual_action_count != ACTION_COUNT:
            raise RuntimeError(
                "GymRubiksCube exposes {} actions; ZeroMethod requires {}"
                .format(actual_action_count, ACTION_COUNT)
            )
        return GymRubiksCubeAdapter(raw_env)
    except Exception:
        close = getattr(raw_env, "close", None)
        if callable(close):
            close()
        raise
