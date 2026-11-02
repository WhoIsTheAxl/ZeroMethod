"""Construct the pinned upstream environment behind the project adapter.

The upstream dependency is imported lazily so importing ZeroMethod core
interfaces remains dependency-free.
"""

from importlib import import_module
from operator import index
from typing import Optional

from .adapter import GymRubiksCubeAdapter


def make_environment(*, scramble_params: int = 0, max_steps: Optional[int] = None) -> GymRubiksCubeAdapter:
    """Create a configured GymRubiksCube environment and wrap it.

    scramble_params=0 creates a solved starting state. Randomized starts
    should currently be produced by RandomWalkStateGenerator so randomness
    is explicit and reproducible rather than using NumPy global RNG state.
    """
    if isinstance(scramble_params, bool):
        raise TypeError("scramble_params must be an integer")
    try:
        scramble_params = index(scramble_params)
    except TypeError as exc:
        raise TypeError("scramble_params must be an integer") from exc
    if scramble_params < 0:
        raise ValueError("scramble_params must not be negative")

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
    raw_env.scramble_params = scramble_params
    raw_env.max_steps = max_steps
    return GymRubiksCubeAdapter(raw_env)
