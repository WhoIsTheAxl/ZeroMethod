"""Normalize the legacy GymRubiksCube API behind a small stable interface.

This module intentionally uses only the Python standard library. The wrapped
environment is injected by the caller, so importing ZeroMethod does not import
Gym, NumPy, pygame, or the upstream submodule.
"""

from dataclasses import dataclass
from operator import index\nfrom typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class StepResult:
    """Result of one environment transition."""

    observation: Tuple[int, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


class GymRubiksCubeAdapter:
    """Adapt GymRubiksCube's legacy reset/step API.

    The upstream environment returns a raw observation from reset() and
    (observation, reward, done, info) from step(). This adapter exposes
    reset() -> (observation, info) and step() -> StepResult.

    A solved state is treated as termination. An upstream done signal without
    a solved state is treated as truncation (for example, max_steps reached).
    """

    OBSERVATION_SIZE = 54
    COLOR_COUNT = 6
    STICKERS_PER_COLOR = 9

    def __init__(self, env: Any) -> None:
        self.env = env
        self.action_count = int(env.action_space.n)
        if self.action_count <= 0:
            raise ValueError("environment must expose at least one action")

    @classmethod
    def _normalize_observation(cls, observation: Any) -> Tuple[int, ...]:
        try:
            values = tuple(index(value) for value in observation)
        except (TypeError, ValueError) as exc:
            raise ValueError("observation must be a one-dimensional sequence of integers") from exc

        if len(values) != cls.OBSERVATION_SIZE:
            raise ValueError(
                "expected an observation with {} stickers, got {}".format(
                    cls.OBSERVATION_SIZE, len(values)
                )
            )
        if any(value < 0 or value >= cls.COLOR_COUNT for value in values):
            raise ValueError("sticker colors must be integers from 0 through 5")

        counts = [values.count(color) for color in range(cls.COLOR_COUNT)]
        if counts != [cls.STICKERS_PER_COLOR] * cls.COLOR_COUNT:
            raise ValueError("each color must occur exactly nine times")

        return values

    def reset(self):
        """Reset the wrapped environment and return (observation, info)."""
        observation = self.env.reset()
        return self._normalize_observation(observation), {}

    def step(self, action: int) -> StepResult:
        """Apply one action using a normalized terminated/truncated result."""
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("action must be an integer")
        if action < 0 or action >= self.action_count:
            raise ValueError(
                "action {} is outside [0, {})".format(action, self.action_count)
            )

        observation, reward, done, info = self.env.step(action)
        normalized = self._normalize_observation(observation)
        solved = bool(self.env.isSolved())
        done = bool(done)

        return StepResult(
            observation=normalized,
            reward=float(reward),
            terminated=solved,
            truncated=done and not solved,
            info=dict(info or {}),
        )

    def close(self) -> None:
        """Close the wrapped environment when it supports close()."""
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
