"""Run one bounded agent episode after a private seeded initialization."""
from dataclasses import dataclass
from operator import index
from typing import Callable, Tuple

from .adapter import GymRubiksCubeAdapter
from .episode import initialize_episode


@dataclass(frozen=True)
class EpisodeResult:
    """Public outcome of one agent episode; contains no scramble history."""

    observation: Tuple[int, ...]
    total_reward: float
    agent_steps: int
    terminated: bool
    truncated: bool
    success: bool


class EpisodeRunner:
    """Execute a callable policy with a step budget that excludes scrambling.

    A policy is a callable receiving the current immutable observation and
    returning one discrete action index. The runner never passes the
    initialization action sequence to the policy.
    """

    def __init__(self, env: GymRubiksCubeAdapter):
        self.env = env

    @staticmethod
    def _validate_max_agent_steps(max_agent_steps: int) -> int:
        if isinstance(max_agent_steps, bool):
            raise TypeError("max_agent_steps must be a non-negative integer")
        try:
            value = index(max_agent_steps)
        except TypeError as exc:
            raise TypeError(
                "max_agent_steps must be a non-negative integer"
            ) from exc
        if value < 0:
            raise ValueError("max_agent_steps must be non-negative")
        return value

    def run(
        self,
        policy: Callable[[Tuple[int, ...]], int],
        *,
        depth: int,
        seed: int,
        max_agent_steps: int,
    ) -> EpisodeResult:
        """Initialize and run a single episode, then return its summary."""
        if not callable(policy):
            raise TypeError("policy must be callable")
        step_limit = self._validate_max_agent_steps(max_agent_steps)

        observation, _ = initialize_episode(
            self.env, depth=depth, seed=seed
        )
        total_reward = 0.0
        agent_steps = 0

        if self.env.is_solved():
            return EpisodeResult(
                observation, total_reward, agent_steps,
                True, False, True,
            )

        while agent_steps < step_limit:
            action = policy(observation)
            if isinstance(action, bool):
                raise TypeError("policy action must be an integer index")
            try:
                action = index(action)
            except TypeError as exc:
                raise TypeError(
                    "policy action must be an integer index"
                ) from exc

            result = self.env.step(action)
            observation = result.observation
            total_reward += result.reward
            agent_steps += 1

            if result.terminated or result.truncated:
                return EpisodeResult(
                    observation=observation,
                    total_reward=total_reward,
                    agent_steps=agent_steps,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    success=result.terminated,
                )

        return EpisodeResult(
            observation=observation,
            total_reward=total_reward,
            agent_steps=agent_steps,
            terminated=False,
            truncated=True,
            success=False,
        )
