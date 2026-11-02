"""Uniform-random policy baseline and reproducible evaluation report."""
import argparse
from dataclasses import asdict, dataclass
import json
from random import Random
from time import perf_counter
from typing import Tuple

from zeromethod.environment.actions import ACTION_COUNT
from zeromethod.environment.factory import make_environment
from zeromethod.environment.runner import EpisodeResult, EpisodeRunner
from zeromethod.environment.state_generation import derive_split_seed


class RandomPolicy:
    """Choose uniformly from all 18 actions; inverse moves are not filtered."""

    def __init__(self, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self._rng = Random(seed)
        self.policy_calls = 0
        self.policy_seconds = 0.0

    def __call__(self, observation: Tuple[int, ...]) -> int:
        start = perf_counter()
        action = self._rng.randrange(ACTION_COUNT)
        self.policy_seconds += perf_counter() - start
        self.policy_calls += 1
        return action


@dataclass(frozen=True)
class BaselineReport:
    """Aggregate metrics plus per-episode outcomes for one experiment."""

    policy: str
    master_seed: int
    episodes: int
    scramble_depth: int
    max_agent_steps: int
    successes: int
    failures: int
    truncations: int
    success_rate: float
    mean_agent_steps: float
    mean_success_steps: float
    mean_reward: float
    policy_calls: int
    total_policy_seconds: float
    mean_policy_latency_seconds: float
    elapsed_seconds: float
    episode_results: Tuple[EpisodeResult, ...]

    def to_dict(self):
        """Return a JSON-serializable representation."""
        data = asdict(self)
        # Keep reports compact; final sticker arrays are available in the
        # in-memory EpisodeResult objects but are not benchmark metrics.
        data["episode_results"] = [
            {
                "total_reward": result.total_reward,
                "agent_steps": result.agent_steps,
                "terminated": result.terminated,
                "truncated": result.truncated,
                "success": result.success,
            }
            for result in self.episode_results
        ]
        return data


def evaluate_random_baseline(
    env,
    *,
    episodes: int,
    scramble_depth: int,
    max_agent_steps: int,
    master_seed: int,
) -> BaselineReport:
    """Evaluate independent seeded uniform-random policies on seeded starts.

    Each episode receives separate deterministic seeds for its scramble and
    policy. The environment is reused and reset by EpisodeRunner. Scramble
    depth must be positive so the benchmark never counts solved resets as
    random-policy successes.
    """
    for name, value, minimum in (
        ("episodes", episodes, 1),
        ("scramble_depth", scramble_depth, 1),
        ("max_agent_steps", max_agent_steps, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("{} must be an integer".format(name))
        if value < minimum:
            raise ValueError("{} must be at least {}".format(name, minimum))
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")

    runner = EpisodeRunner(env)
    results = []
    total_policy_seconds = 0.0
    policy_calls = 0
    started = perf_counter()

    for episode_index in range(episodes):
        scramble_seed = derive_split_seed(
            master_seed, "random-baseline-scramble-{}".format(episode_index)
        )
        policy_seed = derive_split_seed(
            master_seed, "random-baseline-policy-{}".format(episode_index)
        )
        policy = RandomPolicy(policy_seed)
        result = runner.run(
            policy,
            depth=scramble_depth,
            seed=scramble_seed,
            max_agent_steps=max_agent_steps,
        )
        results.append(result)
        total_policy_seconds += policy.policy_seconds
        policy_calls += policy.policy_calls

    elapsed = perf_counter() - started
    successes = sum(result.success for result in results)
    failures = episodes - successes
    truncations = sum(result.truncated for result in results)
    success_steps = [result.agent_steps for result in results if result.success]

    return BaselineReport(
        policy="uniform_random_18_actions",
        master_seed=master_seed,
        episodes=episodes,
        scramble_depth=scramble_depth,
        max_agent_steps=max_agent_steps,
        successes=successes,
        failures=failures,
        truncations=truncations,
        success_rate=successes / episodes,
        mean_agent_steps=sum(r.agent_steps for r in results) / episodes,
        mean_success_steps=(
            sum(success_steps) / len(success_steps) if success_steps else 0.0
        ),
        mean_reward=sum(r.total_reward for r in results) / episodes,
        policy_calls=policy_calls,
        total_policy_seconds=total_policy_seconds,
        mean_policy_latency_seconds=(
            total_policy_seconds / policy_calls if policy_calls else 0.0
        ),
        elapsed_seconds=elapsed,
        episode_results=tuple(results),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--max-agent-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    env = make_environment(max_steps=None)
    try:
        report = evaluate_random_baseline(
            env,
            episodes=args.episodes,
            scramble_depth=args.depth,
            max_agent_steps=args.max_agent_steps,
            master_seed=args.seed,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
