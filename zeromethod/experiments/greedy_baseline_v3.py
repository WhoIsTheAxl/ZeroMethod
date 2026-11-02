"""Run the final greedy one-step-lookahead reference baseline."""
import argparse
import json

from zeromethod.environment.factory import make_environment
from zeromethod.experiments.random_baseline import _evaluate_policy_baseline
from zeromethod.experiments.greedy_policy_v3 import (
    GreedyOneStepPolicy,
)


def evaluate_greedy_baseline_v3(
    env, *, episodes: int, scramble_depth: int,
    max_agent_steps: int, master_seed: int,
):
    """Evaluate the state-aware greedy policy using the shared report schema."""
    return _evaluate_policy_baseline(
        env,
        policy_factory=lambda seed: GreedyOneStepPolicy(env, seed),
        policy_name=GreedyOneStepPolicy.name,
        episodes=episodes,
        scramble_depth=scramble_depth,
        max_agent_steps=max_agent_steps,
        master_seed=master_seed,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Greedy one-step-lookahead cube baseline (V3)."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--max-agent-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    env = make_environment(max_steps=None)
    try:
        report = evaluate_greedy_baseline_v3(
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
