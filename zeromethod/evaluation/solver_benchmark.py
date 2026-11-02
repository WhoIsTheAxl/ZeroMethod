"""Record and benchmark one policy-driven cube-solving attempt.

Scramble construction is measured separately from solving. Per-decision policy
latency, environment transition latency, end-to-end solving time, all selected
actions, and every observed state are retained in a JSON-serializable report.
"""

from dataclasses import asdict, dataclass
import argparse
import json
from operator import index
from time import perf_counter_ns
from typing import Callable, Tuple

from zeromethod.environment.actions import ACTION_COUNT, Action
from zeromethod.environment.episode import initialize_episode


@dataclass(frozen=True)
class RecordedMove:
    """One action selected by the policy and the resulting environment state."""

    step: int
    action: int
    action_name: str
    policy_time_ns: int
    transition_time_ns: int
    elapsed_time_ns: int
    observation: Tuple[int, ...]

    def to_dict(self):
        data = asdict(self)
        data["observation"] = list(self.observation)
        return data


class SolveRecorder:
    """Collect an initial state and an ordered, replayable sequence of moves."""

    def __init__(self, initial_observation):
        self._states = [tuple(initial_observation)]
        self._moves = []

    @property
    def states(self):
        return tuple(self._states)

    @property
    def moves(self):
        return tuple(self._moves)

    def record(self, *, action, policy_time_ns, transition_time_ns,
               elapsed_time_ns, observation):
        if isinstance(action, bool):
            raise TypeError("action must be an integer action index")
        try:
            action = index(action)
        except TypeError as exc:
            raise TypeError("action must be an integer action index") from exc
        if not 0 <= action < ACTION_COUNT:
            raise ValueError("action index must be in [0, {})".format(ACTION_COUNT))
        state = tuple(observation)
        move = RecordedMove(
            step=len(self._moves) + 1,
            action=action,
            action_name=Action(action).name,
            policy_time_ns=int(policy_time_ns),
            transition_time_ns=int(transition_time_ns),
            elapsed_time_ns=int(elapsed_time_ns),
            observation=state,
        )
        self._moves.append(move)
        self._states.append(state)
        return move


@dataclass(frozen=True)
class SolveReport:
    """Detailed timing, outcome, move sequence, and state replay for one run."""

    policy: str
    scramble_depth: int
    scramble_seed: int
    max_agent_steps: int
    success: bool
    final_state_verified: bool
    agent_steps: int
    total_reward: float
    setup_time_ns: int
    solving_time_ns: int
    total_runtime_ns: int
    total_policy_time_ns: int
    total_transition_time_ns: int
    mean_policy_time_ns: float
    moves: Tuple[RecordedMove, ...]
    states: Tuple[Tuple[int, ...], ...]

    @property
    def solving_time_seconds(self):
        return self.solving_time_ns / 1_000_000_000

    @property
    def mean_policy_time_seconds(self):
        return self.mean_policy_time_ns / 1_000_000_000

    def to_dict(self):
        return {
            "policy": self.policy,
            "scramble_depth": self.scramble_depth,
            "scramble_seed": self.scramble_seed,
            "max_agent_steps": self.max_agent_steps,
            "success": self.success,
            "final_state_verified": self.final_state_verified,
            "agent_steps": self.agent_steps,
            "total_reward": self.total_reward,
            "setup_time_ns": self.setup_time_ns,
            "solving_time_ns": self.solving_time_ns,
            "solving_time_seconds": self.solving_time_seconds,
            "total_runtime_ns": self.total_runtime_ns,
            "total_policy_time_ns": self.total_policy_time_ns,
            "total_transition_time_ns": self.total_transition_time_ns,
            "mean_policy_time_ns": self.mean_policy_time_ns,
            "mean_policy_time_seconds": self.mean_policy_time_seconds,
            "moves": [move.to_dict() for move in self.moves],
            "states": [list(state) for state in self.states],
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class SolverBenchmark:
    """Run a policy on one seeded scramble and produce a detailed solve report.

    The policy receives only the current observation. Scramble actions and
    their history are never exposed. The environment remains caller-owned.
    """

    def __init__(self, env):
        self.env = env

    @staticmethod
    def _validate_non_negative_int(name, value):
        if isinstance(value, bool):
            raise TypeError("{} must be an integer".format(name))
        try:
            value = index(value)
        except TypeError as exc:
            raise TypeError("{} must be an integer".format(name)) from exc
        if value < 0:
            raise ValueError("{} must be non-negative".format(name))
        return value

    def run(self, policy: Callable[[Tuple[int, ...]], int], *, depth: int,
            seed: int, max_agent_steps: int, policy_name=None) -> SolveReport:
        if not callable(policy):
            raise TypeError("policy must be callable")
        depth = self._validate_non_negative_int("depth", depth)
        max_agent_steps = self._validate_non_negative_int(
            "max_agent_steps", max_agent_steps
        )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        runtime_start = perf_counter_ns()
        setup_start = runtime_start
        observation, _ = initialize_episode(self.env, depth=depth, seed=seed)
        setup_time_ns = perf_counter_ns() - setup_start
        recorder = SolveRecorder(observation)
        total_reward = 0.0
        total_policy_time_ns = 0
        total_transition_time_ns = 0
        solving_start = perf_counter_ns()
        success = bool(self.env.is_solved())

        while not success and len(recorder.moves) < max_agent_steps:
            policy_start = perf_counter_ns()
            action = policy(observation)
            policy_time_ns = perf_counter_ns() - policy_start
            if isinstance(action, bool):
                raise TypeError("policy action must be an integer action index")
            try:
                action = index(action)
            except TypeError as exc:
                raise TypeError(
                    "policy action must be an integer action index"
                ) from exc

            transition_start = perf_counter_ns()
            result = self.env.step(action)
            transition_time_ns = perf_counter_ns() - transition_start
            observation = result.observation
            total_reward += result.reward
            total_policy_time_ns += policy_time_ns
            total_transition_time_ns += transition_time_ns
            elapsed_time_ns = perf_counter_ns() - solving_start
            recorder.record(
                action=action,
                policy_time_ns=policy_time_ns,
                transition_time_ns=transition_time_ns,
                elapsed_time_ns=elapsed_time_ns,
                observation=observation,
            )
            success = bool(self.env.is_solved())
            if result.terminated or result.truncated:
                break

        solving_time_ns = (
            perf_counter_ns() - solving_start if recorder.moves else 0
        )
        total_runtime_ns = perf_counter_ns() - runtime_start
        # Verify against the environment, not just the reward or policy output.
        final_state_verified = bool(self.env.is_solved())
        success = success and final_state_verified
        moves = recorder.moves
        return SolveReport(
            policy=(policy_name or getattr(policy, "name", policy.__class__.__name__)),
            scramble_depth=depth,
            scramble_seed=seed,
            max_agent_steps=max_agent_steps,
            success=success,
            final_state_verified=final_state_verified,
            agent_steps=len(moves),
            total_reward=total_reward,
            setup_time_ns=setup_time_ns,
            solving_time_ns=solving_time_ns,
            total_runtime_ns=total_runtime_ns,
            total_policy_time_ns=total_policy_time_ns,
            total_transition_time_ns=total_transition_time_ns,
            mean_policy_time_ns=(
                total_policy_time_ns / len(moves) if moves else 0.0
            ),
            moves=moves,
            states=recorder.states,
        )


def main(argv=None):
    """Load a DQN checkpoint and benchmark one reproducible solve attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-agent-steps", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    # Keep PyTorch optional for users who only need the recorder/report types.
    import torch
    from zeromethod.agents.dqn import DQNAgent
    from zeromethod.environment.factory import make_environment

    agent = DQNAgent(seed=args.seed, device=args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=agent.device, weights_only=False
    )
    agent.online_network.load_state_dict(checkpoint["online_network"])
    agent.online_network.eval()

    env = make_environment(max_steps=None)
    try:
        report = SolverBenchmark(env).run(
            lambda observation: agent.act(observation, epsilon=0.0),
            depth=args.depth,
            seed=args.seed,
            max_agent_steps=args.max_agent_steps,
            policy_name="DQN",
        )
        print(report.to_json())
    finally:
        env.close()


if __name__ == "__main__":
    main()
