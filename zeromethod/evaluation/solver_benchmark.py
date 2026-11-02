"""Record and benchmark one policy-driven cube-solving attempt.

Scramble construction is measured separately from solving. The benchmark
cross-checks the upstream solved predicate against the observation, detects
repeated states, and keeps full replay data available without printing it by
default.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from operator import index
from pathlib import Path
from time import perf_counter_ns
from typing import Callable, Tuple

from zeromethod.environment.actions import ACTION_COUNT, Action
from zeromethod.environment.episode import initialize_episode
from zeromethod.environment.solved import is_solved_observation


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
    stop_reason: str
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
            "stop_reason": self.stop_reason,
            "moves": [move.to_dict() for move in self.moves],
            "states": [list(state) for state in self.states],
        }

    def to_compact_dict(self):
        """Return a terminal-friendly summary without 54-sticker state dumps."""
        return {
            "policy": self.policy,
            "scramble_depth": self.scramble_depth,
            "scramble_seed": self.scramble_seed,
            "success": self.success,
            "final_state_verified": self.final_state_verified,
            "agent_steps": self.agent_steps,
            "stop_reason": self.stop_reason,
            "solving_time_ns": self.solving_time_ns,
            "solving_time_seconds": self.solving_time_seconds,
            "setup_time_ns": self.setup_time_ns,
            "total_runtime_ns": self.total_runtime_ns,
            "total_policy_time_ns": self.total_policy_time_ns,
            "mean_policy_time_ns": self.mean_policy_time_ns,
            "total_transition_time_ns": self.total_transition_time_ns,
            "moves": [
                {
                    "step": move.step,
                    "action": move.action,
                    "action_name": move.action_name,
                    "policy_time_ns": move.policy_time_ns,
                    "transition_time_ns": move.transition_time_ns,
                }
                for move in self.moves
            ],
        }

    def to_json(self, *, indent=2, include_states=True):
        payload = self.to_dict() if include_states else self.to_compact_dict()
        return json.dumps(payload, indent=indent, sort_keys=True)


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

    def _verified_solved(self, observation, *, phase):
        observed = is_solved_observation(observation)
        reported = bool(self.env.is_solved())
        if observed != reported:
            raise RuntimeError(
                "solved-state disagreement {}: observation predicate is {}, "
                "environment predicate is {}".format(phase, observed, reported)
            )
        return observed

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

        success = self._verified_solved(observation, phase="after scramble")
        if depth == 1 and success:
            raise RuntimeError(
                "depth-1 scramble remained solved; environment action or "
                "scramble initialization is not being applied"
            )

        seen_states = {tuple(observation)}
        stop_reason = "solved" if success else "step_limit"

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
            if not 0 <= action < ACTION_COUNT:
                raise ValueError("policy action must be in [0, {})".format(ACTION_COUNT))

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

            success = self._verified_solved(
                observation, phase="after agent step {}".format(len(recorder.moves))
            )
            if success:
                stop_reason = "solved"
                break
            if result.terminated or result.truncated:
                stop_reason = "environment_terminated" if result.terminated else "environment_truncated"
                break
            state = tuple(observation)
            if state in seen_states:
                stop_reason = "cycle_detected"
                break
            seen_states.add(state)

        if not success and len(recorder.moves) >= max_agent_steps:
            stop_reason = "step_limit"

        solving_time_ns = perf_counter_ns() - solving_start
        total_runtime_ns = perf_counter_ns() - runtime_start
        final_state_verified = self._verified_solved(
            observation, phase="at benchmark completion"
        )
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
            stop_reason=stop_reason,
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
    parser.add_argument(
        "--full-report", action="store_true",
        help="print every recorded 54-sticker state (large output)",
    )
    parser.add_argument(
        "--output", default=None,
        help="optional path for the complete JSON report, including states",
    )
    args = parser.parse_args(argv)

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
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(report.to_json(), encoding="utf-8")
        print(report.to_json(include_states=args.full_report))
    finally:
        env.close()


if __name__ == "__main__":
    main()
