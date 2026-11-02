"""Final non-learning baseline: greedy one-step lookahead over cube actions."""
from random import Random
from time import perf_counter
from typing import Tuple

from zeromethod.environment.actions import ACTION_COUNT, inverse_action
from zeromethod.environment.state_legality import validate_observation

SOLVED_STATE = tuple(color for color in range(6) for _ in range(9))


def count_correct_stickers(observation: Tuple[int, ...]) -> int:
    """Count facelets matching their solved-position color (0..54)."""
    validate_observation(observation)
    return sum(
        actual == expected
        for actual, expected in zip(observation, SOLVED_STATE)
    )


class GreedyOneStepPolicy:
    """Probe each legal action, score its successor, then choose a maximum.

    Probes use only the public adapter transition API. Each probe is undone
    with its verified inverse action, and the resulting observation must
    exactly match the input before the next candidate is evaluated.

    This is a model-based reference policy, not a learned policy. It performs
    36 environment transitions per decision: 18 forward probes and 18
    reversals. The episode runner then executes the selected action normally.
    """

    name = "greedy_one_step_correct_stickers"

    def __init__(self, env, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if getattr(env, "action_count", None) != ACTION_COUNT:
            raise ValueError("environment must expose exactly 18 actions")
        self.env = env
        self._rng = Random(seed)
        self.policy_calls = 0
        self.policy_seconds = 0.0

    def __call__(self, observation: Tuple[int, ...]) -> int:
        started = perf_counter()
        observation = tuple(observation)
        validate_observation(observation)

        scores = []
        for action in range(ACTION_COUNT):
            successor = self.env.step(action).observation
            scores.append(count_correct_stickers(successor))

            restored = self.env.step(int(inverse_action(action))).observation
            if restored != observation:
                raise RuntimeError(
                    "environment failed to restore the observation after "
                    "probing action {}".format(action)
                )

        best_score = max(scores)
        best_actions = [
            action for action, score in enumerate(scores)
            if score == best_score
        ]
        selected = self._rng.choice(best_actions)
        self.policy_calls += 1
        self.policy_seconds += perf_counter() - started
        return selected
