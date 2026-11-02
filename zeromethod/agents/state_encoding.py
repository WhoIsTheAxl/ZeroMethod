"""Dependency-free observation encoding for neural agents."""

from zeromethod.environment.state_legality import validate_observation

OBSERVATION_SIZE = 54
COLOR_COUNT = 6
ENCODED_SIZE = OBSERVATION_SIZE * COLOR_COUNT


def encode_observation(observation):
    """Encode 54 sticker colors as a flat 324-value one-hot tuple.

    Sticker order is preserved exactly. The function accepts the same
    observation contract as the environment adapter and does not infer
    cubie coordinates or apply a human solving representation.
    """
    observation = tuple(observation)
    validate_observation(observation)
    values = []
    for color in observation:
        values.extend(1.0 if candidate == color else 0.0
                      for candidate in range(COLOR_COUNT))
    return tuple(values)


def encode_batch(observations):
    """Encode an iterable of observations as a tuple of feature tuples."""
    return tuple(encode_observation(observation) for observation in observations)
