# ZeroMethod architecture

## Current state

ZeroMethod is infrastructure-first. It does not yet contain a trained agent, a policy, a replay buffer, or a training loop.

The current code separates project-owned interfaces from the upstream GymRubiksCube implementation:

```text
Agent / training code (not implemented yet)
             |
             v
GymRubiksCubeAdapter
             |
             v
GymRubiksCube (Git submodule)
```

## Adapter contract

`zeromethod.environment.GymRubiksCubeAdapter` wraps an already-created environment object. It does not import GymRubiksCube or any numerical package at module import time.

- `reset() -> (observation, info)`
- `step(action) -> StepResult`
- `close() -> None`

The observation is normalized to an immutable tuple of 54 integer sticker colors. Each color must occur exactly nine times and be in the range 0..5. The adapter validates action type and bounds before delegating.

The upstream environment uses the legacy Gym API, where `done` combines successful termination and episode truncation. The adapter distinguishes these cases by checking the wrapped environment's solved-state predicate:

- solved state: `terminated=True`
- upstream done while unsolved: `truncated=True`

## Action-space warning

The upstream environment exposes 18 actions: clockwise and counter-clockwise versions of six outer layers and three middle slices. It is not equivalent to a conventional outer-face-only action set.

Do not compare raw action counts with standard HTM/QTM move counts without defining a conversion. Any benchmark must state its action set and move metric.

## Dependency boundaries

The adapter and its unit tests use only the Python standard library. The upstream integration tests require the legacy environment's pinned stack. This lets contributors test project-owned interface logic without installing the full upstream dependency set.

## Next architectural milestones

1. Decide and document the canonical cube action set.
2. Validate the upstream sticker permutations against independent cube-state invariants, not only sticker counts and inverse relationships.
3. Add an environment factory that clearly owns upstream dependency imports.
4. Add deterministic seed control and state generation.
5. Add an experiment interface before selecting an RL algorithm.
