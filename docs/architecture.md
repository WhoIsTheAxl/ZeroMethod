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

## State generation boundary

`RandomWalkStateGenerator` accepts a solved observation, an injected state-transform function, the action count, an explicit inverse-action mapping, and a seed. It returns only the resulting immutable 54-sticker observation; it does not return or attach the scramble sequence.

`derive_split_seed(master_seed, split)` derives stable split-specific seeds so training, validation, and evaluation generators can be initialized independently and reproduced later.

The generator prevents an action from being followed immediately by its declared inverse. This only removes trivial adjacent cancellations. It does not guarantee shortest-path depth, uniform sampling, or full cube-state legality. Those properties require separate validation.

## Action-space warning

The upstream environment exposes 18 actions: clockwise and counter-clockwise versions of six outer layers and three middle slices. It is not equivalent to a conventional outer-face-only action set.

Do not compare raw action counts with standard HTM/QTM move counts without defining a conversion. Any benchmark must state its action set and move metric. The exact action notation and a project-wide metric remain to be finalized after independent inspection of the upstream action definitions.

## Dependency boundaries

The adapter, state generator, and their unit tests use only the Python standard library. The upstream integration tests require the legacy environment's pinned stack. This lets contributors test project-owned interface and reproducibility logic without installing the full upstream dependency set.

## Next architectural milestones

1. Independently validate the upstream sticker permutations against corner/edge permutation and orientation invariants, not only sticker counts and inverse relationships.
2. Verify the state generator against the independently validated transition model.
3. Decide and document the canonical cube action set and move metric.
4. Add an environment factory that clearly owns upstream dependency imports.
5. Add an experiment interface before selecting an RL algorithm.
