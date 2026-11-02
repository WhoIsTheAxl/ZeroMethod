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
             |
             v
Independent cube-state legality validator
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

## Independent cube-state legality

`validate_cube_state(observation)` checks more than sticker counts. It verifies:

- exactly 54 integer stickers and nine of each color
- one of each center color, in an arrangement equivalent to a rigid cube orientation
- all eight corner cubies appear exactly once
- corner twist sum is divisible by three
- all twelve edge cubies appear exactly once
- edge flip sum is even
- corner and edge permutation parity match

The upstream environment allows middle-slice turns, which move center stickers. Therefore the validator first canonicalizes the observed center orientation using one of the 24 proper cube rotations, then checks the fixed-center cubie invariants. It rejects center arrangements that cannot be explained by a rigid cube rotation.

This is a fixed-frame physical-state validator, not a solver and not a uniform state sampler. **Known blocker:** the current 1,000-action upstream walk exposes that the validator's center-frame normalization is insufficient for states reached through the upstream middle-slice actions. CI currently fails on reachable states late in that walk. Do not use this validator as a dataset gate until its invariants are reconciled with the environment's movable-center representation.

## State generation boundary

`RandomWalkStateGenerator` accepts a solved observation, an injected state-transform function, the action count, an explicit inverse-action mapping, and a seed. It returns only the resulting immutable 54-sticker observation; it does not return or attach the scramble sequence.

`derive_split_seed(master_seed, split)` derives stable split-specific seeds so training, validation, and evaluation generators can be initialized independently and reproduced later.

The generator prevents an action from being followed immediately by its declared inverse. This only removes trivial adjacent cancellations. It does not guarantee shortest-path depth or uniform sampling.

## Action-space audit

The upstream action mapping is documented in [action_space.md](action_space.md). It exposes 18 quarter-turn actions: six outer faces and three middle slices, each with two directions. The source's face frame and color mapping are recorded there, along with inverse pairs and transition invariants.

For internal experiments, the provisional **environment action count (EAC)** metric assigns cost 1 to every action, including middle-slice turns. EAC is not HTM or QTM. Public benchmarks must state the exact action set and metric.

ZeroMethod now exposes a project-owned `Action` enum with stable values matching all 18 upstream indices, plus `ACTION_COUNT`, `INVERSE_ACTIONS`, and `inverse_action()`. The vertical action names preserve the upstream "down" / "up" wording rather than inventing unverified Singmaster notation. The 18-action interface is the selected project contract for the current environment.

## Dependency boundaries

The adapter, action interface, state generator, legality validator, and their unit tests use only the Python standard library. The upstream integration tests require the legacy environment's pinned stack. The core action-interface tests pass in CI; the upstream random-walk legality integration currently fails and is a blocker before dataset generation or training.

## Next architectural milestones

1. Add an environment factory that clearly owns upstream dependency imports.
2. Add a transition-model consistency test connecting the project-owned action enum to every upstream action permutation.
3. Define episode limits and a reproducible curriculum/evaluation runner.
4. Add an experiment interface and baseline throughput measurements before selecting an RL algorithm.
