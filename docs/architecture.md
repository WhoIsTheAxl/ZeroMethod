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

This is a fixed-frame physical-state validator, not a solver and not a uniform state sampler. The upstream integration test runs a long random action walk, including middle-slice moves, through the validator.

## State generation boundary

`RandomWalkStateGenerator` accepts a solved observation, an injected state-transform function, the action count, an explicit inverse-action mapping, and a seed. It returns only the resulting immutable 54-sticker observation; it does not return or attach the scramble sequence.

`derive_split_seed(master_seed, split)` derives stable split-specific seeds so training, validation, and evaluation generators can be initialized independently and reproduced later.

The generator prevents an action from being followed immediately by its declared inverse. This only removes trivial adjacent cancellations. It does not guarantee shortest-path depth or uniform sampling.

## Action-space audit

The upstream action mapping is documented in [action_space.md](action_space.md). It exposes 18 quarter-turn actions: six outer faces and three middle slices, each with two directions. The source's face frame and color mapping are recorded there, along with inverse pairs and transition invariants.

For internal experiments, the provisional **environment action count (EAC)** metric assigns cost 1 to every action, including middle-slice turns. EAC is not HTM or QTM. Public benchmarks must state the exact action set and metric.

The upstream direction labels for actions 12–17 are "downwards" / "upwards". Do not translate these into Singmaster clockwise notation in code until the geometric viewing convention has been independently verified.

## Dependency boundaries

The adapter, state generator, legality validator, and their unit tests use only the Python standard library. The upstream integration tests require the legacy environment's pinned stack. This lets contributors test project-owned interface, state-generation, and cubie-invariant logic without installing the full upstream dependency set.

## Next architectural milestones

1. Independently audit the upstream action definitions and their intended notation.
2. Verify the state generator against the independently validated transition model.
3. Decide and document the canonical action set and move metric.
4. Add an environment factory that clearly owns upstream dependency imports.
5. Add an experiment interface before selecting an RL algorithm.
