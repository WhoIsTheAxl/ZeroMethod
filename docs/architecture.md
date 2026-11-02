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

## Observation integrity and cube-state legality

`validate_observation(observation)` is the integrity check for the full upstream 18-action environment. It verifies exactly 54 integer stickers, values in 0..5, and exactly nine occurrences of each color.

`validate_cube_state(observation)` is a stronger conventional fixed-center validator. It additionally checks corner/edge identity, orientation sums, and permutation parity after canonicalizing rigid whole-cube orientations. **It is not a reachability validator for the complete 18-action action set.**

The distinction matters because GymRubiksCube exposes E, S, and M middle-slice turns that physically relocate center stickers while leaving the corner layer in place. These transitions are valid in the upstream environment but take the state outside the fixed-center representation assumed by ordinary cubie invariants. A rigid orientation normalization cannot undo that relative center permutation. CI previously demonstrated this mismatch across a 1,000-action walk; this was a validator-scope bug, not evidence that the environment generated invalid observations.

For full-action training data, use the environment's own transition process as the reachability provenance and `validate_observation` as the data-integrity check. Do not describe the weaker observation check as proof of physical reachability. The stronger fixed-center validator remains useful for outer-face-only experiments and deliberately constructed invalid-state tests.

## State generation boundary

`RandomWalkStateGenerator` accepts a solved observation, an injected state-transform function, the action count, an explicit inverse-action mapping, and a seed. It returns only the resulting immutable 54-sticker observation; it does not return or attach the scramble sequence.

`derive_split_seed(master_seed, split)` derives stable split-specific seeds so training, validation, and evaluation generators can be initialized independently and reproduced later.

The generator prevents an action from being followed immediately by its declared inverse. This only removes trivial adjacent cancellations. It does not guarantee shortest-path depth or uniform sampling.

## Action-space audit

The upstream action mapping is documented in [action_space.md](action_space.md). It exposes 18 quarter-turn actions: six outer faces and three middle slices, each with two directions. The source's face frame and color mapping are recorded there, along with inverse pairs and transition invariants.

For internal experiments, the provisional **environment action count (EAC)** metric assigns cost 1 to every action, including middle-slice turns. EAC is not HTM or QTM. Public benchmarks must state the exact action set and metric.

ZeroMethod now exposes a project-owned `Action` enum with stable values matching all 18 upstream indices, plus `ACTION_COUNT`, `INVERSE_ACTIONS`, and `inverse_action()`. The vertical action names preserve the upstream "down" / "up" wording rather than inventing unverified Singmaster notation. The 18-action interface is the selected project contract for the current environment.

## Dependency boundaries

The adapter, action interface, state generator, observation validator, fixed-center cubie validator, and their unit tests use only the Python standard library. The upstream integration tests require the legacy environment's pinned stack. The upstream full-action walk checks observation integrity, while fixed-center cubie invariants are tested separately using outer-face actions.

## Next architectural milestones

1. Add an environment factory that clearly owns upstream dependency imports.
2. Add a transition-model consistency test connecting the project-owned action enum to every upstream action permutation.
3. Define episode limits and a reproducible curriculum/evaluation runner.
4. Add an experiment interface and baseline throughput measurements before selecting an RL algorithm.
