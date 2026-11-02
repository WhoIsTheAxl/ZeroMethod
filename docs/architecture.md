# ZeroMethod architecture

## Current state

ZeroMethod is infrastructure-first. It does not yet contain a trained agent, policy, replay buffer, or training loop.

## Environment factory

make_environment(max_steps=None) lazily imports GymRubiksCube, disables its implicit random scramble, verifies the 18-action contract, and wraps the raw environment. Core package imports remain dependency-free.

## Episode initialization

The upstream environment has no public arbitrary-state injection API. ZeroMethod avoids assigning directly to upstream internal cube-state fields.

initialize_episode(env, depth, seed) resets the environment and replays a seeded random walk through the adapter's normal step interface. It returns only the final immutable observation and an empty info mapping; the action history stays local. The seed, depth, action mapping, and environment version reproduce the same walk.

The upstream max_steps counter includes actions used to construct the starting state. Therefore initialization requires an environment created with max_steps=None. The training runner must enforce its own agent-step limit after initialization. This avoids counting scramble construction against the agent's budget without mutating upstream internals.

Depth is action count, not shortest-path distance. Walks may revisit states or theoretically end solved.

## Adapter and observation contract

GymRubiksCubeAdapter normalizes reset to (observation, info), step to StepResult, and close to None. Observations are immutable tuples of 54 integer stickers, values 0..5, exactly nine of each color.

validate_observation checks data integrity, not reachability. validate_cube_state checks conventional fixed-center cubie invariants and is not valid as a gate for all 18 actions because E/S/M turns relocate centers. Full-action reachability provenance comes from real upstream transitions.

## State generation

RandomWalkStateGenerator is an offline observation-only seeded generator using an injected transform. initialize_episode is its online counterpart, replaying private actions through a live environment. Both avoid immediate inverse actions but do not guarantee uniform sampling or shortest-path depth. derive_split_seed provides stable split-specific seeds.

## Action-space contract

The project-owned Action enum preserves all 18 upstream indices. The provisional environment action count (EAC) metric assigns unit cost to every action; it is not HTM or QTM.

## Dependency boundaries

The adapter, action interface, state generator, episode initializer, and validators use only the Python standard library. Upstream integration tests require pinned legacy dependencies.

## Next milestones

1. Add a reproducible episode runner with agent-step limits that begin after initialization.
2. Measure environment throughput and establish a random-policy baseline.
3. Define the minimal agent interface and training loop.
