# ZeroMethod architecture

## Current state

ZeroMethod has environment and experiment infrastructure, including two random reference policies. It does not yet contain a trained agent, replay buffer, or training loop.

## Environment factory

make_environment(max_steps=None) lazily imports GymRubiksCube, disables its implicit random scramble, verifies the 18-action contract, and wraps the raw environment. Core package imports remain dependency-free.

## Episode initialization

The upstream environment has no public arbitrary-state injection API. ZeroMethod avoids assigning directly to upstream internal cube-state fields.

initialize_episode(env, depth, seed) resets the environment and replays a seeded random walk through the adapter's normal step interface. It returns only the final immutable observation and an empty info mapping; the action history stays local. The seed, depth, action mapping, and environment version reproduce the same walk.

The upstream max_steps counter includes actions used to construct the starting state. Therefore initialization requires an environment created with max_steps=None. The training runner must enforce its own agent-step limit after initialization. This avoids counting scramble construction against the agent's budget without mutating upstream internals.

Depth is action count, not shortest-path distance. Walks may revisit states or theoretically end solved.

## Episode runner

EpisodeRunner receives an adapter and a callable policy. Its run method initializes the seeded starting state, then gives the policy only the current observation. It enforces max_agent_steps independently of upstream scramble steps and returns an immutable EpisodeResult containing final observation, cumulative reward, agent action count, termination/truncation flags, and success. A solved initial state ends at zero agent actions. The runner does not close the environment; callers control environment lifetime.

The runner treats the agent-step budget as a hard cap: reaching it without solving produces truncation. An upstream truncation is preserved. Policy actions are checked as integer indices and then validated by the adapter.

## Random-policy baselines

`zeromethod.experiments.random_baseline` provides v1: a reproducible uniform-random policy that samples all 18 actions, including immediate inverses.

`zeromethod.experiments.random_baseline_v2` provides v2: the same random strategy, except the immediately preceding action's inverse is excluded. The first choice has 18 possibilities; each later choice has 17. This is a one-step constraint only: it does not inspect the cube state, avoid longer cycles, or plan toward a solution.

Both experiments share the same evaluator and report schema. Each episode gets independent deterministic scramble and policy seeds derived from a master seed. Reports include solve rate, failure/truncation counts, mean agent steps, mean successful solve steps, reward, policy-call count, measured policy latency, and total elapsed time. Timing values are observational and not expected to be bit-for-bit reproducible.

Run v1 with `python -m zeromethod.experiments.random_baseline` and v2 with `python -m zeromethod.experiments.random_baseline_v2`. Both accept `--episodes`, `--depth`, `--max-agent-steps`, and `--seed`.

## Random-policy baseline v2

`NoImmediateInverseRandomPolicy` samples uniformly from all 18 actions on its first decision. On each later decision it excludes the inverse of its immediately preceding action, leaving 17 choices. It does not inspect the cube observation, prevent longer cycles, or plan ahead. The v1 and v2 evaluators share the same report schema and episode evaluator.

Run v2 with `python -m zeromethod.experiments.random_baseline_v2`. It accepts the same `--episodes`, `--depth`, `--max-agent-steps`, and `--seed` arguments as v1.

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

1. Compare both random baselines across multiple seeds and scramble depths.
2. Define the minimal agent interface and training loop.
3. Investigate learning algorithms and evaluate generalization.
