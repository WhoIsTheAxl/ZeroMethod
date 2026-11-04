# ZeroMethod architecture

## Current state

ZeroMethod has environment and experiment infrastructure, including three reference policies. The DQN training implementation is being developed separately; this main-branch milestone does not add a trained agent or begin model training.

## Environment factory

make_environment(max_steps=None) lazily imports the pinned GymRubiksCube submodule, disables its implicit random scramble, verifies the 18-action contract, and wraps the raw environment. Core package imports remain dependency-free.

The adapter exposes the configured max_steps value without requiring episode code to reach through to the raw environment. Episode initialization requires this value to be None because upstream counts every step after reset, including scramble construction.

## Upstream state-injection assessment

The pinned submodule revision is `e6ac42990db6a9dcceebe2ee91627f70da71deac`. Its `RubiksCubeEnv.reset()` rebuilds the solved `color_vector`, initializes the transform and terminal flag, resets the step counter, and then applies the optional scramble. Its `step(action)` updates the state through the move transform and updates the terminal/reward fields.

That revision exposes no documented public `set_state`, `set_state_from_observation`, or equivalent arbitrary-state injection method. Although `color_vector` is an attribute, assigning it directly would bypass the environment's transition bookkeeping and could desynchronize terminal and rendering state. ZeroMethod therefore does not mutate it. The supported boundary is reset plus real step calls.

If direct state loading becomes necessary, the recommended next step is to request an upstream public state-loading API that validates the 54-sticker observation and synchronizes the cube state, terminal flag, step counter, and rendering representation. ZeroMethod can then wrap that supported API and add an upstream integration test. Until such an API exists, replaying legal actions is the safe boundary.

## Reproducible episode initialization

`make_environment(max_steps=None)` creates a solved-start environment by setting upstream `scramble_params=0`. `initialize_episode(env, depth, seed)` then:

1. Validates the requested depth, seed, 18-action contract, and unbounded upstream step configuration.
2. Draws a deterministic random walk from the seed, excluding an action's immediate inverse.
3. Resets the environment and verifies that reset produced a solved observation.
4. Applies each scramble action through the adapter's normal `step()` interface.
5. Returns only the final immutable 54-sticker observation and an empty info mapping.

The action sampler is shared with `RandomWalkStateGenerator`, so the offline observation generator and live episode initializer use the same seeded action sequence. The action sequence remains local and is never attached to the observation or passed to a policy.

The upstream environment terminates as soon as a move solves the cube. If a candidate walk reaches solved before its final requested action, the initializer discards that partial attempt, resets, and draws another candidate from the same deterministic random stream. It never calls `step()` after termination. A solved state reached exactly on the final requested action is a valid depth-exact starting state.

Depth is the number of successfully replayed actions, not shortest-path distance. Random walks are not uniform samples of the cube state space and can revisit states. The observation validator checks sticker shape, values, and counts; reachability provenance comes from actual upstream transitions.

## Episode runner

EpisodeRunner receives an adapter and a callable policy. Its run method initializes the seeded starting state, then gives the policy only the current observation. It enforces max_agent_steps independently of scramble construction and returns an immutable EpisodeResult containing final observation, cumulative reward, agent action count, termination/truncation flags, and success. A solved initial state ends at zero agent actions. The runner does not close the environment; callers control environment lifetime.

The runner treats the agent-step budget as a hard cap: reaching it without solving produces truncation. An upstream truncation is preserved. Policy actions are checked as integer indices and then validated by the adapter.

## Random-policy baselines

`zeromethod.experiments.random_baseline` provides v1: a reproducible uniform-random policy that samples all 18 actions, including immediate inverses.

`zeromethod.experiments.random_baseline_v2` provides v2: the same random strategy, except the immediately preceding action's inverse is excluded. The first choice has 18 possibilities; each later choice has 17. This is a one-step constraint only: it does not inspect the cube state, avoid longer cycles, or plan toward a solution.

Both experiments share the same evaluator and report schema. Each episode gets independent deterministic scramble and policy seeds derived from a master seed. Reports include solve rate, failure/truncation counts, mean agent steps, mean successful solve steps, reward, policy-call count, measured policy latency, and total elapsed time. Timing values are observational and not expected to be bit-for-bit reproducible.

Run v1 with `python -m zeromethod.experiments.random_baseline` and v2 with `python -m zeromethod.experiments.random_baseline_v2`. Both accept `--episodes`, `--depth`, `--max-agent-steps`, and `--seed`.

## Adapter and observation contract

GymRubiksCubeAdapter normalizes reset to (observation, info), step to StepResult, and close to None. Observations are immutable tuples of 54 integer stickers, values 0..5, exactly nine of each color.

validate_observation checks data integrity, not reachability. validate_cube_state checks conventional fixed-center cubie invariants and is not valid as a gate for all 18 actions because E/S/M turns relocate centers. Full-action reachability provenance comes from real upstream transitions.

## State generation

RandomWalkStateGenerator is an offline observation-only seeded generator using an injected transform. Its private action sampler is the same one used by initialize_episode. Both exclude immediate inverse actions and reject candidate walks that become solved before the final requested action, matching upstream terminal semantics. Neither guarantees uniform sampling or shortest-path depth. derive_split_seed provides stable split-specific seeds.

## Action-space contract

The project-owned Action enum preserves all 18 upstream indices. The provisional environment action count (EAC) metric assigns unit cost to every action; it is not HTM or QTM.

## Dependency boundaries

The adapter, action interface, state generator, episode initializer, and validators use only the Python standard library. Upstream integration tests require the pinned legacy dependencies.

## Next milestones

1. Compare both random baselines across multiple seeds and scramble depths.
2. Finalize the minimal agent interface and training loop on a separate experiment branch.
3. Investigate learning algorithms and evaluate generalization.
