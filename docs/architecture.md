# ZeroMethod architecture

## Current state

ZeroMethod includes the validated environment layer, reference policies, a first experimental DQN training pipeline, and detailed solve benchmarking. The DQN has shown promising depth-1 results but is not yet a general-purpose cube solver.

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


## First learned agent: DQN

The initial trainable agent is a sparse-reward Deep Q-Network. Its observation encoder preserves upstream sticker order and expands each of the 54 color values into six one-hot values (324 input features). A two-hidden-layer MLP maps the encoded observation to 18 action values.

The agent uses an online Q-network, a periodically synchronized target network, a bounded replay buffer, epsilon-greedy exploration, Huber loss, and gradient clipping. Training starts on one device (CUDA when available); multiple visible GPUs are not automatically used.

The trainer initializes each episode through `initialize_episode` and passes only observations, rewards, and transition boundaries to the agent. The scramble action sequence is never stored in replay or passed to the policy. Training and validation use separate deterministic seed namespaces.

The first reward function is deliberately unshaped: the upstream environment's +1 solved reward and 0 otherwise. Sticker-count shaping is excluded because it can optimize a proxy objective rather than solving the cube. The default curriculum increases random-walk depth from 1 to 3 every 2,000 training episodes. This is an experimental starting schedule, not a claim that DQN will solve arbitrary cube states.

Checkpoints save online/target weights, optimizer state, training step count, and experiment metadata. Replay contents and RNG state are not yet checkpointed, so the checkpoint is not an exact resumable training snapshot.


## Solve recording and benchmarking

`zeromethod.evaluation.solver_benchmark` provides `SolveRecorder`, `SolverBenchmark`, and `SolveReport`. A benchmark initializes a seeded scramble without exposing its action history to the policy, then records each selected action and resulting observation. The report includes the ordered move list, all intermediate states (initial state plus one per move), per-decision policy latency, environment transition latency, solving time, setup time, total runtime, reward, and environment-verified success.

Solving time begins after scramble setup and covers the policy decisions plus environment transitions. Policy latency is measured separately from transition latency. All durations use `time.perf_counter_ns`; they are machine- and runtime-dependent. Action names preserve the upstream `Action` enum semantics rather than assuming unverified Singmaster notation. The report can be serialized with `to_json()` for later analysis or replay.

Run one DQN checkpoint against a reproducible scramble with:

```bash
PYTHONPATH=third_party/GymRubiksCube python -m zeromethod.evaluation.solver_benchmark \\
  --checkpoint artifacts/dqn_curriculum_v1.pt \\
  --depth 3 --seed 42 --max-agent-steps 100 --device cuda:0
```

The environment is caller-owned and remains open only for the benchmark duration. The benchmark is an evaluation tool; it does not modify model weights or replay data.

## Next milestones

1. Evaluate the DQN on independent seeds and increasing scramble depths.
2. Improve checkpoint reproducibility and training-resume support.
3. Investigate stronger state representations and learning algorithms.
