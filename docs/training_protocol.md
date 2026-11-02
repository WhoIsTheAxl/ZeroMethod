# Training and evaluation protocol

## Agent information boundary

The agent receives only its current observation, legal action set, reward, and episode boundary. It must not receive scramble actions, reverse action lists, precomputed solutions, or human solving demonstrations.

initialize_episode(env, depth, seed) constructs a starting state by replaying a seeded random walk through real environment transitions. It returns only the final observation and an empty info mapping. The action history is local to the initializer.

## Episode initialization constraints

The upstream environment has no public arbitrary-state injection API. ZeroMethod avoids direct mutation of its internal cube state. It resets to solved and executes generated actions through the adapter.

Create environments with make_environment(max_steps=None) when using initialize_episode. Upstream max_steps counts scramble-construction actions too. The episode runner enforces the agent's own step limit after initialization.

Depth is the number of actions replayed, not shortest-path distance. Immediate inverse pairs are excluded during scramble generation. For a fixed environment version, action mapping, depth, and seed reproduce the same walk. The info mapping intentionally contains no scramble metadata.

## State integrity and reachability

Observations must contain 54 integer stickers, values 0..5, and nine of each color. Reachability provenance comes from upstream transitions. The conventional fixed-center validator is not suitable for the full 18-action set because middle-slice turns relocate centers.

## Curriculum and reproducibility

A staged curriculum may increase random-walk depth. Derive distinct training, validation, and evaluation seeds with derive_split_seed. Record source commit, environment/dependency versions, seeds, split, depth schedule, action set, EAC metric, algorithm, hyperparameters, runtime, and metrics.

Random walks are not uniform samples of cube state space, and depth is not guaranteed optimal distance.

## Episode runner contract

Use EpisodeRunner(env).run(policy, depth=..., seed=..., max_agent_steps=...) for bounded episodes. The policy is a callable from the current observation to one action index. It is never called during scramble construction. The returned result records cumulative reward, agent-only step count, final observation, termination/truncation, and success. A zero-step budget is valid and immediately truncates an unsolved starting state. The runner does not own or close the environment.

## Random-policy baselines

Baseline v1:

```bash
python -m zeromethod.experiments.random_baseline --episodes 100 --depth 10 --max-agent-steps 100 --seed 42
```

V1 samples uniformly from all 18 actions, including immediate inverses.

Baseline v2:

```bash
python -m zeromethod.experiments.random_baseline_v2 --episodes 100 --depth 10 --max-agent-steps 100 --seed 42
```

V2 samples randomly but excludes the inverse of its immediately preceding action. It does not inspect the cube observation or plan ahead.

## Final baseline v3

```bash
python -m zeromethod.experiments.greedy_baseline_v3 --episodes 100 --depth 10 --max-agent-steps 100 --seed 42
```

V3 is state-aware and performs one-step lookahead. For each decision it applies each of the 18 actions through the public environment adapter, scores the resulting observation by counting correctly positioned stickers (0..54), and undoes the probe with the corresponding inverse action. It verifies that every probe restores the exact input observation. It then selects uniformly among the highest-scoring actions, using a seeded RNG for reproducible tie-breaking.

V3 is a model-based reference policy, not a learned agent. It performs 36 environment transitions per decision (18 probes and 18 reversals), plus the runner's actual selected transition. Probe rewards are intentionally ignored. The heuristic is not a distance metric and can become trapped in local optima or cycles. It is a modest state-aware reference, not a guaranteed solver. It uses the supplied observation and public transition API; it does not inspect private cube state or receive scramble history.

All evaluators use deterministic per-episode scramble and policy seeds and the same JSON report schema. Compare success rate, truncations, mean agent steps, mean successful solve steps, reward, policy-call count, policy latency, and wall-clock elapsed time. Timing is machine-dependent and should be reported separately from outcome metrics.

## Evaluation and metrics

Keep training, validation, and final evaluation states separate. Report solve success rate, inference latency, solution length under a declared metric, run-to-run consistency, and generalization across independent seeds.

## Current limitations

V1 and V2 are random reference policies; V3 is a one-step model-based heuristic. The first DQN training pipeline now exists and has produced a promising depth-1 pilot result, but no general-purpose solver has been demonstrated. The upstream reward is sparse (+1 solved, otherwise 0). Any reward shaping must be documented and checked for solution-information leakage.


## First DQN training protocol

The first learned agent is a standard sparse-reward DQN. The state encoder expands each sticker color into a six-element one-hot vector, producing 324 features. The MLP emits one Q-value for each of the 18 environment actions. The agent uses replay sampling, a target network, epsilon-greedy exploration, Huber loss, and gradient clipping.

Run from the repository root after installing the pinned upstream environment dependencies and PyTorch:

```bash
PYTHONPATH=third_party/GymRubiksCube python -m zeromethod.training.dqn \
  --episodes 10000 --start-depth 1 --max-depth 3 \
  --episodes-per-depth 2000 --max-agent-steps 20 \
  --evaluation-interval 500 --evaluation-episodes 100 \
  --seed 42 --checkpoint artifacts/dqn_v1.pt
```

The default curriculum increases random-walk depth from 1 to 3 in 2,000-episode stages. Epsilon decreases linearly from 1.0 to 0.05 over 50,000 agent transitions. Replay learning starts after 1,000 transitions; minibatches contain 64 transitions. The target network synchronizes every 1,000 optimizer updates. These are initial experimental hyperparameters, not tuned values.

The reward is the upstream sparse reward (+1 on solved, otherwise 0). No intermediate sticker-count reward is used. A transition is marked done when the episode terminates or reaches its configured agent-step boundary. Evaluation uses a distinct seed namespace and greedy actions; evaluation experiences are not added to replay.

The initial implementation trains on one GPU even when multiple GPUs are visible. The checkpoint stores model and optimizer state plus experiment metadata, but does not yet preserve replay-buffer contents or random-generator states; it is therefore a model artifact, not an exact training-resume checkpoint. Record the source commit, Kaggle runtime, PyTorch/CUDA versions, and full CLI arguments with benchmark results.


## Single-attempt solve benchmark and replay

Use `SolverBenchmark` when a detailed trace of one attempt is needed, rather than only aggregate success rates. It records the initial scrambled observation, every action selected by the policy, the resulting observation after each action, and nanosecond timing measurements.

For a trained DQN checkpoint:

```bash
PYTHONPATH=third_party/GymRubiksCube python -m zeromethod.evaluation.solver_benchmark \\
  --checkpoint artifacts/dqn_curriculum_v1.pt \\
  --depth 3 --seed 42 --max-agent-steps 100 --device cuda:0
```

The JSON report distinguishes setup time (scramble construction), solving time (policy decisions plus environment transitions), total runtime, policy inference time, and environment transition time. `moves` is the ordered action trace; `states` contains the initial observation followed by each post-move observation. The final success flag is checked against the environment's solved-state predicate. Action names use the project-owned upstream action identifiers; they are not presented as Singmaster notation because some upstream direction conventions have not been independently established.

For fair comparisons, hold scramble depth, seed, step budget, environment version, and hardware constant. A single solve's latency is descriptive; use many independent seeds and report distributions for performance claims.
