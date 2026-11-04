# Training and evaluation protocol

## Agent information boundary

The agent receives only its current observation, legal action set, reward, and episode boundary. It must not receive scramble actions, reverse action lists, precomputed solutions, or human solving demonstrations.

`initialize_episode(env, depth, seed)` constructs a starting state by replaying a seeded random walk through real environment transitions. It returns only the final observation and an empty info mapping. The action history is local to the initializer and is never passed to the policy.

## Episode initialization constraints

The pinned upstream environment has no public arbitrary-state injection API. In the inspected revision, `reset()` always reconstructs the solved color vector and `step()` applies a move through the environment transition function. ZeroMethod avoids assigning directly to upstream `color_vector` or other internal state.

Create environments with `make_environment(max_steps=None)` when using `initialize_episode`. Upstream `max_steps` counts scramble-construction actions too. The episode runner enforces the agent's own step limit after initialization.

The offline `RandomWalkStateGenerator` and live initializer share one private action sampler. For the same seed, depth, action mapping, and environment revision, they produce the same walk. Immediate inverse pairs are excluded. If an intermediate prefix solves the cube, that candidate is discarded and another candidate is drawn deterministically; this prevents calling upstream `step()` after it has terminated. A solve on the final requested action is allowed.

Depth is the number of actions successfully replayed, not shortest-path distance. Random walks are not uniform samples of cube state space, and depth is not guaranteed optimal distance. The info mapping intentionally contains no scramble metadata.

## State integrity and reachability

Observations must contain 54 integer stickers, values 0..5, and nine of each color. Reachability provenance comes from real upstream transitions. The conventional fixed-center validator is not suitable for the full 18-action set because middle-slice turns relocate centers.

## Curriculum and reproducibility

A staged curriculum may increase random-walk depth. Derive distinct training, validation, and evaluation seeds with derive_split_seed. Record source commit, environment/submodule revision, dependency versions, seeds, split, depth schedule, action set, EAC metric, algorithm, hyperparameters, runtime, and metrics.

## Episode runner contract

Use `EpisodeRunner(env).run(policy, depth=..., seed=..., max_agent_steps=...)` for bounded episodes. The policy is a callable from the current observation to one action index. It is never called during scramble construction. The returned result records cumulative reward, agent-only step count, final observation, termination/truncation, and success. A zero-step budget is valid and immediately truncates an unsolved starting state. The runner does not own or close the environment.

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

This milestone establishes reproducible state generation and safe episode initialization only. It does not add or train an agent. V1 and V2 are random reference policies; V3 is a one-step model-based heuristic. The upstream reward is sparse (+1 solved, otherwise 0). Any future reward shaping must be documented and checked for solution-information leakage.
