# Training and evaluation protocol

## Agent information boundary

The agent receives only its current observation, legal action set, reward, and episode boundary. It must not receive scramble actions, reverse action lists, precomputed solutions, or human solving demonstrations.

initialize_episode(env, depth, seed) constructs a starting state by replaying a seeded random walk through real environment transitions. It returns only the final observation and an empty info mapping. The action history is local to the initializer.

## Episode initialization constraints

The upstream environment has no public arbitrary-state injection API. ZeroMethod avoids direct mutation of its internal cube state. It resets to solved and executes generated actions through the adapter.

Create environments with make_environment(max_steps=None) when using initialize_episode. Upstream max_steps counts scramble-construction actions too. A future training runner must enforce the agent's own step limit after initialization.

Depth is the number of actions replayed, not shortest-path distance. Immediate inverse pairs are excluded. For a fixed environment version, action mapping, depth, and seed reproduce the same walk. The info mapping intentionally contains no scramble metadata.

## State integrity and reachability

Observations must contain 54 integer stickers, values 0..5, and nine of each color. Reachability provenance comes from upstream transitions. The conventional fixed-center validator is not suitable for the full 18-action set because middle-slice turns relocate centers.

## Curriculum and reproducibility

A staged curriculum may increase random-walk depth. Derive distinct training, validation, and evaluation seeds with derive_split_seed. Record source commit, environment/dependency versions, seeds, split, depth schedule, action set, EAC metric, algorithm, hyperparameters, runtime, and metrics.

Random walks are not uniform samples of cube state space, and depth is not guaranteed optimal distance.

## Evaluation and metrics

Keep training, validation, and final evaluation states separate. Report solve success rate, inference latency, solution length under a declared metric, run-to-run consistency, and generalization across independent seeds.

## Current limitations

No agent, training loop, or trained model exists yet. The upstream reward is sparse (+1 solved, otherwise 0). Any reward shaping must be documented and checked for solution-information leakage.
