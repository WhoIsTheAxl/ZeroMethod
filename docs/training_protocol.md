# Training and evaluation protocol

This document records constraints for experiments. It is not a claim that a training system already exists.

## Agent information boundary

The agent may receive:
- The current cube observation.
- The legal action set.
- The reward and episode boundary.

The agent must not receive:
- The scramble sequence.
- A reverse action list.
- A precomputed solution.
- Human solving demonstrations or named human methods.

Environment code may retain internal state needed to execute transitions, but the training interface must not expose hidden scramble metadata. The seeded random-walk generator returns only the resulting immutable observation; it does not attach the generated action sequence.

## Curriculum experiments

A staged curriculum may begin with shallow states and increase difficulty. Scramble depth is only a rough difficulty proxy: different sequences can cancel, revisit states, or produce the same state.

Training curriculum examples:
1. one-move states
2. two-move states
3. three-move states
4. progressively deeper states

The generator accepts an explicit seed. Recreate it with the same seed to reproduce its sequence of generated observations. Use derive_split_seed(master_seed, split_name) to derive stable, distinct seeds for training, validation, and evaluation. Record the master seed, split name, derived seed, and depth schedule for every experiment.

The generator avoids immediately selecting the inverse of the previous action. This reduces trivial two-step cancellations but does not make random walks uniform or guarantee a particular state's shortest-path distance.

## Evaluation separation

Final evaluation must use independently generated randomized cube states that were not used during training or hyperparameter selection.

Random action walks are useful for controlled curriculum experiments, but they are not automatically uniform samples from the cube state space. Do not label a fixed-length random walk as a uniform random-state benchmark.

Maintain separate training, validation, and final test sets. Derive and record a different seed for each split. The final test set must remain untouched until a model/configuration is frozen.

## Metrics

Report at least:
- solve success rate
- inference latency (clearly state warm or cold)
- solution length under a declared move metric
- run-to-run consistency
- generalization across independently generated states

Do not conflate solution-generation latency with physical robot or human execution time.

## Reproducibility

Each experiment should record:
- source commit
- environment and dependency versions
- algorithm and hyperparameters
- master and derived random seeds
- training-state generation procedure
- evaluation-state generation procedure
- scramble depth distribution
- hardware/runtime details
- metrics and raw result files

## Current limitations

The upstream GymRubiksCube environment uses 18 actions, including middle slices. Before reporting move efficiency, ZeroMethod must define whether this action set is retained or mapped to a conventional move metric.

The current upstream reward is sparse: +1 for solved and 0 otherwise. Any future reward shaping must be documented and reviewed to ensure it does not encode a human solving method or leak solution information.

The current state generator produces random walks from the solved state. It is a reproducible generator for controlled experiments, not an independently verified uniform sampler over legal cube states. Corner/edge orientation and permutation invariants still need independent validation before this generator is used for benchmark claims.
