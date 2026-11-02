# ZeroMethod

[![CI](https://github.com/WhoIsTheAxl/ZeroMethod/actions/workflows/ci.yml/badge.svg)](https://github.com/WhoIsTheAxl/ZeroMethod/actions/workflows/ci.yml)

**An experimental AI project exploring how an agent can learn to solve the Rubik's Cube through interaction.**

ZeroMethod aims to develop a cube-solving agent that learns the mechanics of the cube from its environment rather than being given a human solving method such as CFOP or Roux. Speed is a major goal, but solving ability and generalization come first.

> **Project status: Environment and baseline infrastructure.**  
> The repository includes a dependency-free environment adapter, a stable 18-action API, seeded episode initialization, an episode runner, observation-integrity checks, two random-policy baselines, unit tests, and GitHub Actions CI. No trained agent or solver is available yet.

## Project goals

- Learn cube mechanics through legal actions and environment feedback.
- Avoid hard-coding human solving methods or providing solution demonstrations.
- Evaluate on previously unseen randomized cube states.
- Measure solving success, solution length, consistency, and computation time.
- Optimize speed only after reliable solving behavior has been established.

## Current environment

ZeroMethod currently tracks [GymRubiksCube](https://github.com/mgroling/GymRubiksCube) as a Git submodule in `third_party/GymRubiksCube`.

The upstream environment currently provides:

| Property | Value |
| --- | --- |
| Gym environment ID | `RubiksCube-v0` |
| Observation | 54 sticker values |
| Action space | 18 discrete actions |
| Reward | +1 when the cube is solved; otherwise 0 |
| Rendering | Optional; not required for training |

The 18 actions include outer-layer and middle-slice turns. ZeroMethod retains all 18 as its canonical action set. The project-owned `Action` enum maps directly to upstream indices, with an explicit inverse-action mapping. Direction names preserve upstream semantics; no unverified Singmaster conversion is assumed.

### Compatibility note

GymRubiksCube uses the legacy Gym API: `reset()` returns an observation, and `step()` returns four values. Its setup file also pins older dependency versions. Compatibility with newer Python, NumPy, and Gym releases has not yet been established. Do not assume the upstream pinned dependencies will install cleanly in a modern Kaggle runtime; dependency compatibility is a separate validation task.

## Clone

Clone the repository together with its submodule:

```bash
git clone --recurse-submodules https://github.com/WhoIsTheAxl/ZeroMethod.git
cd ZeroMethod
```

If you already cloned the repository without its submodule:

```bash
git submodule update --init --recursive
```

## Random-policy baseline

Run a reproducible uniform-random baseline from the repository root after installing the upstream environment dependencies:

```bash
python -m zeromethod.experiments.random_baseline --episodes 100 --depth 10 --max-agent-steps 100 --seed 42
```

The command prints a JSON report with success rate, episode outcomes, agent steps, reward, policy-call count, policy latency, and elapsed time. The baseline samples uniformly from all 18 actions, including inverse moves. It is intentionally unoptimized. Scramble and policy seeds are derived independently from the master seed. Scramble depth must be positive to avoid counting solved resets as successes.

## Random-policy baseline v2

Run the second baseline from the repository root:

```bash
python -m zeromethod.experiments.random_baseline_v2 --episodes 100 --depth 10 --max-agent-steps 100 --seed 42
```

Unlike v1, v2 excludes the inverse of its immediately preceding action. The first action is selected from all 18 actions; later actions are selected uniformly from the remaining 17. It remains a random policy: it does not inspect the cube state, plan ahead, or prevent longer cycles. Both baselines share the same per-episode policy seed namespace so their first random choice is paired for the same master seed.

## Environment validation

The test suite has four layers:

- **Dependency-free adapter tests:** validate the adapter using a fake environment.
- **Dependency-free state-generation tests:** validate deterministic random walks and split-seed derivation.
- **Dependency-free observation-integrity tests:** validate shape, sticker range, and color counts.
- **Dependency-free fixed-center cubie tests:** validate center orientation, corner/edge uniqueness, twist/flip sums, and permutation parity for the fixed-center move subset.
- **Upstream integration tests:** validate the pinned GymRubiksCube implementation without opening its graphical renderer, verify all 18 action permutations geometrically, and check observation integrity throughout a 1,000-action random walk. A separate outer-face-only walk checks fixed-center cubie invariants.

The upstream project was originally tested with Python 3.8.10 and pins older dependency versions. CI currently validates it with a separate, explicitly pinned compatibility stack on Python 3.10 rather than installing the upstream package metadata (which contains an invalid legacy Gym dependency specifier). The integration job installs:

```bash
python -m pip install numpy==1.23.5 numba==0.57.1 pygame==2.5.2 Pillow==10.0.1 scipy==1.10.1 gym==0.25.2
```

Then expose the submodule on the import path when running the upstream tests:

```bash
PYTHONPATH=third_party/GymRubiksCube python -m unittest discover -s tests -p "test_environment.py" -v
```

Run the dependency-free tests from the repository root on any supported Python 3 version:

```bash
python -m unittest discover -s tests -p "test_adapter.py" -v
python -m unittest discover -s tests -p "test_actions.py" -v
python -m unittest discover -s tests -p "test_state_generation.py" -v
python -m unittest discover -s tests -p "test_state_legality.py" -v
```

The upstream integration tests currently verify:

- Reset produces a solved 54-sticker state.
- Each of the 18 actions is accepted and preserves nine stickers of each color.
- Every action's sticker permutation is a bijection.
- Each documented inverse-action pair restores the solved state.
- Applying any quarter-turn action four times restores the solved state.
- Solved-state detection and terminal reward behavior work as expected.

These are environment integrity checks, not proof that every physically reachable cube state is represented correctly. Further legality and compatibility checks are still required.

The state-generation tests verify:

- The same seed reproduces the same sequence of observations.
- Split seeds are stable and distinct for training, validation, and evaluation.
- Generated observations contain only 54 stickers with nine of each color.
- The generated observation does not carry scramble-history metadata.
- Invalid depth and inverse-action maps are rejected.

The state generator uses seeded random walks from the solved state and avoids immediately applying the previous action's inverse. `validate_observation` checks the full 18-action observation contract. The stronger `validate_cube_state` checks conventional fixed-center cubie invariants and is tested only on outer-face-turn states; it is not a gate for states containing upstream middle-slice turns. Random walks are not uniform samples of the state space.

GitHub Actions runs the dependency-free tests on Python 3.12 and the upstream integration suite on Python 3.10.

See [the environment architecture](docs/architecture.md) and [training/evaluation protocol](docs/training_protocol.md) for design constraints.

## Planned development

1. **Environment validation** — establish reliable state, action, and reward behavior.
2. **Environment adapter** — isolate the upstream environment behind a ZeroMethod interface.
3. **State generation and legality** — validate cubie invariants and define evaluation sampling.
4. **Baseline experiments** — benchmark environment throughput and establish reproducible evaluation.
5. **Agent experiments** — investigate learning approaches without injecting human solving algorithms.
6. **Generalization and performance** — evaluate on independent states and measure inference latency separately from solution quality.

## Training and evaluation principles

- Training and evaluation states must be kept separate.
- The agent must not receive the scramble sequence or a precomputed solution.
- Random move sequences are useful for curriculum experiments, but final evaluation should use independently generated randomized cube states.
- Report success rate, solution length, inference time, and consistency—not just the best run.

## Repository layout

```text
ZeroMethod/
├── .github/workflows/ci.yml
├── docs/
│   ├── action_space.md
│   ├── architecture.md
│   └── training_protocol.md
├── tests/
│   ├── test_actions.py
│   ├── test_adapter.py
│   ├── test_environment.py
│   ├── test_episode.py
│   ├── test_runner.py
│   ├── test_random_baseline.py
│   └── test_state_generation.py
├── zeromethod/
│   ├── environment/
│   │   ├── actions.py
│   │   ├── adapter.py
│   │   ├── episode.py
│   │   ├── factory.py
│   │   ├── runner.py
│   │   ├── state_generation.py
│   │   └── state_legality.py
│   └── experiments/
│       └── random_baseline.py
├── third_party/
│   └── GymRubiksCube/   # Upstream environment (Git submodule)
├── .gitmodules
├── LICENSE
└── README.md
```

The repository will expand as the environment adapter, experiments, and agent implementation are added.

## License

The root project includes a GNU General Public License v3.0 license file. Third-party components remain subject to their own license terms; see the upstream GymRubiksCube repository for its licensing information.
