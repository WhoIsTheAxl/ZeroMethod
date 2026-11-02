# ZeroMethod

**An experimental AI project exploring how an agent can learn to solve the Rubik's Cube through interaction.**

ZeroMethod aims to develop a cube-solving agent that learns the mechanics of the cube from its environment rather than being given a human solving method such as CFOP or Roux. Speed is a major goal, but solving ability and generalization come first.

> **Project status: Environment validation.**  
> The repository contains the environment dependency and its first contract tests. No trained agent or solver is available yet.

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

The 18 actions include outer-layer and middle-slice turns. Their permutations and inverse relationships are covered by the initial validation tests.

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

## Environment validation

The first test suite checks the upstream environment's basic contract without opening its graphical renderer.

The upstream project was tested with Python 3.8.10. Its setup file pins older versions of NumPy, Gym, and other packages, so installation may fail on newer Python runtimes. In a compatible environment, install the submodule in editable mode:

```bash
python -m pip install -e third_party/GymRubiksCube
```

Then run the tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests currently verify:

- Reset produces a solved 54-sticker state.
- Each of the 18 actions is accepted and preserves nine stickers of each color.
- Every action's sticker permutation is a bijection.
- Each documented inverse-action pair restores the solved state.
- Applying any quarter-turn action four times restores the solved state.
- Solved-state detection and terminal reward behavior work as expected.

These are environment integrity checks, not proof that every physically reachable cube state is represented correctly. Further legality and compatibility checks are still required.

## Planned development

1. **Environment validation** — establish reliable state, action, and reward behavior.
2. **Environment adapter** — isolate the upstream environment behind a ZeroMethod interface.
3. **Baseline experiments** — benchmark environment throughput and establish reproducible evaluation.
4. **Agent experiments** — investigate learning approaches without injecting human solving algorithms.
5. **Generalization tests** — evaluate on independent randomized states not used during training.
6. **Performance work** — measure inference latency and solution quality separately.

## Training and evaluation principles

- Training and evaluation states must be kept separate.
- The agent must not receive the scramble sequence or a precomputed solution.
- Random move sequences are useful for curriculum experiments, but final evaluation should use independently generated randomized cube states.
- Report success rate, solution length, inference time, and consistency—not just the best run.

## Repository layout

```text
ZeroMethod/
├── tests/
│   └── test_environment.py
├── third_party/
│   └── GymRubiksCube/   # Upstream environment (Git submodule)
├── .gitmodules
├── LICENSE
└── README.md
```

The repository will expand as the environment adapter, experiments, and agent implementation are added.

## License

The root project includes a GNU General Public License v3.0 license file. Third-party components remain subject to their own license terms; see the upstream GymRubiksCube repository for its licensing information.
