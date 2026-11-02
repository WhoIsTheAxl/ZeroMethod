# ZeroMethod

**An experimental AI project exploring how an agent can learn to solve the Rubik's Cube through interaction.**

ZeroMethod aims to develop a cube-solving agent that learns the mechanics of the cube from its environment rather than being given a human solving method such as CFOP or Roux. Speed is a major goal, but solving ability and generalization come first.

> **Project status: Early setup / environment validation.**  
> The repository currently contains the environment dependency. No trained agent or solver is available yet.

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
| Rendering | Optional; should be disabled during training |

The 18 actions include layer and middle-slice turns. The action model and state transitions still need to be validated before training begins.

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

## Planned development

1. **Environment validation** — verify reset, all legal actions, inverse moves, solved-state detection, and cube-state invariants.
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
├── third_party/
│   └── GymRubiksCube/   # Upstream environment (Git submodule)
├── .gitmodules
├── LICENSE
└── README.md
```

The project structure will expand as the environment adapter, validation tests, experiments, and agent implementation are added.

## License

The root project includes a GNU General Public License v3.0 license file. Third-party components remain subject to their own license terms; see the upstream GymRubiksCube repository for its licensing information.
