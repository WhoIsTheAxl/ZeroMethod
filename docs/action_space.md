# Upstream action-space audit

## Scope and source

This document audits the action mapping implemented by the pinned
`GymRubiksCube` submodule, specifically
`gym_rubiks_cube/envs/rubiksCubeEnv.py` and its
`TransformCubeObject.transformation_permutations` table.

The upstream source describes its action directions from the perspective of
looking at the red face, with the white face facing upward. Its face colors are:

| Color ID | Color | Face |
| ---: | --- | --- |
| 0 | White | U (up) |
| 1 | Red | F (front) |
| 2 | Blue | R (right) |
| 3 | Orange | B (back) |
| 4 | Green | L (left) |
| 5 | Yellow | D (down) |

The observation is a flat array of 54 stickers, grouped into six consecutive
nine-sticker face blocks in the order U, F, R, B, L, D.

## Action mapping

The action space is `Discrete(18)`. Each action applies a quarter-turn sticker
permutation. The action index is also the index into
`TransformCubeObject.transformation_permutations`.

| Index | Upstream description | Slice / face | Inverse |
| ---: | --- | --- | ---: |
| 0 | Top-layer counter-clockwise | U | 1 |
| 1 | Top-layer clockwise | U | 0 |
| 2 | Middle-layer horizontal counter-clockwise | E (between U and D) | 3 |
| 3 | Middle-layer horizontal clockwise | E (between U and D) | 2 |
| 4 | Bottom-layer counter-clockwise | D | 5 |
| 5 | Bottom-layer clockwise | D | 4 |
| 6 | Back-layer counter-clockwise | B | 7 |
| 7 | Back-layer clockwise | B | 6 |
| 8 | Middle-layer counter-clockwise | S (between F and B) | 9 |
| 9 | Middle-layer clockwise | S (between F and B) | 8 |
| 10 | Front-layer counter-clockwise | F | 11 |
| 11 | Front-layer clockwise | F | 10 |
| 12 | Left-layer downwards | L | 13 |
| 13 | Left-layer upwards | L | 12 |
| 14 | Middle-layer vertical downwards | M (between L and R) | 15 |
| 15 | Middle-layer vertical upwards | M (between L and R) | 14 |
| 16 | Right-layer downwards | R | 17 |
| 17 | Right-layer upwards | R | 16 |

The E/S/M labels identify the physical slice location. For actions 12–17, this
document preserves the upstream source's unambiguous directional wording
("downwards" / "upwards") rather than assigning clockwise notation without an
explicit, independently verified viewing convention.

## Verified transition properties

The repository's upstream integration tests check these properties directly
against the implementation:

- Every action's 54-entry sticker permutation is a bijection.
- Each listed inverse pair restores the original state when applied in
  succession.
- Applying any one action four times restores the original state.
- A seeded 1,000-action walk, including all three middle slices, remains
  accepted by the independent cube-state legality validator.
- The inverse map is the adjacent-pair rule: even action index + 1 is its
  inverse; odd action index - 1 is its inverse.

These checks validate the transition permutations and their group behavior.
They do not independently prove that every upstream direction label matches
standard Singmaster notation.

## Consequences for ZeroMethod

### The action set is not standard outer-face-only HTM

The environment includes six outer-face turns and three middle-slice turns,
each in two directions. It therefore has 18 primitive actions, rather than
the 12 quarter-turn actions of an outer-face-only QTM action set.

Do not report solution length as standard HTM or QTM without defining a
conversion. In particular, a middle-slice turn must not silently be counted as
a conventional outer-face turn.

### Provisional project move metric

Until a deliberate action-set change is made, use **environment action count**
(EAC) for internal experiments:

- Every one of the 18 discrete actions costs 1 EAC.
- A sequence's EAC length is the number of actions emitted by the policy.
- Inverse actions are distinct choices and each costs 1 EAC.
- No half-turn action exists; a half-turn requires two quarter-turn actions.
- Middle-slice turns cost the same as outer-layer turns in EAC.

EAC is a project-defined metric, not a claim of equivalence to HTM, QTM, or
human execution effort. Any paper, benchmark, or public result must name the
metric and action set explicitly.

### Reward and termination

The upstream environment returns reward +1 when the resulting state is solved,
and 0 otherwise. It terminates on a solved state or when the optional
`max_steps` limit is reached. The reward does not distinguish shorter from
longer unsolved paths.

## Open questions

1. Decide whether ZeroMethod keeps all 18 actions or introduces a separately
   validated outer-face-only action mode.
2. Independently verify the clockwise / counter-clockwise labels against a
   fixed geometric viewing convention before exposing Singmaster strings in
   code or experiment logs.
3. Benchmark the cost and learning impact of middle-slice actions rather than
   assuming that a larger action space is automatically beneficial.
