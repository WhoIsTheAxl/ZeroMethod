"""Independent 3x3x3 cube-state legality checks.

Facelet order follows GymRubiksCube's observation layout:
U, F, R, B, L, D (nine stickers per face), with colors:
white=0, red=1, blue=2, orange=3, green=4, yellow=5.

Cubie tables are expressed in the conventional Kociemba U,R,F,D,L,B
orientation and remapped to the upstream facelet layout.
"""

from itertools import permutations, product
from operator import index
from typing import Iterable, Tuple


# Convert a conventional facelet index (U,R,F,D,L,B) into the upstream
# observation layout (U,F,R,B,L,D).
_STANDARD_TO_UPSTREAM_FACE = (0, 18, 9, 45, 36, 27)

# Face order in the upstream observation: U, F, R, B, L, D.
_FACE_NORMALS = (
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 0),
    (0, 0, -1),
    (-1, 0, 0),
    (0, -1, 0),
)
_COLOR_NORMALS = _FACE_NORMALS
_CENTER_FACELETS = (4, 13, 22, 31, 40, 49)


def _facelet_geometry():
    geometry = []
    for face in range(6):
        for row in range(3):
            for column in range(3):
                if face == 0:       # U: viewed from above, back to front
                    position = (column - 1, 1, row - 1)
                elif face == 1:     # F: viewed from front
                    position = (column - 1, 1 - row, 1)
                elif face == 2:     # R: viewed from right, front to back
                    position = (1, 1 - row, 1 - column)
                elif face == 3:     # B: viewed from back
                    position = (1 - column, 1 - row, -1)
                elif face == 4:     # L: viewed from left, back to front
                    position = (-1, 1 - row, column - 1)
                else:               # D: viewed from below, front to back
                    position = (column - 1, -1, 1 - row)
                geometry.append((position, _FACE_NORMALS[face]))
    return tuple(geometry)


_FACELET_GEOMETRY = _facelet_geometry()
_FACELET_BY_GEOMETRY = {
    geometry: index for index, geometry in enumerate(_FACELET_GEOMETRY)
}


def _rotation_matrices():
    matrices = []
    for axes in permutations((0, 1, 2)):
        inversions = sum(
            axes[i] > axes[j]
            for i in range(3)
            for j in range(i + 1, 3)
        )
        permutation_sign = -1 if inversions % 2 else 1
        for signs in product((-1, 1), repeat=3):
            if permutation_sign * signs[0] * signs[1] * signs[2] == 1:
                matrices.append((axes, signs))
    return tuple(matrices)


_ROTATIONS = _rotation_matrices()


def _rotate_vector(vector, rotation):
    axes, signs = rotation
    return tuple(signs[i] * vector[axes[i]] for i in range(3))


def _canonicalize_frame(state):
    """Rotate a valid center arrangement into the canonical color frame."""
    current_normals = [None] * 6
    for face, center_index in enumerate(_CENTER_FACELETS):
        color = state[center_index]
        if color < 0 or color > 5 or current_normals[color] is not None:
            raise ValueError("center stickers must contain each color exactly once")
        current_normals[color] = _FACE_NORMALS[face]

    for rotation in _ROTATIONS:
        if all(
            _rotate_vector(current_normals[color], rotation)
            == _COLOR_NORMALS[color]
            for color in range(6)
        ):
            canonical = [None] * 54
            for source, (position, normal) in enumerate(_FACELET_GEOMETRY):
                target_geometry = (
                    _rotate_vector(position, rotation),
                    _rotate_vector(normal, rotation),
                )
                target = _FACELET_BY_GEOMETRY.get(target_geometry)
                if target is None:
                    raise ValueError("cube orientation did not map facelets correctly")
                canonical[target] = state[source]
            return tuple(canonical)

    raise ValueError("center colors do not form a rigid cube orientation")


def _facelet(standard_index: int) -> int:
    face, offset = divmod(standard_index, 9)
    return _STANDARD_TO_UPSTREAM_FACE[face] + offset


def _remap(table):
    return tuple(tuple(_facelet(position) for position in piece) for piece in table)


_CORNER_FACELETS = _remap((
    (8, 9, 20),    # URF: U9, R1, F3
    (6, 18, 38),   # UFL: U7, F1, L3
    (0, 36, 47),   # ULB: U1, L1, B3
    (2, 45, 11),   # UBR: U3, B1, R3
    (29, 26, 15),  # DFR: D3, F9, R7
    (27, 44, 24),  # DLF: D1, L9, F7
    (33, 53, 42),  # DBL: D7, B9, L7
    (35, 17, 51),  # DRB: D9, R9, B7
))

_CORNER_COLORS = (
    (0, 2, 1),  # URF
    (0, 1, 4),  # UFL
    (0, 4, 3),  # ULB
    (0, 3, 2),  # UBR
    (5, 1, 2),  # DFR
    (5, 4, 1),  # DLF
    (5, 3, 4),  # DBL
    (5, 2, 3),  # DRB
)

_EDGE_FACELETS = _remap((
    (5, 10),   # UR: U6, R2
    (7, 19),   # UF: U8, F2
    (3, 37),   # UL: U4, L2
    (1, 46),   # UB: U2, B2
    (32, 16),  # DR: D6, R8
    (28, 25),  # DF: D2, F8
    (30, 43),  # DL: D4, L8
    (34, 52),  # DB: D8, B8
    (23, 12),  # FR: F6, R4
    (21, 41),  # FL: F4, L6
    (50, 39),  # BL: B6, L4
    (48, 14),  # BR: B4, R6
))

_EDGE_COLORS = (
    (0, 2),  # UR
    (0, 1),  # UF
    (0, 4),  # UL
    (0, 3),  # UB
    (5, 2),  # DR
    (5, 1),  # DF
    (5, 4),  # DL
    (5, 3),  # DB
    (1, 2),  # FR
    (1, 4),  # FL
    (3, 4),  # BL
    (3, 2),  # BR
)


def _permutation_parity(permutation: Tuple[int, ...]) -> int:
    inversions = 0
    for i, value in enumerate(permutation):
        for later in permutation[i + 1:]:
            if value > later:
                inversions += 1
    return inversions & 1


def validate_observation(observation: Iterable[int]) -> None:
    """Validate the 54-sticker observation contract, not physical reachability."""
    try:
        state = tuple(index(value) for value in observation)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation must contain integer sticker colors") from exc
    if len(state) != 54:
        raise ValueError("observation must contain exactly 54 stickers")
    if any(color < 0 or color > 5 for color in state):
        raise ValueError("sticker colors must be in the range 0 through 5")
    if tuple(state.count(color) for color in range(6)) != (9, 9, 9, 9, 9, 9):
        raise ValueError("each sticker color must occur exactly nine times")


def is_valid_observation(observation: Iterable[int]) -> bool:
    """Return whether an observation satisfies the 54-sticker contract."""
    try:
        validate_observation(observation)
    except (TypeError, ValueError):
        return False
    return True


def validate_cube_state(observation: Iterable[int]) -> None:
    """Validate conventional fixed-center cubie invariants.

    This is not applicable to arbitrary states produced by upstream
    middle-slice actions, which relocate center stickers.
    """
    try:
        state = tuple(index(value) for value in observation)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation must contain integer sticker colors") from exc

    if len(state) != 54:
        raise ValueError("observation must contain exactly 54 stickers")
    if any(color < 0 or color > 5 for color in state):
        raise ValueError("sticker colors must be in the range 0 through 5")

    counts = tuple(state.count(color) for color in range(6))
    if counts != (9, 9, 9, 9, 9, 9):
        raise ValueError("each sticker color must occur exactly nine times")

    state = _canonicalize_frame(state)

    centers = _CENTER_FACELETS
    if tuple(state[position] for position in centers) != (0, 1, 2, 3, 4, 5):
        raise ValueError("center colors or face orientation do not match the solved frame")

    corner_permutation = []
    corner_orientation_sum = 0
    for facelets in _CORNER_FACELETS:
        colors = tuple(state[position] for position in facelets)
        orientation = next(
            (i for i, color in enumerate(colors) if color in (0, 5)),
            None,
        )
        if orientation is None:
            raise ValueError("corner cubie has no U/D color")

        first = colors[(orientation + 1) % 3]
        second = colors[(orientation + 2) % 3]
        piece = next(
            (
                cubie
                for cubie, solved_colors in enumerate(_CORNER_COLORS)
                if solved_colors[1] == first and solved_colors[2] == second
            ),
            None,
        )
        if piece is None:
            raise ValueError("corner contains an impossible color combination")
        corner_permutation.append(piece)
        corner_orientation_sum += orientation

    if len(set(corner_permutation)) != 8:
        raise ValueError("corner cubies must each appear exactly once")
    if corner_orientation_sum % 3 != 0:
        raise ValueError("corner twist sum must be divisible by three")

    edge_permutation = []
    edge_orientation_sum = 0
    for facelets in _EDGE_FACELETS:
        colors = tuple(state[position] for position in facelets)
        piece = next(
            (
                cubie
                for cubie, solved_colors in enumerate(_EDGE_COLORS)
                if colors == solved_colors or colors == solved_colors[::-1]
            ),
            None,
        )
        if piece is None:
            raise ValueError("edge contains an impossible color combination")

        edge_permutation.append(piece)
        if colors != _EDGE_COLORS[piece]:
            edge_orientation_sum += 1

    if len(set(edge_permutation)) != 12:
        raise ValueError("edge cubies must each appear exactly once")
    if edge_orientation_sum % 2 != 0:
        raise ValueError("edge flip sum must be even")

    if _permutation_parity(tuple(corner_permutation)) != _permutation_parity(
        tuple(edge_permutation)
    ):
        raise ValueError("corner and edge permutation parity must match")


def is_legal_cube_state(observation: Iterable[int]) -> bool:
    """Return whether observation is a physically reachable fixed-frame state."""
    try:
        validate_cube_state(observation)
    except (TypeError, ValueError):
        return False
    return True
