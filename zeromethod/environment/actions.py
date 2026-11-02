"""Stable project-owned names for the upstream 18-action cube interface.

Values intentionally match GymRubiksCube's action indices. Direction names
preserve the upstream descriptions; they do not imply unverified Singmaster
clockwise conventions for the vertical actions.
"""

from enum import IntEnum
from operator import index


class Action(IntEnum):
    """One primitive quarter-turn action in the pinned upstream environment."""

    U_COUNTERCLOCKWISE = 0
    U_CLOCKWISE = 1
    E_COUNTERCLOCKWISE = 2
    E_CLOCKWISE = 3
    D_COUNTERCLOCKWISE = 4
    D_CLOCKWISE = 5
    B_COUNTERCLOCKWISE = 6
    B_CLOCKWISE = 7
    S_COUNTERCLOCKWISE = 8
    S_CLOCKWISE = 9
    F_COUNTERCLOCKWISE = 10
    F_CLOCKWISE = 11
    L_DOWN = 12
    L_UP = 13
    M_DOWN = 14
    M_UP = 15
    R_DOWN = 16
    R_UP = 17


ACTION_COUNT = 18

# Each pair is an inverse pair in the upstream transformation table.
INVERSE_ACTIONS = (
    Action.U_CLOCKWISE,
    Action.U_COUNTERCLOCKWISE,
    Action.E_CLOCKWISE,
    Action.E_COUNTERCLOCKWISE,
    Action.D_CLOCKWISE,
    Action.D_COUNTERCLOCKWISE,
    Action.B_CLOCKWISE,
    Action.B_COUNTERCLOCKWISE,
    Action.S_CLOCKWISE,
    Action.S_COUNTERCLOCKWISE,
    Action.F_CLOCKWISE,
    Action.F_COUNTERCLOCKWISE,
    Action.L_UP,
    Action.L_DOWN,
    Action.M_UP,
    Action.M_DOWN,
    Action.R_UP,
    Action.R_DOWN,
)


def inverse_action(action: int) -> Action:
    """Return the inverse action, rejecting booleans and invalid indices."""
    if isinstance(action, bool):
        raise TypeError("action must be an integer action index")
    try:
        action_index = index(action)
    except TypeError as exc:
        raise TypeError("action must be an integer action index") from exc

    if not 0 <= action_index < ACTION_COUNT:
        raise ValueError("action index must be in [0, {})".format(ACTION_COUNT))
    return INVERSE_ACTIONS[action_index]
