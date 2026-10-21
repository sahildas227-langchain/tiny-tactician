"""Static evaluation of an Othello position.

The features are the standard ones used by strong Othello engines: corners,
edges, the corner-adjacent "X" and "C" squares, mobility, potential mobility
and the disc differential.  The evaluation is always from the point of view of
the player to move, and a positive score means that player is better off.
"""

from __future__ import annotations

from tiny_tactician.games.reversi import DARK, DIRS, M, _shift, legal_moves, popcount

__all__ = ["evaluate", "CORNERS", "X_SQUARES", "C_SQUARES"]

CORNERS = (0, 7, 56, 63)
X_SQUARES = (1, 8, 6, 15, 48, 57, 62, 55)
C_SQUARES = (9, 14, 50, 54)

W_CORNER = 50
W_EDGE = 2
W_X = -20
W_C = -20
W_MOBILITY = 10
W_POTENTIAL_MOBILITY = 4


def _mask(indices) -> int:
    return sum(1 << i for i in indices)


CORNER_MASK = _mask(CORNERS)
X_MASK = _mask(X_SQUARES)
C_MASK = _mask(C_SQUARES)
EDGE_MASK = _mask([i for i in range(64) if (i // 8 in (0, 7) or i % 8 in (0, 7)) and i not in CORNERS])


def _neighbours(bb: int) -> int:
    """Every square orthogonally or diagonally adjacent to a square in ``bb``."""
    out = 0
    for amt, left, pre in DIRS:
        out |= _shift(bb, amt, left, pre)
    return out


def evaluate(dark: int, light: int, color: int) -> float:
    """Evaluate the position for the player ``color`` to move."""
    me, opp = (dark, light) if color == DARK else (light, dark)
    empties = ~(dark | light) & M
    my_potential = popcount(me & _neighbours(empties))
    opp_potential = popcount(opp & _neighbours(empties))
    disc_difference = popcount(me) - popcount(opp)
    # The disc count matters more the fuller the board gets.
    disc_weight = 1.0 + 9.0 * (64 - popcount(empties)) / 64.0
    return (
        W_CORNER * (popcount(me & CORNER_MASK) - popcount(opp & CORNER_MASK))
        + W_EDGE * (popcount(me & EDGE_MASK) - popcount(opp & EDGE_MASK))
        + W_X * (popcount(me & X_MASK) - popcount(opp & X_MASK))
        + W_C * (popcount(me & C_MASK) - popcount(opp & C_MASK))
        + W_MOBILITY * (popcount(legal_moves(dark, light, color)) - popcount(legal_moves(dark, light, -color)))
        + W_POTENTIAL_MOBILITY * (my_potential - opp_potential)
        + disc_weight * disc_difference
    )
