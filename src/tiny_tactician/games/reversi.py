"""Pure-Python bitboard Othello (Reversi) engine.

Board geometry
--------------
Squares are indexed ``idx = row * 8 + col`` with ``row`` counted from rank 1
(bottom) and ``col`` from file a (left), so ``idx 0 == a1`` and ``idx 63 ==
h8``.  This is standard Othello notation: the four central discs are d4/e5
(dark) and e4/d5 (light).  A position is two 64-bit Python integers, ``dark``
and ``light``, where a set bit means a piece of that color occupies the square.
Python ints are arbitrary-precision, so every operation is an exact integer
bit-op and the engine needs no third-party deps.

Dark (X) moves first.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

__all__ = [
    "DARK", "LIGHT", "EMPTY", "PASS",
    "M", "FILE_A", "FILE_H", "DIRS", "WEIGHTS",
    "start", "legal_moves", "is_legal", "flips", "play",
    "game_over", "popcount", "score", "outcome",
    "idx_to_rc", "rc_to_idx", "parse_move", "format_move",
    "to_grid", "render", "heuristic_prior", "random_move",
    "Reversi",
]

DARK = 1
LIGHT = -1
EMPTY = 0
PASS = 0

M = (1 << 64) - 1
FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080

# Each direction is (shift_amount, left_shift?, pre_mask).
# `pre_mask` strips the file that would wrap around during the shift, so a piece
# on the a-file never "wraps" into the h-file of an adjacent row.
# Row indices grow towards rank 8, i.e. upwards on the printed board.
DIRS = [
    (1, True,  M & ~FILE_H),   # E  (col+1)
    (1, False, M & ~FILE_A),   # W  (col-1)
    (8, True,  M),             # N  (row+1, towards rank 8)
    (8, False, M),             # S  (row-1, towards rank 1)
    (7, True,  M & ~FILE_A),   # NW (row+1, col-1)
    (7, False, M & ~FILE_H),   # SE (row-1, col+1)
    (9, True,  M & ~FILE_H),   # NE (row+1, col+1)
    (9, False, M & ~FILE_A),   # SW (row-1, col-1)
]

# Classic Othello positional weights (corners maximal, the squares diagonally
# adjacent to a corner minimal).  Owning a high-weight square is good for the
# side that owns it, so the same table serves both colors.
WEIGHTS = [
    [100, -50, 30, 10, 10, 30, -50, 100],
    [-50, -75, -3, -3, -3, -3, -75, -50],
    [30, -3, 20, 5, 5, 20, -3, 30],
    [10, -3, 5, 1, 1, 5, -3, 10],
    [10, -3, 5, 1, 1, 5, -3, 10],
    [30, -3, 20, 5, 5, 20, -3, 30],
    [-50, -75, -3, -3, -3, -3, -75, -50],
    [100, -50, 30, 10, 10, 30, -50, 100],
]


def _shift(x: int, amt: int, left: bool, pre: int) -> int:
    x &= pre
    return ((x << amt) & M) if left else ((x >> amt) & M)


def _set_bits(x: int):
    i = 0
    while x:
        if x & 1:
            yield i
        x >>= 1
        i += 1


def popcount(x: int) -> int:
    return x.bit_count()


def start() -> tuple[int, int, int]:
    """Return ``(dark, light, DARK)`` for the standard opening position.

    The four central discs are placed as in every Othello reference: the first
    player (dark) holds e4 and d5, the second player (light) holds d4 and e5.
    Dark's four legal opening moves are therefore d3, c4, f5 and e6.
    """
    dark = (1 << (3 * 8 + 4)) | (1 << (4 * 8 + 3))   # e4, d5
    light = (1 << (3 * 8 + 3)) | (1 << (4 * 8 + 4))  # d4, e5
    return dark, light, DARK


def _legal(player: int, opp: int) -> int:
    """Bitboard of legal moves for ``player`` given ``opp`` (core, 2-arg)."""
    empties = ~(player | opp) & M
    moves = 0
    for amt, left, pre in DIRS:
        # Opponent pieces directly adjacent to a friendly piece start a capture
        # chain; grow the chain one step at a time.
        t = _shift(player, amt, left, pre) & opp
        while t:
            # The square just past a chain end is a legal move if it is empty.
            # Every chain depth must be considered: a short chain and a long
            # chain can run in the same direction, and the short one ends on an
            # empty square that the long chain would hide.
            moves |= _shift(t, amt, left, pre) & empties
            t = _shift(t, amt, left, pre) & opp
    return moves


def _flips(player: int, opp: int, move: int) -> int:
    """Bitboard of opponent pieces that ``move`` flips (core, 3-arg)."""
    flip = 0
    for amt, left, pre in DIRS:
        # Walk outwards from the move square, one square at a time, collecting
        # the run of opponent discs.  The frontier stays a single square, so
        # the walk cannot jump across two separate runs.
        t = _shift(move, amt, left, pre)
        chain = 0
        while t & opp:
            chain |= t
            t = _shift(t, amt, left, pre)
        # The run only captures if it is terminated by a friendly disc.
        if t & player:
            flip |= chain
    return flip


def legal_moves(dark: int, light: int, color: int) -> int:
    """Bitboard of the empty squares where ``color`` may legally move."""
    if color == DARK:
        return _legal(dark, light)
    return _legal(light, dark)


def is_legal(dark: int, light: int, color: int, move: int) -> bool:
    move &= M
    return move != 0 and (legal_moves(dark, light, color) & move) == move


def flips(dark: int, light: int, color: int, move: int) -> int:
    """Bitboard of opponent pieces that ``move`` would flip (0 if illegal)."""
    if color == DARK:
        return _flips(dark, light, move)
    return _flips(light, dark, move)


def play(dark: int, light: int, color: int, move: int) -> tuple[int, int]:
    """Apply a legal move for ``color``; return updated ``(dark, light)``."""
    if color == DARK:
        player, opp = dark, light
    else:
        player, opp = light, dark
    flip = _flips(player, opp, move)
    player = player | move | flip
    opp = opp ^ flip
    if color == DARK:
        return player, opp
    return opp, player


def game_over(dark: int, light: int) -> bool:
    return legal_moves(dark, light, DARK) == 0 and legal_moves(dark, light, LIGHT) == 0


def score(dark: int, light: int) -> tuple[int, int]:
    return popcount(dark), popcount(light)


def outcome(dark: int, light: int) -> int:
    """Game result from DARK's perspective: DARK(+1) / LIGHT(-1) / draw(0)."""
    ds, ls = popcount(dark), popcount(light)
    return DARK if ds > ls else LIGHT if ls > ds else EMPTY


def idx_to_rc(idx: int) -> tuple[int, int]:
    return divmod(idx, 8)


def rc_to_idx(row: int, col: int) -> int:
    return row * 8 + col


def parse_move(text: str) -> int:
    """Parse ``a1``..``h8`` (col letter + row digit, row 1 = top) or an index.

    Returns a single-bit bitboard, or ``PASS`` (0) for pass / empty input.
    """
    s = text.strip().lower()
    if s in ("", "pass", "p", "-"):
        return PASS
    if len(s) == 2 and "a" <= s[0] <= "h" and s[1].isdigit():
        col = ord(s[0]) - ord("a")
        row = int(s[1]) - 1
        if 0 <= row < 8 and 0 <= col < 8:
            return 1 << (row * 8 + col)
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < 64:
            return 1 << idx
    raise ValueError(f"cannot parse move: {text!r}")


def format_move(move: int) -> str:
    if move == 0:
        return "pass"
    idx = move.bit_length() - 1
    row, col = idx_to_rc(idx)
    return f"{chr(ord('a') + col)}{row + 1}"


def to_grid(dark: int, light: int) -> list[list[int]]:
    """8x8 grid: 1 = dark, 2 = light, 0 = empty.

    Row 0 is rank 1 (the bottom row), matching the bit layout.
    """
    grid = [[0] * 8 for _ in range(8)]
    for row in range(8):
        for col in range(8):
            bit = 1 << (row * 8 + col)
            if dark & bit:
                grid[row][col] = 1
            elif light & bit:
                grid[row][col] = 2
    return grid


def render(dark: int, light: int, color: int = DARK, legal_mask: int = 0, last: int = 0) -> str:
    """ASCII board. X=dark, O=light, *=legal empty, #=last move.

    The board is drawn with rank 8 at the top, as in standard Othello notation.
    """
    lines = ["   a b c d e f g h"]
    for row in range(7, -1, -1):
        cells = []
        for col in range(8):
            bit = 1 << (row * 8 + col)
            if dark & bit:
                cells.append("X")
            elif light & bit:
                cells.append("O")
            elif bit == last:
                cells.append("#")
            elif legal_mask & bit:
                cells.append("*")
            else:
                cells.append(".")
        lines.append(f" {row + 1}  " + " ".join(cells))
    lines.append("   a b c d e f g h")
    lines.append(f"  Current: {'X' if color == DARK else 'O'}  (X=dark, O=light, *=legal, #=last)")
    return "\n".join(lines)


def heuristic_prior(dark: int, light: int, color: int, temperature: float = 3.0) -> dict[int, float]:
    """Positional prior over legal moves (corner/edge weights, softmax).

    Used as a sensible floor beneath the language model's own priors, and as the
    engine when the model is disabled (``--no-llm``).
    """
    moves = legal_moves(dark, light, color)
    if moves == 0:
        return {}
    scores: dict[int, float] = {}
    for idx in _set_bits(moves):
        r, c = idx_to_rc(idx)
        scores[1 << idx] = float(WEIGHTS[r][c])
    top = max(scores.values())
    exps = {b: math.exp((v - top) * temperature) for b, v in scores.items()}
    z = sum(exps.values())
    return {b: v / z for b, v in exps.items()}


def random_move(dark: int, light: int, color: int, rng: random.Random) -> int:
    bits = [1 << i for i in _set_bits(legal_moves(dark, light, color))]
    return rng.choice(bits) if bits else PASS


@dataclass
class Reversi:
    dark: int
    light: int
    color: int = DARK
    last: int = 0

    @classmethod
    def start(cls) -> Reversi:
        d, l, _ = start()
        return cls(d, l)

    def legal(self) -> int:
        return legal_moves(self.dark, self.light, self.color)

    def legal_list(self) -> list[int]:
        return [1 << i for i in _set_bits(self.legal())]

    def can_move(self) -> bool:
        return self.legal() != 0

    def play(self, move: int) -> Reversi:
        if move == PASS:
            self.color = -self.color
            self.last = 0
            return self
        if not is_legal(self.dark, self.light, self.color, move):
            raise ValueError(f"illegal move {format_move(move)!r}")
        self.dark, self.light = play(self.dark, self.light, self.color, move)
        self.last = move
        self.color = -self.color
        # If the opponent now has no legal move, they must pass and the turn
        # reverts to the side that just moved.
        if not self.can_move():
            self.color = -self.color
        return self

    def is_over(self) -> bool:
        return game_over(self.dark, self.light)

    def winner(self) -> int:
        ds, ls = score(self.dark, self.light)
        return DARK if ds > ls else LIGHT if ls > ds else EMPTY

    def render(self, legal_mask: int | None = None) -> str:
        mask = self.legal() if legal_mask is None else legal_mask
        return render(self.dark, self.light, self.color, mask, self.last)

    def copy(self) -> Reversi:
        return Reversi(self.dark, self.light, self.color, self.last)
