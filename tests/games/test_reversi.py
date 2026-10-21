"""Tests for the bitboard Othello engine, cross-checked against a slow but
obviously-correct array-based reference implementation."""

from __future__ import annotations

import random

import pytest

from tiny_tactician.games.reversi import (
    DARK,
    LIGHT,
    PASS,
    Reversi,
    format_move,
    game_over,
    legal_moves,
    outcome,
    parse_move,
    play,
    random_move,
    start,
    to_grid,
)

DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _arr(dark, light):
    g = [[0] * 8 for _ in range(8)]
    for r in range(8):
        for c in range(8):
            bit = 1 << (r * 8 + c)
            if dark & bit:
                g[r][c] = 1
            elif light & bit:
                g[r][c] = 2
    return g


def _arr_legal(dark, light, color):
    pc, opc = (1, 2) if color == DARK else (2, 1)
    g = _arr(dark, light)
    res = 0
    for r in range(8):
        for c in range(8):
            if g[r][c] != 0:
                continue
            ok = False
            for dr, dc in DIRS8:
                rr, cc = r + dr, c + dc
                seen = 0
                while 0 <= rr < 8 and 0 <= cc < 8 and g[rr][cc] == opc:
                    seen += 1
                    rr += dr
                    cc += dc
                if seen > 0 and 0 <= rr < 8 and 0 <= cc < 8 and g[rr][cc] == pc:
                    ok = True
                    break
            if ok:
                res |= 1 << (r * 8 + c)
    return res


def _arr_play(dark, light, color, move):
    pc, opc = (1, 2) if color == DARK else (2, 1)
    g = _arr(dark, light)
    idx = move.bit_length() - 1
    r, c = divmod(idx, 8)
    g[r][c] = pc
    for dr, dc in DIRS8:
        rr, cc = r + dr, c + dc
        chain = []
        while 0 <= rr < 8 and 0 <= cc < 8 and g[rr][cc] == opc:
            chain.append((rr, cc))
            rr += dr
            cc += dc
        if chain and 0 <= rr < 8 and 0 <= cc < 8 and g[rr][cc] == pc:
            for a, b in chain:
                g[a][b] = pc
    nd = nl = 0
    for r in range(8):
        for c in range(8):
            v = g[r][c]
            if v == 1:
                nd |= 1 << (r * 8 + c)
            elif v == 2:
                nl |= 1 << (r * 8 + c)
    return nd, nl


def _random_game(rng):
    """Play a random game, cross-checking the engine against the array oracle."""
    dark, light, color = start()
    for _ in range(130):
        lm = legal_moves(dark, light, color)
        assert lm == _arr_legal(dark, light, color)
        if lm == 0:
            color = -color
            if legal_moves(dark, light, color) == 0:
                break
            continue
        move = rng.choice([1 << i for i in _set_bits_gen(lm)])
        dark, light = play(dark, light, color, move)
        assert to_grid(dark, light) == _arr(dark, light)
        color = -color
    return dark, light


def _set_bits_gen(x):
    i = 0
    while x:
        if x & 1:
            yield i
        x >>= 1
        i += 1


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_engine_matches_array_oracle(seed):
    dark, light = _random_game(random.Random(seed))
    assert game_over(dark, light)
    # final disc counts agree with the array reference
    assert (dark.bit_count(), light.bit_count()) == (
        sum(row.count(1) for row in _arr(dark, light)),
        sum(row.count(2) for row in _arr(dark, light)),
    )
    assert outcome(dark, light) in (DARK, LIGHT, 0)


def test_starting_position():
    dark, light, color = start()
    assert legal_moves(dark, light, color).bit_count() == 4
    assert game_over(dark, light) is False
    # every opening move flips exactly one disc
    for move in [1 << i for i in _set_bits_gen(legal_moves(dark, light, color))]:
        nd, nl = play(dark, light, color, move)
        assert nd.bit_count() == 4
        assert nl.bit_count() == 1
        assert to_grid(nd, nl) == _arr(nd, nl)


def test_multiple_chains_in_one_direction():
    """Two capture chains running in the same direction must both be found.

    Row 0: X at a1, O at b1, O at c1, X at e1, O at f1.  The chain a1-b1-c1
    ends at d1 and the chain e1-f1 ends at g1, so both d1 and g1 are legal.
    """
    dark = (1 << 0) | (1 << 4)
    light = (1 << 1) | (1 << 2) | (1 << 5)
    legal = legal_moves(dark, light, DARK)
    assert (1 << 3) & legal, "d1 missing (end of the long chain)"
    assert (1 << 6) & legal, "g1 missing (end of the short chain)"


def test_format_parse_roundtrip():
    for idx in range(64):
        bit = 1 << idx
        assert parse_move(format_move(bit)) == bit
    assert parse_move("pass") == PASS
    assert parse_move("") == PASS
    with pytest.raises(ValueError):
        parse_move("zz")


def test_reversi_dataclass_human_like_play():
    g = Reversi.start()
    assert g.legal() != 0
    g.play(g.legal_list()[0])
    assert g.color == LIGHT
    assert g.legal() != 0
    assert not g.is_over()


def test_game_eventually_ends_between_optimal_and_random():
    rng = random.Random(123)
    dark, light, color = start()
    while not game_over(dark, light):
        if legal_moves(dark, light, color) == 0:
            color = -color
            continue
        dark, light = play(dark, light, color, random_move(dark, light, color, rng))
        color = -color
    assert game_over(dark, light)
    assert outcome(dark, light) in (DARK, LIGHT, 0)
