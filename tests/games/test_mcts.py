"""Tests for the PUCT MCTS search."""

from __future__ import annotations

import random

import pytest

from tiny_tactician.games.mcts import rollout, search
from tiny_tactician.games.reversi import DARK, LIGHT, game_over, legal_moves, play, random_move, start


def _bits(x):
    i = 0
    while x:
        if x & 1:
            yield i
        x >>= 1
        i += 1


def test_search_returns_legal_move():
    dark, light, color = start()
    policy, best, _root = search(dark, light, color, n_sims=80, seed=0)
    lm = legal_moves(dark, light, color)
    assert best != 0 and (lm & best) == best
    # policy mass is only on legal moves and sums to ~1
    assert sum(policy.values()) == pytest.approx(1.0, abs=1e-9)
    for m in policy:
        assert (lm & m) == m


def test_search_is_deterministic_with_seed():
    dark, light, color = start()
    _, best1, _ = search(dark, light, color, n_sims=80, seed=7)
    _, best2, _ = search(dark, light, color, n_sims=80, seed=7)
    assert best1 == best2


def test_search_respects_prior():
    """A prior that puts all its mass on one legal move must steer the first
    expansion there, so the search starts by exploring that move."""
    dark, light, color = start()
    target = 1 << 19
    assert target & legal_moves(dark, light, color), "target must be a legal move"
    policy, best, _root = search(dark, light, color, n_sims=1, prior_fn=lambda d, l, c: {target: 1.0}, seed=3)
    assert best == target
    assert policy[target] == 1.0


def test_rollout_returns_valid_outcome():
    dark, light, color = start()
    val = rollout(dark, light, color, random.Random(0))
    assert val in (DARK, LIGHT, 0)


def test_full_game_completes_mcts_vs_random():
    dark, light, color = start()
    rng = random.Random(42)
    moves = 0
    while not game_over(dark, light):
        if legal_moves(dark, light, color) == 0:
            color = -color
            continue
        if color == DARK:
            _, m, _ = search(dark, light, DARK, n_sims=40, seed=rng.random())
            dark, light = play(dark, light, DARK, m)
        else:
            dark, light = play(dark, light, LIGHT, random_move(dark, light, LIGHT, rng))
        color = -color
        moves += 1
    assert moves <= 130  # sanity: didn't loop forever
    assert game_over(dark, light)


@pytest.mark.slow
def test_mcts_beats_random_over_few_games():
    wins = 0
    rng = random.Random(0)
    for _game in range(8):
        dark, light, color = start()
        while not game_over(dark, light):
            if legal_moves(dark, light, color) == 0:
                color = -color
                continue
            if color == DARK:
                _, m, _ = search(dark, light, DARK, n_sims=150, seed=rng.random())
                dark, light = play(dark, light, DARK, m)
            else:
                dark, light = play(dark, light, LIGHT, random_move(dark, light, LIGHT, rng))
            color = -color
        if dark.bit_count() > light.bit_count():
            wins += 1
    assert wins >= 6  # MCTS as dark should beat random play most of the time
