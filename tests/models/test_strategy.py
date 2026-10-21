"""Tests for the small-model strategy layer.

The model itself is not loaded here: these tests cover the parts that turn raw
model output into something the search can trust - parsing, log-probability
priors and blending.
"""

from __future__ import annotations

import math

import pytest

from tiny_tactician.games.reversi import DARK, format_move, legal_moves, start
from tiny_tactician.models.strategy import (
    blend_priors,
    logprob_prior,
    parse_ranked_moves,
)


def test_parse_ranked_moves_keeps_order_and_legality():
    legal = legal_moves(*start()[:2], DARK)
    parsed = parse_ranked_moves("My best moves are c4, then d3 and f5.", legal)
    assert set(parsed) == {1 << 26, 1 << 19, 1 << 37}  # c4, d3, f5
    assert parsed[1 << 26] > parsed[1 << 19] > parsed[1 << 37]
    assert sum(parsed.values()) == pytest.approx(1.0)


def test_parse_ranked_moves_rejects_illegal_squares():
    legal = legal_moves(*start()[:2], DARK)
    parsed = parse_ranked_moves("z9, c3, d3", legal)
    assert set(parsed) == {1 << 19}  # only d3 is legal


def test_parse_ranked_moves_ignores_duplicates():
    legal = legal_moves(*start()[:2], DARK)
    parsed = parse_ranked_moves("d3 d3 c4", legal)
    assert len(parsed) == 2


def test_logprob_prior_normalises_over_options():
    logprobs = [
        {"token": "1", "logprob": -0.5},
        {"token": "2", "logprob": -2.5},
        {"token": "3", "logprob": -2.0},
        {"token": "banana", "logprob": -0.1},
    ]
    prior = logprob_prior(logprobs, 4)
    assert set(prior) == {0, 1, 2}  # non-numeric tokens ignored
    assert sum(prior.values()) == pytest.approx(1.0)
    # option 1 has the best logprob, option 3 the second best, option 2 the worst
    assert prior[0] > prior[2] > prior[1]
    assert prior[0] == pytest.approx(math.exp(-0.5) / (math.exp(-0.5) + math.exp(-2.0) + math.exp(-2.5)))


def test_logprob_prior_ignores_out_of_range_options():
    prior = logprob_prior([{"token": "9", "logprob": -0.1}], 4)
    assert prior == {}


def test_logprob_prior_empty_input():
    assert logprob_prior([], 4) == {}
    assert logprob_prior([{"token": "x", "logprob": 0.0}], 4) == {}


def test_blend_priors_keeps_model_signal_and_legal_support():
    legal = legal_moves(*start()[:2], DARK)
    heuristic = {1 << i: 1.0 / 4 for i in range(64) if legal >> i & 1}
    model = {1 << 19: 1.0}
    blended = blend_priors(model, heuristic, weight=0.5)
    assert set(blended) == set(heuristic)
    assert sum(blended.values()) == pytest.approx(1.0)
    assert blended[1 << 19] > blended[1 << 26]


def test_blend_priors_falls_back_to_heuristic():
    legal = legal_moves(*start()[:2], DARK)
    heuristic = {1 << i: 1.0 / 4 for i in range(64) if legal >> i & 1}
    assert blend_priors({}, heuristic, weight=0.5) == heuristic
    assert blend_priors({1 << 19: 1.0}, {}, weight=0.5) == {1 << 19: 1.0}


def test_openers_are_the_standard_four():
    """The opening must be the classic d3/c4/f5/e6, matching how Othello is
    taught - the model's own knowledge depends on this convention."""
    dark, light, color = start()
    legal = legal_moves(dark, light, color)
    assert sorted(format_move(1 << i) for i in range(64) if legal >> i & 1) == ["c4", "d3", "e6", "f5"]
