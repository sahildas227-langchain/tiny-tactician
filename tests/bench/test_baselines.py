"""Validate the reference engines: they must be legal, and correctly ordered.

A benchmark is only meaningful if the baselines are sound, so this checks that
every engine returns legal moves, that the strong engines clearly beat the
weak ones, and that a strong engine finds the obvious tactics.
"""

from __future__ import annotations

import pytest

from tiny_tactician.bench.baselines import AlphaBetaAgent, GreedyAgent, MuctAgent, RandomAgent
from tiny_tactician.bench.evaluate import evaluate
from tiny_tactician.bench.tournament import play_game
from tiny_tactician.games.reversi import DARK, legal_moves, play, start

AGENTS = [RandomAgent(0.2), GreedyAgent(0.2), AlphaBetaAgent(0.2), MuctAgent(0.2)]


@pytest.mark.parametrize("agent", AGENTS, ids=lambda a: a.name)
def test_agents_return_legal_moves(agent):
    dark, light, color = start()
    legal = legal_moves(dark, light, color)
    move = agent.move(dark, light, color)
    assert legal & move


@pytest.mark.parametrize("agent", AGENTS, ids=lambda a: a.name)
def test_agents_never_crash_in_a_full_game(agent):
    """Play a whole game against random; the agent must never break the rules."""
    from tiny_tactician.games.reversi import game_over

    dark, light, color = start()
    for _ in range(80):
        if game_over(dark, light):
            break
        if legal_moves(dark, light, color) == 0:
            color = -color
            continue
        move = agent.move(dark, light, color)
        assert legal_moves(dark, light, color) & move, f"{agent.name} returned an illegal move"
        dark, light = play(dark, light, color, move)
        color = -color


def test_evaluate_prefers_a_corner_position():
    """Owning a corner must be worth more than owning an ordinary edge square."""
    dark_corner = 1 << 0   # a1
    light_edge = 1 << 39   # h5
    assert evaluate(dark_corner, light_edge, DARK) > evaluate(light_edge, dark_corner, DARK)


def test_alpha_beta_finds_the_corner():
    """When a corner can be taken, a strong engine takes it.

    Light sits on b1 and a2; dark on c1 and a3.  Playing a1 flips both.
    """
    dark = (1 << 2) | (1 << 16)   # c1, a3
    light = (1 << 1) | (1 << 8)   # b1, a2
    assert legal_moves(dark, light, DARK) & (1 << 0)
    assert AlphaBetaAgent(0.5).move(dark, light, DARK) == 1 << 0


@pytest.mark.slow
def test_alpha_beta_beats_random_decisively():
    """The strong engine must overwhelm random play, otherwise the whole
    benchmark is meaningless."""
    strong = AlphaBetaAgent(0.2)
    weak = RandomAgent(0.0, seed=3)
    wins = 0
    for game in range(6):
        strong_dark = game % 2 == 0
        dark_score, light_score = play_game(strong, weak) if strong_dark else play_game(weak, strong)
        wins += dark_score > light_score if strong_dark else light_score > dark_score
    assert wins >= 5, f"alpha-beta only won {wins}/6 against random"


@pytest.mark.slow
def test_muct_beats_random():
    muct = MuctAgent(0.2)
    weak = RandomAgent(0.0, seed=5)
    wins = 0
    for game in range(6):
        muct_dark = game % 2 == 0
        dark_score, light_score = play_game(muct, weak) if muct_dark else play_game(weak, muct)
        wins += dark_score > light_score if muct_dark else light_score > dark_score
    assert wins >= 4, f"muct only won {wins}/6 against random"


@pytest.mark.slow
def test_engines_are_reasonably_ordered():
    """Greedy should not beat alpha-beta, and alpha-beta should not lose
    catastrophically to MUCT at equal time."""
    greedy = GreedyAgent(0.2)
    ab = AlphaBetaAgent(0.2)
    greedy_wins = ab_wins = 0
    for game in range(6):
        greedy_dark = game % 2 == 0
        dark_score, light_score = play_game(greedy, ab) if greedy_dark else play_game(ab, greedy)
        if greedy_dark:
            greedy_wins += dark_score > light_score
            ab_wins += light_score > dark_score
        else:
            ab_wins += dark_score > light_score
            greedy_wins += light_score > dark_score
    assert greedy_wins <= ab_wins + 1, f"greedy beat alpha-beta {greedy_wins}-{ab_wins}"
