"""Tests for the agent wiring: model prior, search and commentary."""

from __future__ import annotations

import pytest

from tiny_tactician.agents.board_agent import BoardAgent
from tiny_tactician.environments.reversi_env import GameSession
from tiny_tactician.games.reversi import DARK, LIGHT, legal_moves, start


class FakeStrategy:
    """Stands in for the language model with a fixed, known preference."""

    def __init__(self, preferred: int):
        self.preferred = preferred
        self.explained: list[int] = []

    def propose_moves(self, dark, light, color):
        legal = legal_moves(dark, light, color)
        if not legal & self.preferred:
            return {}
        return {self.preferred: 1.0}

    def explain(self, dark, light, color, move):
        self.explained.append(move)
        return "because the corner is worth the most."


def test_agent_uses_model_prior():
    dark, light, color = start()
    preferred = 1 << 19  # d3
    agent = BoardAgent(FakeStrategy(preferred), n_sims=50, seed=0)
    choice = agent.choose_move(dark, light, color)
    assert choice.move == preferred
    assert choice.model_prior == {preferred: 1.0}
    assert choice.comment


def test_agent_without_model_still_plays_legal_moves():
    dark, light, color = start()
    agent = BoardAgent(None, n_sims=50, seed=0, use_model_prior=False)
    choice = agent.choose_move(dark, light, color)
    assert legal_moves(dark, light, color) & choice.move
    assert choice.comment == ""


def test_model_prior_is_only_used_during_the_opening():
    """The model's knowledge only holds up in the opening, so the agent stops
    consulting it once the game is under way."""
    dark, light, color = start()
    strategy = FakeStrategy(1 << 19)
    agent = BoardAgent(strategy, n_sims=20, seed=0, model_prior_plies=8)
    assert agent.uses_model_prior(dark, light, plies=0)
    assert not agent.uses_model_prior(dark, light, plies=9)


def test_plies_played_is_tracked_exactly_when_given():
    dark, light, color = start()
    agent = BoardAgent(None, n_sims=10, seed=0)
    assert agent._plies_played(dark, light, plies=0) == 0
    assert agent._plies_played(dark, light, plies=17) == 17
    # without an exact count the disc count gives a lower bound
    assert agent._plies_played(dark, light) == 0


def test_agent_survives_a_model_that_understands_nothing():
    """An unusable model output must not break the agent."""

    class Useless(FakeStrategy):
        def propose_moves(self, dark, light, color):
            return {}

    dark, light, color = start()
    agent = BoardAgent(Useless(0), n_sims=50, seed=0)
    choice = agent.choose_move(dark, light, color)
    assert legal_moves(dark, light, color) & choice.move


def test_game_session_rejects_illegal_moves():
    session = GameSession(human_color=LIGHT, agent_color=DARK)
    with pytest.raises(ValueError):
        session.parse_human_move("a1")  # occupied
    with pytest.raises(ValueError):
        session.parse_human_move("z9")  # not a square
    with pytest.raises(ValueError):
        session.parse_human_move("pass")  # but a legal move exists
    assert session.parse_human_move("d3") == 1 << 19


def test_game_session_tracks_the_game_to_the_end():
    session = GameSession(human_color=DARK, agent_color=LIGHT)
    while not session.is_over():
        if session.legal() == 0:
            session.reversi.play(0)
            continue
        session.play(session.legal_list()[0])
    dark, light, result = session.result()
    assert dark + light == 64
    assert result in (DARK, LIGHT, 0)
