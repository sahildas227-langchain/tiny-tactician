"""Round-robin tournament between the agent and the reference engines.

Every agent gets the same wall-clock budget per move, so the comparison is
"what plays better in one second" rather than "who runs more simulations".
The two sides alternate between dark and light so that the first-move
advantage cannot decide the result.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from tiny_tactician.games.reversi import DARK, game_over, legal_moves, play, popcount, start

__all__ = ["play_game", "round_robin", "TournamentResult", "elo_from_score"]


def play_game(dark_agent, light_agent) -> tuple[int, int]:
    """Play one game; return the final ``(dark, light)`` disc counts."""
    dark, light, color = start()
    while not game_over(dark, light):
        if legal_moves(dark, light, color) == 0:
            color = -color
            continue
        agent = dark_agent if color == DARK else light_agent
        move = agent.move(dark, light, color)
        if not move or not (legal_moves(dark, light, color) & move):
            # An agent that returns nothing legal loses the move.
            bits = [1 << i for i in range(64) if legal_moves(dark, light, color) >> i & 1]
            move = random.choice(bits) if bits else 0
        if move:
            dark, light = play(dark, light, color, move)
        color = -color
    return popcount(dark), popcount(light)


@dataclass
class TournamentResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    disc_total: int = 0
    opponents: dict[str, tuple[int, int, int]] = field(default_factory=dict)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        """0.5 for a draw, 1 for a win - the standard tournament score."""
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def record_against(self, opponent: str, won: bool, drew: bool, disc: int) -> None:
        current = self.opponents.setdefault(opponent, [0, 0, 0])
        if drew:
            current[1] += 1
        elif won:
            current[0] += 1
        else:
            current[2] += 1
        self.wins += 1 if won else 0
        self.losses += 0 if won or drew else 1
        self.draws += 1 if drew else 0
        self.disc_total += disc


def elo_from_score(score: float) -> float:
    """Rating difference implied by a score against an equal opponent."""
    if score <= 0:
        return -800.0
    if score >= 1:
        return 800.0
    return -400.0 * math.log10(1 / score - 1)


def round_robin(agents: dict, games_per_pair: int = 20, seed: int = 0) -> dict[str, TournamentResult]:
    """Play every ordered pair of agents, alternating the side each one starts on."""
    random.seed(seed)
    results = {name: TournamentResult() for name in agents}
    names = list(agents)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for game in range(games_per_pair):
                first, second = (a, b) if game % 2 == 0 else (b, a)
                dark_score, light_score = play_game(agents[first], agents[second])
                a_dark = game % 2 == 0
                a_score = dark_score if a_dark else light_score
                b_score = light_score if a_dark else dark_score
                for name, score, other in ((a, a_score, b_score), (b, b_score, a_score)):
                    drew = score == other
                    won = score > other
                    results[name].record_against(
                        other_name := (b if name == a else a),
                        won=won,
                        drew=drew,
                        disc=score - other,
                    )
                    del other_name
    return results
