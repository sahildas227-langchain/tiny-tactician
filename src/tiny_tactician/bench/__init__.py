"""Benchmarks: reference engines, round-robin tournaments and throughput."""

from tiny_tactician.bench.baselines import (
    AlphaBetaAgent,
    GreedyAgent,
    MuctAgent,
    RandomAgent,
    build_agents,
)
from tiny_tactician.bench.evaluate import evaluate
from tiny_tactician.bench.tournament import (
    TournamentResult,
    elo_from_score,
    play_game,
    round_robin,
)

__all__ = [
    "AlphaBetaAgent",
    "GreedyAgent",
    "MuctAgent",
    "RandomAgent",
    "TournamentResult",
    "build_agents",
    "elo_from_score",
    "evaluate",
    "play_game",
    "round_robin",
]
