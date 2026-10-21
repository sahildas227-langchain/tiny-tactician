"""Baseline Othello engines used to measure what TinyTactician adds.

Four reference points, from weakest to strongest:

``random``    uniform random legal move.
``greedy``    one ply of the static evaluation.
``alpha-beta``  iterative-deepening negamax with the static evaluation - the
              classic way to write a strong Othello engine.
``muct``      Monte Carlo UCT with progressive bias - the canonical
              Monte-Carlo approach for Othello (Ito, 2010).

Every agent exposes the same interface::

    move(dark: int, light: int, color: int) -> int

returning a single-bit move, and is given the same wall-clock budget per move
so that the comparison is fair rather than a comparison of simulation counts.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from tiny_tactician.bench.evaluate import evaluate
from tiny_tactician.games.mcts import weighted_move
from tiny_tactician.games.reversi import (
    DARK,
    PASS,
    WEIGHTS,
    _set_bits,
    flips,
    game_over,
    legal_moves,
    play,
)

__all__ = ["RandomAgent", "GreedyAgent", "AlphaBetaAgent", "MuctAgent", "build_agents"]

_INF = 10**9
_TERMINAL_SCALE = 1000


class RandomAgent:
    name = "random"

    def __init__(self, time_budget: float = 1.0, seed: int = 0):
        self.time_budget = time_budget
        self.rng = random.Random(seed)

    def move(self, dark: int, light: int, color: int) -> int:
        bits = [1 << i for i in _set_bits(legal_moves(dark, light, color))]
        return self.rng.choice(bits) if bits else PASS


class GreedyAgent:
    name = "greedy"

    def __init__(self, time_budget: float = 1.0, seed: int = 0):
        self.time_budget = time_budget

    def move(self, dark: int, light: int, color: int) -> int:
        legal = legal_moves(dark, light, color)
        best, best_score = PASS, -math.inf
        for move in [1 << i for i in _set_bits(legal)]:
            nd, nl = play(dark, light, color, move)
            score = evaluate(nd, nl, -color)
            if score > best_score:
                best, best_score = move, score
        return best


class AlphaBetaAgent:
    """Iterative-deepening negamax with a static evaluation."""

    name = "alpha-beta"

    def __init__(self, time_budget: float = 1.0, max_depth: int = 20, seed: int = 0):
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.rng = random.Random(seed)
        self.nodes = 0
        self.expired = False

    def move(self, dark: int, light: int, color: int) -> int:
        deadline = time.monotonic() + self.time_budget
        legal = [1 << i for i in _set_bits(legal_moves(dark, light, color))]
        if not legal:
            return PASS
        best = legal[0]
        for depth in range(1, self.max_depth + 1):
            self.expired = False
            score, move = self._negamax(dark, light, color, depth, -_INF, _INF, deadline)
            if not self.expired and move:
                best = move
            if self.expired or time.monotonic() >= deadline:
                break
        return best

    def _ordered(self, dark: int, light: int, color: int, legal: int) -> list[int]:
        """Corners first, then moves that flip the most discs."""
        moves = [1 << i for i in _set_bits(legal)]

        def key(move: int):
            idx = move.bit_length() - 1
            row, col = divmod(idx, 8)
            is_corner = row in (0, 7) and col in (0, 7)
            return (not is_corner, -flips(dark, light, color, move).bit_count(), -WEIGHTS[row][col])

        return sorted(moves, key=key)

    def _negamax(self, dark: int, light: int, color: int, depth: int, alpha: int, beta: int, deadline: float):
        self.nodes += 1
        if self.nodes % 512 == 0 and time.monotonic() >= deadline:
            self.expired = True
            return 0, PASS
        legal = legal_moves(dark, light, color)
        if legal == 0:
            if legal_moves(dark, light, -color) == 0:
                # Terminal: the side to move has lost more discs than the opponent.
                mine = (dark if color == DARK else light).bit_count()
                theirs = (light if color == DARK else dark).bit_count()
                return (mine - theirs) * _TERMINAL_SCALE, PASS
            # Forced pass: the opponent moves, so the sign flips.
            score, _ = self._negamax(dark, light, -color, depth, -beta, -alpha, deadline)
            return -score, PASS
        if depth == 0:
            return evaluate(dark, light, color), PASS
        best_score, best_move = -_INF, PASS
        for move in self._ordered(dark, light, color, legal):
            nd, nl = play(dark, light, color, move)
            score, _ = self._negamax(nd, nl, -color, depth - 1, -beta, -alpha, deadline)
            score = -score
            if score > best_score:
                best_score, best_move = score, move
            if best_score > alpha:
                alpha = best_score
            if alpha >= beta:
                break
        return best_score, best_move


@dataclass
class _MuctNode:
    dark: int
    light: int
    color: int
    parent: _MuctNode | None = None
    move: int = 0
    untried: list[int] = field(default_factory=list)
    children: dict[int, _MuctNode] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0


class MuctAgent:
    """Monte Carlo UCT with progressive bias (Ito, 2010)."""

    name = "muct"

    def __init__(self, time_budget: float = 1.0, c: float = 0.7, bias: float = 40.0, seed: int = 0, heuristic_weight: float = 0.5):
        self.time_budget = time_budget
        self.c = c
        self.bias = bias
        self.heuristic_weight = heuristic_weight
        self.rng = random.Random(seed)
        self.playouts = 0

    def move(self, dark: int, light: int, color: int) -> int:
        deadline = time.monotonic() + self.time_budget
        legal = legal_moves(dark, light, color)
        if legal == 0:
            return PASS
        root = _MuctNode(dark, light, color)
        root.untried = [1 << i for i in _set_bits(legal)]
        best = root.untried[0]
        best_visits = -1
        while time.monotonic() < deadline:
            node = self._select(root)
            if node.untried:
                move = node.untried.pop()
                nd, nl = play(node.dark, node.light, node.color, move)
                child = _MuctNode(nd, nl, -node.color, parent=node, move=move)
                child.untried = [1 << i for i in _set_bits(legal_moves(nd, nl, -node.color))]
                node.children[move] = child
                node = child
            reward = self._simulate(node)
            self._backpropagate(node, reward)
            for move, child in root.children.items():
                if child.visits > best_visits:
                    best_visits, best = child.visits, move
        return best

    def _select(self, node: _MuctNode) -> _MuctNode:
        while node.children and not node.untried:
            log_parent = math.log(node.visits + 1)
            node = max(
                node.children.values(),
                key=lambda child: self._uct(node, child, log_parent),
            )
        return node

    def _uct(self, parent: _MuctNode, child: _MuctNode, log_parent: float) -> float:
        if child.visits == 0:
            return math.inf
        # A node's value is stored from its own point of view, and a child's
        # point of view is the opponent of the parent's, so it is flipped here.
        exploitation = 1.0 - child.value / child.visits
        exploration = self.c * math.sqrt(log_parent / child.visits)
        progressive = self.bias * child.visits / (parent.visits + child.visits * self.bias + 1)
        return exploitation + exploration + progressive

    def _simulate(self, node: _MuctNode) -> float:
        """Random play-out; returns 1.0 for a win for the node's side."""
        self.playouts += 1
        dark, light, color = node.dark, node.light, node.color
        while not game_over(dark, light):
            legal = legal_moves(dark, light, color)
            if legal == 0:
                color = -color
                continue
            if self.rng.random() < self.heuristic_weight:
                move = weighted_move(dark, light, color, self.rng)
            else:
                bits = [1 << i for i in _set_bits(legal)]
                move = self.rng.choice(bits)
            dark, light = play(dark, light, color, move)
            color = -color
        mine = node.dark.bit_count() if node.color == DARK else node.light.bit_count()
        theirs = node.light.bit_count() if node.color == DARK else node.dark.bit_count()
        if mine > theirs:
            return 1.0
        if mine < theirs:
            return 0.0
        return 0.5

    def _backpropagate(self, node: _MuctNode, reward: float) -> None:
        current = node
        value = reward
        while current is not None:
            current.visits += 1
            current.value += value
            value = 1.0 - value
            current = current.parent


def build_agents(time_budget: float, seed: int = 0) -> dict:
    """The baseline agents, keyed by name."""
    return {
        "random": RandomAgent(time_budget, seed),
        "greedy": GreedyAgent(time_budget, seed),
        "alpha-beta": AlphaBetaAgent(time_budget, seed=seed),
        "muct": MuctAgent(time_budget, seed=seed),
    }
