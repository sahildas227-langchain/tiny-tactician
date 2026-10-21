"""PUCT Monte-Carlo Tree Search over the bitboard Reversi engine.

The search provides the play *strength*: it explores the game tree while using a
prior distribution (``prior_fn``) to guide expansion.  In this project the prior
is supplied by the 360M SmolLM2 model (its strategic positional preferences),
with the classic corner/edge weight table as a robust fallback.  Rollouts use
heuristic-guided playouts.  Everything runs over pure-Python bitboards, so
thousands of simulations complete in a couple of seconds with no third-party
ML stack.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from tiny_tactician.games.reversi import (
    DARK,
    PASS,
    WEIGHTS,
    _set_bits,
    game_over,
    legal_moves,
    outcome,
    play,
)

__all__ = ["Node", "search", "rollout"]

_C_PUCT = 1.41


@dataclass
class Node:
    dark: int
    light: int
    color: int
    parent: Node | None = None
    move: int = 0
    children: dict[int, Node] = field(default_factory=dict)
    P: dict[int, float] = field(default_factory=dict)
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    expanded: bool = False
    terminal: bool = False

    def expand(self, prior_fn=None, rng=None, root_noise=False, noise_eps=0.25):
        if self.terminal:
            return
        legal_bit = legal_moves(self.dark, self.light, self.color)
        if legal_bit == 0:
            self.terminal = True
            return
        legal = [1 << i for i in _set_bits(legal_bit)]
        prior = prior_fn(self.dark, self.light, self.color) if prior_fn else {}
        p = {m: float(prior.get(m, 0.0)) for m in legal}
        total = sum(p.values())
        if total <= 0:
            u = 1.0 / len(legal)
            p = {m: u for m in legal}
        else:
            p = {m: v / total for m, v in p.items()}
        if root_noise and rng is not None and len(legal) > 1:
            p = self._dirichlet(p, legal, rng, noise_eps)
        for m in legal:
            nd, nt = play(self.dark, self.light, self.color, m)
            self.children[m] = Node(nd, nt, -self.color, parent=self, move=m)
            self.P[m] = p[m]
        self.expanded = True

    @staticmethod
    def _dirichlet(p, legal, rng, eps):
        gammas = [rng.gammavariate(0.03, 1.0) for _ in legal]
        total = sum(gammas)
        if total <= 0:
            noise = {m: 1.0 / len(legal) for m in legal}
        else:
            noise = {m: g / total for g, m in zip(gammas, legal)}
        return {m: (1 - eps) * p[m] + eps * noise[m] for m in legal}


def select_child(node: Node, c_puct: float) -> Node | None:
    """PUCT child selection.  ``Q`` is stored from the child's own side's point
    of view, so it must be negated to score the move for the parent."""
    if not node.children:
        return None
    sqrt_n = math.sqrt(max(node.N, 1))
    best, best_score = None, float("-inf")
    for m, ch in node.children.items():
        if ch.N == 0:
            score = 1e9 + node.P.get(m, 0.0)  # unvisited first, then by prior
        else:
            score = -ch.Q + c_puct * node.P.get(m, 0.0) * sqrt_n / (1 + ch.N)
        if score > best_score:
            best_score, best = score, ch
    return best


def backprop(node: Node | None, value: float) -> None:
    """Propagate a value up the tree, flipping sign at every level."""
    while node is not None:
        node.N += 1
        node.W += value
        node.Q = node.W / node.N
        value = -value
        node = node.parent


def _weighted_move(dark: int, light: int, color: int, rng: random.Random) -> int:
    legal_bit = legal_moves(dark, light, color)
    bits = [1 << i for i in _set_bits(legal_bit)]
    if not bits:
        return PASS
    if len(bits) == 1:
        return bits[0]
    weights = [WEIGHTS[(b.bit_length() - 1) // 8][(b.bit_length() - 1) % 8] for b in bits]
    top = max(weights)
    exps = [math.exp(w - top) for w in weights]
    z = sum(exps)
    acc = 0.0
    r = rng.random() * z
    for b, e in zip(bits, exps):
        acc += e
        if r <= acc:
            return b
    return bits[-1]


def _rand_move(legal_bit: int, rng: random.Random) -> int:
    bits = [1 << i for i in _set_bits(legal_bit)]
    return rng.choice(bits) if bits else PASS


def rollout(dark: int, light: int, color: int, rng: random.Random, heuristic_weight: float = 0.5) -> int:
    """Simulate to a terminal state from ``color`` to move; returns the
    DARK-perspective outcome (DARK=+1, LIGHT=-1, draw=0)."""
    while not game_over(dark, light):
        legal_bit = legal_moves(dark, light, color)
        if legal_bit == 0:
            color = -color
            continue
        move = _weighted_move(dark, light, color, rng) if rng.random() < heuristic_weight else _rand_move(legal_bit, rng)
        dark, light = play(dark, light, color, move)
        color = -color
    return outcome(dark, light)


def search(
    dark: int,
    light: int,
    color: int,
    n_sims: int = 1000,
    prior_fn=None,
    c_puct: float = _C_PUCT,
    seed: int = 0,
    root_noise: bool = False,
    noise_eps: float = 0.25,
) -> tuple[dict[int, float], int, Node]:
    """Run a PUCT search; return ``(policy, best_move, root)``.

    ``policy`` is the normalized visit-count distribution over legal moves and
    ``best_move`` is the most visited move (``PASS`` if the position is
    terminal).
    """
    rng = random.Random(seed)
    root = Node(dark, light, color)
    root.expand(prior_fn, rng=rng, root_noise=root_noise, noise_eps=noise_eps)
    for _ in range(n_sims):
        node = root
        while node.expanded and not node.terminal:
            child = select_child(node, c_puct)
            if child is None:
                break
            node = child
        leaf = node
        leaf.expand(prior_fn, rng=rng)
        value = rollout(leaf.dark, leaf.light, leaf.color, rng)
        node_value = value if leaf.color == DARK else -value
        backprop(leaf, node_value)
    policy: dict[int, float] = {}
    if root.children:
        total = sum(c.N for c in root.children.values())
        if total > 0:
            policy = {m: c.N / total for m, c in root.children.items()}
    best = max(root.children, key=lambda m: root.children[m].N) if root.children else PASS
    return policy, best, root
