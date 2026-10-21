"""The board-game agent: small-model reasoning + Monte-Carlo Tree Search.

Division of labour
------------------
* The **search** (MCTS/PUCT) supplies the play strength.
* The **small model** explains the chosen move in the language of Othello
  strategy, and contributes a prior during the opening phase - the only part
  of the game where its memorised knowledge holds up.

The model is consulted once per move (not once per node) so that a 360M model
on a laptop stays fast enough to play interactively.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from tiny_tactician.games.mcts import search
from tiny_tactician.games.reversi import (
    DARK,
    format_move,
    heuristic_prior,
    legal_moves,
)
from tiny_tactician.models.strategy import StrategyModel, blend_priors

__all__ = ["BoardAgent", "MoveChoice"]


@dataclass
class MoveChoice:
    move: int
    policy: dict[int, float] = field(default_factory=dict)
    prior: dict[int, float] = field(default_factory=dict)
    model_prior: dict[int, float] = field(default_factory=dict)
    comment: str = ""

    @property
    def square(self) -> str:
        return format_move(self.move)

    def top_moves(self, n: int = 3) -> list[tuple[str, float]]:
        ranked = sorted(self.policy.items(), key=lambda kv: -kv[1])[:n]
        return [(format_move(move), prob) for move, prob in ranked]


@dataclass
class BoardAgent:
    """Plays Othello by combining a small language model with MCTS.

    The model's prior is only consulted for the opening phase
    (``model_prior_plies``).  That is where a 360M model's memorised Othello
    knowledge is actually reliable - it knows the standard openings and their
    relative strengths - while its mid-game judgement is not.  Letting the
    search take over afterwards is a measured decision, not an assumption; see
    the ``ablate`` command.
    """

    strategy: StrategyModel | None = None
    n_sims: int = 400
    c_puct: float = 1.41
    seed: int = 0
    use_model_prior: bool = True
    model_weight: float = 0.5
    model_prior_plies: int = 8
    explain: bool = True
    time_budget: float = 0.0

    def _plies_played(self, dark: int, light: int, plies: int | None = None) -> int:
        """Number of moves played so far (the opening position counts as zero).

        The caller can pass the exact count; otherwise it is estimated from the
        number of discs on the board, which is a lower bound because a single
        move can flip any number of discs.
        """
        if plies is not None:
            return plies
        return ((dark | light).bit_count() - 4) // 2

    def _root_prior(self, dark: int, light: int, color: int, plies: int | None = None) -> tuple[dict[int, float], dict[int, float]]:
        """Model shortlist for the current position plus the blended prior."""
        heuristic = heuristic_prior(dark, light, color)
        if not self.use_model_prior or self.strategy is None:
            return {}, heuristic
        if self._plies_played(dark, light, plies) > self.model_prior_plies:
            return {}, heuristic
        model_prior = self.strategy.propose_moves(dark, light, color)
        return model_prior, blend_priors(model_prior, heuristic, self.model_weight)

    def choose_move(
        self,
        dark: int,
        light: int,
        color: int,
        comment: bool | None = None,
        plies: int | None = None,
    ) -> MoveChoice:
        """Pick a move for ``color`` and (optionally) explain it."""
        # The budget starts before the model is consulted, so the model costs
        # the agent time out of the same allowance as the search.
        deadline = (time.monotonic() + self.time_budget) if self.time_budget > 0 else None
        model_prior, prior = self._root_prior(dark, light, color, plies)
        root_position = (dark, light, color)

        def prior_fn(d: int, l: int, c: int) -> dict[int, float]:
            # The model prior is only meaningful (and only affordable) for the
            # position being decided; deeper nodes fall back to the heuristic.
            if (d, l, c) == root_position:
                return prior
            return heuristic_prior(d, l, c)

        policy, best, _root = search(
            dark,
            light,
            color,
            n_sims=self.n_sims,
            prior_fn=prior_fn,
            c_puct=self.c_puct,
            seed=self.seed,
            deadline=deadline,
        )
        choice = MoveChoice(move=best, policy=policy, prior=prior, model_prior=model_prior)
        if (self.explain if comment is None else comment) and self.strategy is not None and best:
            choice.comment = self.strategy.explain(dark, light, color, best)
        return choice

    def legal_moves(self, dark: int, light: int, color: int) -> list[int]:
        return [1 << i for i in range(64) if legal_moves(dark, light, color) >> i & 1]

    def uses_model_prior(self, dark: int, light: int, plies: int | None = None) -> bool:
        """Whether the model is consulted for this position."""
        return (
            self.use_model_prior
            and self.strategy is not None
            and self._plies_played(dark, light, plies) <= self.model_prior_plies
        )

    @staticmethod
    def side_name(color: int) -> str:
        return "dark (X)" if color == DARK else "light (O)"
