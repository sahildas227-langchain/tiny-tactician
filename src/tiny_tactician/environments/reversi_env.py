"""Game environment: the board, the move loop and the terminal presentation."""

from __future__ import annotations

from dataclasses import dataclass, field

from tiny_tactician.games.reversi import (
    DARK,
    LIGHT,
    PASS,
    Reversi,
    format_move,
    legal_moves,
    outcome,
    parse_move,
    score,
)

__all__ = ["GameSession"]


@dataclass
class GameSession:
    """Holds the board state and drives one game to completion."""

    human_color: int = LIGHT
    agent_color: int = DARK
    reversi: Reversi = field(default_factory=Reversi.start)
    move_log: list[tuple[int, int]] = field(default_factory=list)
    agent_comments: list[str] = field(default_factory=list)

    @property
    def color(self) -> int:
        return self.reversi.color

    @property
    def plies(self) -> int:
        """How many moves have been played in this game."""
        return len(self.move_log)

    def legal(self) -> int:
        return legal_moves(self.reversi.dark, self.reversi.light, self.reversi.color)

    def legal_squares(self) -> list[str]:
        return [format_move(1 << i) for i in range(64) if self.legal() >> i & 1]

    def legal_list(self) -> list[int]:
        return [1 << i for i in range(64) if self.legal() >> i & 1]

    def is_over(self) -> bool:
        return self.reversi.is_over()

    def play(self, move: int) -> None:
        self.reversi.play(move)
        self.move_log.append((self.reversi.color, move))

    def parse_human_move(self, text: str) -> int:
        """Parse and validate a human move; raises ValueError if illegal."""
        move = parse_move(text)
        if move == PASS:
            if self.legal() != 0:
                raise ValueError("you have a legal move, you may not pass")
            return PASS
        if not (self.legal() & move):
            raise ValueError(f"{format_move(move)} is not a legal move")
        return move

    def result(self) -> tuple[int, int, int]:
        """Return ``(dark, light, outcome)``."""
        dark, light = score(self.reversi.dark, self.reversi.light)
        return dark, light, outcome(self.reversi.dark, self.reversi.light)

    def winner_name(self) -> str:
        _, _, result = self.result()
        if result == DARK:
            return "dark (X)"
        if result == LIGHT:
            return "light (O)"
        return "a draw"
