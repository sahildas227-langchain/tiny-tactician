"""The small model itself: SmolLM2-360M-Instruct as a Reversi strategist.

The 360M-parameter model is far too small to search a game tree, and - as this
project's ablation shows - too small to read an 8x8 board reliably either.  It
is, however, good at the one thing this project needs from a language model:
short, concrete explanations grounded in Othello knowledge.

Two techniques keep it honest:

* **GBNF grammar.** When we need it to *name* a square, the output is
  constrained to the legal squares, so it cannot hallucinate an illegal move.
* **Token log-probabilities.** When we need its *opinion* rather than its
  text, we read the log-probabilities over the next token.

Everything runs locally; the model file is a plain GGUF on disk and the
``llama`` binary does the maths on this machine.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from tiny_tactician.games.reversi import (
    DARK,
    format_move,
    legal_moves,
    render,
)

__all__ = ["StrategyModel", "parse_ranked_moves", "blend_priors", "logprob_prior"]

_MOVE_RE = re.compile(r"\b([a-h][1-8])\b", re.IGNORECASE)

_SYSTEM = (
    "You are a world-class Othello (Reversi) player. "
    "Othello is played with black and white discs on an 8x8 board. "
    "You are precise, tactical and concise, and you never mention chess pieces."
)


def _chat_prompt(user: str) -> str:
    """Render a single-turn chat prompt in SmolLM2's chat format."""
    return f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def parse_ranked_moves(text: str, legal: int) -> dict[int, float]:
    """Extract a ranked shortlist of moves from free-form model output.

    Only legal moves are accepted, so a hallucinated square can never enter
    the search.
    """
    if not legal:
        return {}
    legal_by_name = {format_move(1 << i): 1 << i for i in range(64) if legal >> i & 1}
    seen: list[int] = []
    for token in _MOVE_RE.findall(text):
        move = legal_by_name.get(token.lower())
        if move is not None and move not in seen:
            seen.append(move)
    if not seen:
        return {}
    n = len(seen)
    raw = {move: float(n - i) for i, move in enumerate(seen)}
    total = sum(raw.values())
    return {move: value / total for move, value in raw.items()}


def logprob_prior(logprobs: list[dict], option_count: int) -> dict[int, float]:
    """Turn option log-probabilities into a normalized distribution over options.

    ``logprobs`` is a list of ``{"token", "logprob"}`` dictionaries; only
    tokens that are the index of an available option are considered.
    """
    weights: dict[int, float] = {}
    for entry in logprobs:
        token = entry.get("token", "").strip()
        if not token.isdigit():
            continue
        index = int(token) - 1
        if 0 <= index < option_count:
            weights[index] = max(weights.get(index, -math.inf), float(entry.get("logprob", -math.inf)))
    if not weights:
        return {}
    top = max(weights.values())
    exps = {index: math.exp(value - top) for index, value in weights.items()}
    total = sum(exps.values())
    return {index: value / total for index, value in exps.items()}


def blend_priors(model_prior: dict[int, float], heuristic_prior: dict[int, float], weight: float = 0.5) -> dict[int, float]:
    """Mix the model's distribution with the positional heuristic.

    The model contributes its ranking; the heuristic guarantees that a legal,
    sensible prior always exists even when the model output cannot be parsed.
    """
    if not heuristic_prior:
        return dict(model_prior)
    if not model_prior:
        return dict(heuristic_prior)
    blended = {
        move: (1 - weight) * heuristic_prior.get(move, 0.0) + weight * model_prior.get(move, 0.0)
        for move in heuristic_prior
    }
    total = sum(blended.values())
    if total <= 0:
        return dict(heuristic_prior)
    return {move: value / total for move, value in blended.items()}


@dataclass
class StrategyModel:
    """Wraps a local llama.cpp server with Reversi-specific prompting."""

    server: object
    model_name: str = "SmolLM2-360M-Instruct"
    n_predict_text: int = 110
    model_weight: float = 0.5
    n_probs: int = 20

    def _complete(self, user: str, n_predict: int) -> str:
        return self.server.complete(_chat_prompt(user), n_predict=n_predict)

    def board_text(self, dark: int, light: int, color: int, legal_mask: int = 0) -> str:
        return render(dark, light, color, legal_mask)

    def legal_squares(self, dark: int, light: int, color: int) -> list[str]:
        legal = legal_moves(dark, light, color)
        return [format_move(1 << i) for i in range(64) if legal >> i & 1]

    def propose_moves(self, dark: int, light: int, color: int) -> dict[int, float]:
        """Ask the model which legal move it prefers, and how strongly.

        The legal moves are numbered so that each option is a single token;
        the model's log-probabilities over those tokens are the prior.
        """
        legal = legal_moves(dark, light, color)
        squares = [format_move(1 << i) for i in range(64) if legal >> i & 1]
        moves = [1 << i for i in range(64) if legal >> i & 1]
        if not squares:
            return {}
        options = "\n".join(f"{index + 1}. {square}" for index, square in enumerate(squares))
        user = (
            f"Othello position (X=dark, O=light, '.',=empty, '*'=a legal square):\n\n"
            f"{self.board_text(dark, light, color, legal)}\n\n"
            f"Which move is best?\n{options}\n\n"
            "Reply with the number of the best move only."
        )
        logprobs = self.server.complete_logprobs(_chat_prompt(user), n_predict=1, n_probs=self.n_probs)
        distribution = logprob_prior(logprobs[0] if logprobs else [], len(squares))
        if distribution:
            return {moves[index]: probability for index, probability in distribution.items()}
        # Fallback: sample a move under a grammar that only allows legal squares.
        grammar = 'root ::= "' + '" | "'.join(squares) + '"\n'
        text = self.server.complete(_chat_prompt(user), n_predict=8, grammar=grammar)
        return parse_ranked_moves(text, legal)

    def explain(self, dark: int, light: int, color: int, move: int) -> str:
        """Ask the model to justify a move in one short paragraph."""
        side = "X (dark)" if color == DARK else "O (light)"
        square = format_move(move)
        user = (
            f"You are playing {side} in Othello, a board game with black and white discs on an 8x8 board.\n\n"
            f"{self.board_text(dark, light, color, legal_moves(dark, light, color))}\n\n"
            f"You are about to play {square}.\n"
            "In two sentences, explain why this is a strong move. Name the concrete reason, "
            "for example a corner, an edge, mobility, or a tactic. Do not mention chess pieces."
        )
        return _clean(self._complete(user, self.n_predict_text))

    def describe_position(self, dark: int, light: int, color: int) -> str:
        """Ask the model to read the whole position (used by the `explain` command)."""
        side = "X (dark)" if color == DARK else "O (light)"
        user = (
            f"You are playing {side} in Othello, a board game with black and white discs on an 8x8 board.\n\n"
            f"{self.board_text(dark, light, color, legal_moves(dark, light, color))}\n\n"
            f"Legal moves: {', '.join(self.legal_squares(dark, light, color))}\n\n"
            "Describe the position in three sentences: who has the advantage, which squares matter, "
            "and what you want to do next. Do not mention chess pieces."
        )
        return _clean(self._complete(user, self.n_predict_text + 24))


def _clean(text: str) -> str:
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    text = re.sub(r"^```[a-z]*\s*", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # A truncated generation ends mid-sentence; drop the dangling fragment.
    if text and text[-1] not in ".!?":
        head, _, tail = text.rpartition(" ")
        if len(tail) > 0 and head:
            text = head.rstrip(",;:") + "."
    return text


def default_model_path() -> Path:
    """Locate the bundled GGUF weights."""
    env = os.environ.get("TINYTACTICIAN_MODEL")
    candidates = [Path(env)] if env else []
    candidates += [
        Path(__file__).resolve().parents[3] / "models" / "smollm2-360m-instruct-q4_k_m.gguf",
        Path.cwd() / "models" / "smollm2-360m-instruct-q4_k_m.gguf",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "model weights not found. Run 'tiny-tactician download' or place "
        "smollm2-360m-instruct-q4_k_m.gguf in ./models/"
    )
