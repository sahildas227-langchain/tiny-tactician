"""Sweep the model-prior blend weight and report the win rate for each.

This is the experiment behind the `model_weight` default: the 360M model is a
weak Othello player on its own, so the question is how much of the search prior
it should be allowed to influence.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_tactician.agents.board_agent import BoardAgent
from tiny_tactician.environments.reversi_env import GameSession
from tiny_tactician.games.reversi import DARK, PASS
from tiny_tactician.models.llamacpp import LlamaCppConfig, LlamaCppServer
from tiny_tactician.models.strategy import StrategyModel

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "smollm2-360m-instruct-q4_k_m.gguf"
WEIGHTS = [0.0, 0.1, 0.2, 0.35, 0.5]
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SIMS = int(sys.argv[2]) if len(sys.argv) > 2 else 250


def play_game(model_agent: BoardAgent, plain_agent: BoardAgent, seed: int) -> int:
    session = GameSession(human_color=DARK, agent_color=-DARK)
    while not session.is_over():
        if session.legal() == 0:
            session.reversi.play(PASS)
            continue
        agent = model_agent if session.color == DARK else plain_agent
        choice = agent.choose_move(session.reversi.dark, session.reversi.light, session.color)
        session.play(choice.move)
    return session.result()[2]


def main() -> int:
    server = LlamaCppServer(LlamaCppConfig(model_path=MODEL, n_threads=12))
    server.start()
    strategy = StrategyModel(server)
    plain = BoardAgent(None, n_sims=SIMS, seed=0, use_model_prior=False, explain=False)
    print(f"games={GAMES} sims={SIMS}")
    for weight in WEIGHTS:
        agent = BoardAgent(strategy, n_sims=SIMS, seed=0, model_weight=weight, explain=False)
        wins = draws = 0
        started = time.time()
        for game in range(GAMES):
            result = play_game(agent, plain, seed=game)
            if result == DARK:
                wins += 1
            elif result == 0:
                draws += 1
        print(
            f"weight={weight:<5} model_wins={wins}/{GAMES} draws={draws} "
            f"win_rate={wins / GAMES:.0%}  ({time.time() - started:.0f}s)",
            flush=True,
        )
    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
