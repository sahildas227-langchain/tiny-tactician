"""End-to-end smoke test: start the local llama.cpp server and query it."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_tactician.agents.board_agent import BoardAgent
from tiny_tactician.games.reversi import DARK, format_move, legal_moves, start
from tiny_tactician.models.llamacpp import LlamaCppConfig, LlamaCppServer
from tiny_tactician.models.strategy import StrategyModel

MODEL = Path(__file__).resolve().parents[1] / "models" / "smollm2-360m-instruct-q4_k_m.gguf"


def main() -> int:
    started = time.time()
    server = LlamaCppServer(LlamaCppConfig(model_path=MODEL, n_threads=12))
    server.start()
    print(f"server ready in {time.time() - started:.1f}s at {server.endpoint}")
    strategy = StrategyModel(server)
    dark, light, color = start()
    legal = legal_moves(dark, light, color)
    print("legal opening moves:", [format_move(1 << i) for i in range(64) if legal >> i & 1])

    t0 = time.time()
    prior = strategy.propose_moves(dark, light, color)
    print(f"model shortlist ({time.time() - t0:.1f}s):", [format_move(m) for m in sorted(prior, key=lambda k: -prior[k])])

    t0 = time.time()
    agent = BoardAgent(strategy, n_sims=200, seed=1)
    choice = agent.choose_move(dark, light, DARK)
    print(f"agent chose {choice.square} in {time.time() - t0:.1f}s")
    print("policy top:", choice.top_moves(3))
    print("commentary:", choice.comment)

    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
