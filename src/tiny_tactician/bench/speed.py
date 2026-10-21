"""Throughput benchmarks for the engine, the search and the model."""

from __future__ import annotations

import time

from tiny_tactician.games.mcts import search
from tiny_tactician.games.reversi import legal_moves, play, start

__all__ = ["benchmark_engine", "benchmark_search", "benchmark_model"]


def benchmark_engine(iterations: int = 20000) -> dict:
    """Measure the raw bitboard engine throughput."""
    dark, light, color = start()
    begin = time.perf_counter()
    for _ in range(iterations):
        legal_moves(dark, light, color)
    legal_per_sec = iterations / (time.perf_counter() - begin)

    begin = time.perf_counter()
    for _ in range(iterations):
        play(dark, light, color, 1 << 19)
    plays_per_sec = iterations / (time.perf_counter() - begin)
    return {
        "legal_moves_per_sec": legal_per_sec,
        "play_moves_per_sec": plays_per_sec,
        "iterations": iterations,
    }


def benchmark_search(sim_counts=(100, 1000, 5000), seed: int = 0) -> list[dict]:
    """Measure Monte-Carlo simulations per second at a few search depths."""
    dark, light, color = start()
    rows = []
    for n_sims in sim_counts:
        begin = time.perf_counter()
        search(dark, light, color, n_sims=n_sims, seed=seed)
        elapsed = time.perf_counter() - begin
        rows.append({"n_sims": n_sims, "seconds": elapsed, "sims_per_sec": n_sims / elapsed})
    return rows


def benchmark_model(server, prompt: str, n_predict: int = 48, runs: int = 5) -> dict:
    """Measure the model's prompt-processing and generation throughput."""
    begin = time.perf_counter()
    timings = {}
    for _ in range(runs):
        result = server.complete_timed(prompt, n_predict=n_predict, cache_prompt=False)
        timings = result.get("timings") or timings
    elapsed = (time.perf_counter() - begin) / runs
    return {
        "seconds_per_call": elapsed,
        "runs": runs,
        "n_predict": n_predict,
        "prompt_tokens": timings.get("prompt_n"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_tokens": timings.get("predicted_n"),
        "predicted_per_second": timings.get("predicted_per_second"),
    }
