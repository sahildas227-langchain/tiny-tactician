"""TinyTactician command line interface.

Subcommands
-----------
``play``      play against the agent in your terminal
``selfplay``  let the agent play itself (used for benchmarking)
``explain``   ask the model to read a position
``doctor``    check that the local model and runtime are in place
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tiny_tactician.agents.board_agent import BoardAgent
from tiny_tactician.environments.reversi_env import GameSession
from tiny_tactician.games.reversi import (
    DARK,
    LIGHT,
    PASS,
    Reversi,
    format_move,
    parse_move,
)
from tiny_tactician.models.gguf import parameter_count
from tiny_tactician.models.llamacpp import LlamaCppConfig, LlamaCppServer, find_llama_binary
from tiny_tactician.models.strategy import StrategyModel, default_model_path

app = typer.Typer(add_completion=False, help="TinyTactician - a 360M-parameter Othello master.")
console = Console(highlight=False)

MODEL_LABEL = "SmolLM2-360M-Instruct (Q4_K_M, 361.8M params)"
REPO_ID = "QuantFactory/SmolLM2-360M-Instruct-GGUF"
REPO_FILE = "SmolLM2-360M-Instruct.Q4_K_M.gguf"


def _build_agent(
    model_path: Path | None,
    sims: int,
    seed: int,
    use_model: bool,
    explain: bool,
    threads: int,
    gpu_layers: int,
    model_weight: float = 0.5,
) -> tuple[BoardAgent, LlamaCppServer | None]:
    """Create the agent, starting the local model server when it is needed."""
    if not use_model:
        return BoardAgent(None, n_sims=sims, seed=seed, use_model_prior=False, explain=False), None
    path = model_path or default_model_path()
    server = LlamaCppServer(
        LlamaCppConfig(
            model_path=path,
            n_threads=threads,
            n_gpu_layers=gpu_layers,
        )
    )
    server.start()
    strategy = StrategyModel(server)
    return BoardAgent(strategy, n_sims=sims, seed=seed, explain=explain, model_weight=model_weight), server


def _print_board(session: GameSession) -> None:
    console.print(session.reversi.render())


@app.command()
def play(
    model_path: Path = typer.Option(None, "--model", help="Path to the GGUF weights."),
    sims: int = typer.Option(600, "--sims", help="MCTS simulations per move."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    side: str = typer.Option("dark", "--side", help="Play as 'dark' or 'light'."),
    no_model: bool = typer.Option(False, "--no-model", help="Disable the language model (search only)."),
    no_explain: bool = typer.Option(False, "--no-explain", help="Skip the model's commentary."),
    threads: int = typer.Option(0, "--threads", help="CPU threads for inference (0 = all)."),
    gpu_layers: int = typer.Option(0, "--gpu-layers", help="Layers to offload to the GPU."),
) -> None:
    """Play Othello against the agent."""
    human = DARK if side.lower().startswith("d") else LIGHT
    agent = BoardAgent(None)
    server: LlamaCppServer | None = None
    if not no_model:
        console.print(f"[bold green]Loading {MODEL_LABEL}...[/bold green]")
        agent, server = _build_agent(model_path, sims, seed, True, not no_explain, threads, gpu_layers)
    else:
        agent = BoardAgent(None, n_sims=sims, seed=seed, use_model_prior=False, explain=False)
    console.print(f"[bold]You are {'dark (X)' if human == DARK else 'light (O)'}.[/bold] You move first.")
    console.print("[dim]Type a square like 'e3', or 'pass'. Type 'quit' to stop.[/dim]\n")

    session = GameSession(human_color=human, agent_color=-human)
    try:
        while not session.is_over():
            _print_board(session)
            legal = session.legal()
            if legal == 0:
                console.print("[yellow]No legal move - passing.[/yellow]")
                session.reversi.play(PASS)
                continue
            if session.color == human:
                console.print(f"[bold cyan]Your turn.[/bold cyan] Legal: {', '.join(session.legal_squares())}")
                while True:
                    text = typer.prompt("move", default="", show_default=False)
                    if text.strip().lower() in ("q", "quit", "exit"):
                        console.print("[yellow]Game aborted.[/yellow]")
                        return
                    try:
                        move = session.parse_human_move(text)
                        break
                    except ValueError as exc:
                        console.print(f"[red]{exc}[/red]")
                session.play(move)
                console.print(f"[green]You played {format_move(move)}.[/green]\n")
            else:
                console.print("[bold magenta]Agent thinking...[/bold magenta]")
                started = time.time()
                choice = agent.choose_move(session.reversi.dark, session.reversi.light, session.color, plies=session.plies)
                elapsed = time.time() - started
                session.play(choice.move)
                console.print(f"[bold magenta]Agent played {choice.square}[/bold magenta] [dim]({elapsed:.1f}s)[/dim]")
                if choice.comment:
                    console.print(Panel(choice.comment, title="model's reasoning", border_style="magenta"))
                if choice.model_prior:
                    listed = ", ".join(f"{format_move(m)}" for m in sorted(choice.model_prior, key=lambda k: -choice.model_prior[k]))
                    console.print(f"[dim]model shortlist: {listed}[/dim]")
                console.print()
        dark, light, _ = session.result()
        console.print(Panel(f"[bold]Final score: dark {dark} - light {light}[/bold]\n{session.winner_name()} wins!",
                            title="game over", border_style="green"))
    finally:
        if server is not None:
            server.stop()


@app.command()
def selfplay(
    model_path: Path = typer.Option(None, "--model", help="Path to the GGUF weights."),
    games: int = typer.Option(5, "--games", help="Number of self-play games."),
    sims: int = typer.Option(600, "--sims", help="MCTS simulations per move."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    no_model: bool = typer.Option(False, "--no-model", help="Disable the language model (search only)."),
    threads: int = typer.Option(0, "--threads", help="CPU threads for inference (0 = all)."),
    gpu_layers: int = typer.Option(0, "--gpu-layers", help="Layers to offload to the GPU."),
) -> None:
    """Let the agent play itself; reports the score over N games."""
    agent, server = _build_agent(model_path, sims, seed, not no_model, False, threads, gpu_layers)
    wins = draws = 0
    try:
        for game in range(games):
            session = GameSession(human_color=DARK, agent_color=LIGHT)
            while not session.is_over():
                legal = session.legal()
                if legal == 0:
                    session.reversi.play(PASS)
                    continue
                choice = agent.choose_move(session.reversi.dark, session.reversi.light, session.color, plies=session.plies)
                session.play(choice.move)
            dark, light, result = session.result()
            if result == DARK:
                wins += 1
            elif result == 0:
                draws += 1
            console.print(f"game {game + 1}/{games}: dark {dark} - light {light}  ({session.winner_name()})")
        table = Table(title="self-play summary")
        table.add_column("metric")
        table.add_column("value")
        table.add_row("games", str(games))
        table.add_row("dark wins", str(wins))
        table.add_row("draws", str(draws))
        table.add_row("light wins", str(games - wins - draws))
        table.add_row("model", "off (search only)" if no_model else MODEL_LABEL)
        console.print(table)
    finally:
        if server is not None:
            server.stop()


@app.command()
def explain(
    model_path: Path = typer.Option(None, "--model", help="Path to the GGUF weights."),
    moves: str = typer.Option("", "--moves", help="Comma separated squares to play first, e.g. 'e3,f4'."),
    threads: int = typer.Option(0, "--threads", help="CPU threads for inference (0 = all)."),
    gpu_layers: int = typer.Option(0, "--gpu-layers", help="Layers to offload to the GPU."),
) -> None:
    """Ask the small model to read and explain a position."""
    path = model_path or default_model_path()
    server = LlamaCppServer(LlamaCppConfig(model_path=path, n_threads=threads, n_gpu_layers=gpu_layers))
    server.start()
    try:
        strategy = StrategyModel(server)
        board = Reversi.start()
        for token in [t.strip() for t in moves.split(",") if t.strip()]:
            board.play(parse_move(token))
        console.print(board.render())
        console.print(Panel(strategy.describe_position(board.dark, board.light, board.color),
                            title="model's read of the position", border_style="cyan"))
    finally:
        server.stop()


@app.command()
def ablate(
    model_path: Path = typer.Option(None, "--model", help="Path to the GGUF weights."),
    games: int = typer.Option(10, "--games", help="Number of games to play."),
    sims: int = typer.Option(300, "--sims", help="MCTS simulations per move."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    threads: int = typer.Option(0, "--threads", help="CPU threads for inference (0 = all)."),
    gpu_layers: int = typer.Option(0, "--gpu-layers", help="Layers to offload to the GPU."),
    model_weight: float = typer.Option(0.5, "--model-weight", help="How much weight to give the model prior (0-1)."),
    prior_plies: int = typer.Option(8, "--prior-plies", help="Use the model prior only for the first N plies."),
) -> None:
    """Measure what the model contributes.

    Plays the agent *with* the model's strategy prior against the *same* agent
    with search only.  A win rate above 50% means the 360M model genuinely
    improves play, rather than merely decorating it.
    """
    console.print("[bold]Ablation: model prior vs. search only[/bold]")
    console.print(f"[dim]{MODEL_LABEL} vs. the identical agent without the model prior[/dim]\n")
    agent_with_model, server = _build_agent(model_path, sims, seed, True, False, threads, gpu_layers, model_weight)
    agent_with_model.model_prior_plies = prior_plies
    wins = losses = draws = 0
    try:
        for game in range(games):
            # Alternate the side the model plays on, and vary the search seed,
            # so that the first-move advantage does not decide the experiment.
            model_is_dark = game % 2 == 0
            model_agent = BoardAgent(
                agent_with_model.strategy,
                n_sims=sims,
                seed=seed + game,
                model_weight=model_weight,
                model_prior_plies=prior_plies,
                use_model_prior=True,
                explain=False,
            )
            plain_agent = BoardAgent(
                None,
                n_sims=sims,
                seed=seed + game,
                use_model_prior=False,
                explain=False,
            )
            session = GameSession(human_color=DARK, agent_color=LIGHT)
            while not session.is_over():
                if session.legal() == 0:
                    session.reversi.play(PASS)
                    continue
                agent = model_agent if session.color == DARK else plain_agent
                choice = agent.choose_move(session.reversi.dark, session.reversi.light, session.color, plies=session.plies)
                session.play(choice.move)
            dark, light, result = session.result()
            model_score, other_score = (dark, light) if model_is_dark else (light, dark)
            if model_score > other_score:
                wins += 1
            elif model_score == other_score:
                draws += 1
            else:
                losses += 1
            console.print(f"game {game + 1}/{games} (model as {'dark' if model_is_dark else 'light'}): "
                          f"model {model_score} - search-only {other_score}")
        table = Table(title="ablation result")
        table.add_column("outcome")
        table.add_column("count")
        table.add_row("model wins", str(wins))
        table.add_row("draws", str(draws))
        table.add_row("search-only wins", str(losses))
        table.add_row("model win rate", f"{wins / games:.0%}")
        console.print(table)
    finally:
        if server is not None:
            server.stop()


@app.command()
def download(
    model_path: Path = typer.Option(None, "--model", help="Where to write the GGUF file."),
) -> None:
    """Download the model weights (361M parameters, ~270 MB)."""
    destination = model_path or Path("models") / "smollm2-360m-instruct-q4_k_m.gguf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        console.print("[red]huggingface_hub is required: pip install huggingface_hub[/red]")
        raise typer.Exit(code=1)
    console.print(f"[bold green]Downloading {MODEL_LABEL}...[/bold green]")
    with console.status("[bold green]downloading...[/bold green]"):
        cached = hf_hub_download(REPO_ID, REPO_FILE)
        if Path(cached).resolve() != destination.resolve():
            destination.write_bytes(Path(cached).read_bytes())
    console.print(f"[green]Saved to {destination}[/green]")


@app.command()
def doctor(
    model_path: Path = typer.Option(None, "--model", help="Path to the GGUF weights."),
) -> None:
    """Verify the local runtime and report the model's parameter count."""
    table = Table(title="TinyTactician doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    try:
        binary = find_llama_binary()
        table.add_row("llama.cpp binary", "[green]ok[/green]", binary)
    except FileNotFoundError as exc:
        table.add_row("llama.cpp binary", "[red]missing[/red]", str(exc))
    try:
        path = model_path or default_model_path()
        count = parameter_count(path)
        status = "[green]ok[/green]" if count <= 500_000_000 else "[red]over budget[/red]"
        table.add_row("model file", "[green]ok[/green]", str(path))
        table.add_row("parameters", status, f"{count:,} / 500,000,000")
    except Exception as exc:  # noqa: BLE001 - a diagnostic command should report, not crash
        table.add_row("model file", "[red]missing[/red]", str(exc))
    table.add_row("inference", "[green]local[/green]", "llama.cpp on this machine, no external API")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
