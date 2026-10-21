# TinyTactician overview

TinyTactician is an on-device board-game agent: a **360M-parameter language
model** provides the strategy, **Monte-Carlo Tree Search** provides the play
strength, and a **terminal UI** puts a human on the other side of the board.

The project is structured as

```bash
tiny_tactician/games        # the game itself: board representation and search
tiny_tactician/models        # the on-device language model (llama.cpp) and prompting
tiny_tactician/agents        # the agent: ties the model, the search and the commentary
tiny_tactician/environments  # game state and the move loop
tiny_tactician/run           # run scripts and the command line interface
```

The idea of this project is to write the simplest, smallest, most readable
agent.

- The project embraces polymorphism: Every individual class should be simple, but we offer alternatives
- Every use case should start with a run script, that picks one agent, environment, and model class to run

# Style guide

1. Target python 3.10 or higher
2. Use python with type annotations. Use `list` instead of `List`.
3. Use `pathlib` instead of `os.path`. Use `Path.read_text()` over `with ...open()` constructs.
4. Use `typer` to add interfaces
5. Keep code comments to a minimum and only highlight particularly logically challenging things
6. Do not append to the README unless specifically requested
7. Use `dataclass` for keeping track config
8. Do not catch exceptions unless explicitly told to.
9. Write concise, short, minimal code.
10. In most cases, avoid initializing variables just to pass them to a function. Instead just pass the expression to the function directly.
11. Not every exception has to be caught. Exceptions are a good way to show problems to a user.
12. This repository rewards minimal code. Try to be as concise as possible.

Here's an example for rule 10:

```python
# bad
a = func()
Class(a)

# good
Class(func())
```

# Test style

1. Use `pytest`, not `unittest`.
2. <IMPORTANT>Do not mock/patch anything that you're not explicitly asked to do</IMPORTANT>
3. Avoid writing trivial tests. Every test should test for at least one, preferably multiple points of failure
4. Avoid splitting up code in multiple lines like this: `a=func()\n assert a=b`. Instead, just do `assert func() == b`
5. The first argument to `pytest.mark.parametrize` should be a tuple (not a string! not a list!), the second argument must be a list (not a tuple!).

Here's an example for rule 4:

```python
# bad
result = func()
assert result == b

# good
assert func() == b
```

# Commit messages

Use the following format for commit messages:

- `ci: description` for all testing related changes, and changes to github workflows etc.
- `dev: description` for development related changes, including updates to the cursor or claude rules
- `fix(component): description` for bug fixes
- `feat(component): description` for new features
- `enh(component): description` for enhancements
- `docs: description` for documentation
- `ref(component): description` for refactoring
- `chore(component): description` for maintenance tasks (pre-commit hooks, imports, etc.)

Generally, the description should focus on the intent of the change, not the implementation details.

## Style notes

<IMPORTANT>Do **NOT** add "Co-authored-by: Cursor" lines to the commit message or to the trailer.</IMPORTANT>

# Reviewing

While preparing the commit message, flag critical issues that should be addressed before committing. Do not flag style issues or minor changes.

Flag the following:

- Anything that might raise an unhandled exception in an unintentional manner
- Anything that looks logically wrong or inconsistent
- Breaking changes to protocols/interfaces without corresponding updates to implementations

Flag the following style issue as minor:

- Imports not at top of file

# Components

Use these component names in parentheses for `fix`, `feat`, `enh`, and `ref` commits:

- `games` - Changes to the board game engine and the search (reversi, mcts)
- `models` - Changes to the on-device model layer (llamacpp, strategy, gguf)
- `agents` - Changes to the agent classes (board_agent)
- `env` - Changes to the game environment (reversi_env)
- `run` - Changes to run scripts and the command line interface
- `deps` - Dependency updates
