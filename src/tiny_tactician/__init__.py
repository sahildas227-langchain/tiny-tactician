"""TinyTactician: a sub-500M-parameter, on-device Reversi (Othello) master.

This package retargets the minimal, polymorphic
agent / environment / model architecture popularized by mini-swe-agent to a
single, narrow task: play Othello in the terminal using a 360M-parameter
SmolLM2 model for strategic priors and commentary, accelerated by Monte-Carlo
Tree Search (PUCT). All inference runs locally via llama.cpp (no cloud APIs).
"""

__version__ = "0.1.0"
