This project was submitted to the ryze.ai hackathon by sail_pvp.

# TinyTactician

A **360-million-parameter Othello master that plays entirely on your own machine**.

No cloud, no API keys, no telemetry. The model is a local GGUF file, the engine
is pure Python, and the whole thing fits in a terminal.

```
   a b c d e f g h
8  . . . . . . . .
7  . . . . . . . .
6  . . . . . . . .
5  . . . O X . . .
4  . . . X O . . .
3  . . . . . . . .
2  . . . . . . . .
1  . . . . . . . .
   a b c d e f g h
```

## The short version

A 360M model cannot play Othello. It cannot read an 8x8 board - we measured
that, and this project is partly about the measurement.

What a 360M model *is* good at is short, concrete explanations grounded in
domain knowledge. So TinyTactician splits the job:

| | Job | Size used |
|---|---|---|
| **MCTS / PUCT** | Play strength | search does the work |
| **SmolLM2-360M** | Explains every move in the language of Othello strategy | 361M is plenty |

The result is an opponent that plays a strong game **and** tells you *why*,
in concepts you can learn from - a tutor as much as an opponent. And because
the teacher only needs to write two sentences, a 360M model is exactly the
right size: a 7B model would cost twenty times the compute to say the same
thing, more slowly.

## What we measured

The honest part first. The original design had the model supply the search
prior directly. We built an ablation to find out whether that helped:

```bash
tiny-tactician ablate --games 12 --sims 250
```

```
      ablation result
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ outcome          ┃ count ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ model wins       │ 6     │
│ draws            │ 1     │
│ search-only wins │ 5     │
│ model win rate   │ 50%   │
└──────────────────┴───────┘
```

**Exactly neutral.** We dug into why, with three experiments:

1. **Does the model read the board?** We presented the legal moves as a
   numbered list and read the model's token log-probabilities over the option
   numbers. It answered "1" for *every* position, in *every* ordering - the
   choice tracked the list position, never the board. The numbers were a
   beautiful, sharply-peaked distribution over the wrong thing.
2. **Does it know the value of a square?** Asked to rate squares from 1 to 10,
   it answered "1" for all sixty-four. Asked "is this a corner? / an edge? /
   next to a corner?", it answered "yes" to all three, for every square.
3. **Can it judge two concrete moves?** Given two moves described by their
   consequences ("a1, a corner square, flips 3 discs" vs "e4, an inner
   square, flips 1 disc"), its preference, after averaging out its
   position bias by swapping the two options, was 0.50 every time.

A 360M model has no board perception to speak of. So we stopped asking it to
play, and asked it to *explain* - which is what small models are actually good
at. The ablation command is kept in the repo, because the negative result is
the most useful thing we learned.

The one place its knowledge does hold up is the opening: asked to rank the
standard openings it reliably answers c4, d3, e6. The agent therefore uses the
model's prior for the first few plies only, and hands over to the search
afterwards. That is a measured decision, not a guess.

## Benchmarks

### Method

Fairness is the whole problem in engine benchmarks, so the rules are strict:

- **Equal wall-clock, not equal simulations.** Every engine gets the same
  **0.30 s per move**. Comparing "1000 simulations" against "1000
  simulations" would only flatter whichever engine has the cheaper iteration.
- **The sides alternate.** Each pairing is played from both sides, so the
  first-move advantage cannot decide the result.
- **The model spends the same budget.** When the model is consulted, its
  latency comes out of the same 0.30 s allowance as the search.
- **The play-out policy is shared.** MUCT and PUCT use the same rollouts, so
  the comparison is between search algorithms rather than between play-out
  policies.

Reproduce it with one command:

```bash
tiny-tactician bench --games 6 --time-budget 0.3 --seed 1
```

### Reference engines

| engine | what it is |
|---|---|
| `random` | uniform random legal move |
| `greedy` | one ply of the static evaluation |
| `alpha-beta` | iterative-deepening negamax - the classic way to write a strong Othello engine |
| `muct` | Monte Carlo UCT with progressive bias (Ito, 2010) - the canonical Monte-Carlo method for Othello |

The baselines are covered by tests that check they return legal moves, that
alpha-beta takes a corner whenever one is available, and that the strong
engines beat random decisively. A benchmark only means something if the
reference engines are sound.

### Results

Round robin, 6 games per pairing, 0.30 s per move, seed 1:

| engine | W | L | D | score | Elo | vs random |
|---|---|---|---|---|---|---|
| `alpha-beta` | 28 | 1 | 1 | **95.0%** | +512 | 100% |
| `tiny-tactician` (search only) | 21 | 7 | 2 | **73.3%** | +176 | 91.7% |
| `tiny-tactician` (with model) | 21 | 9 | 0 | **70.0%** | +147 | 100% |
| `muct` | 12 | 18 | 0 | 40.0% | -70 | 100% |
| `random` | 4 | 25 | 1 | 15.0% | -301 | - |
| `greedy` | 2 | 28 | 0 | 6.7% | -458 | 33.3% |

Head to head (win rate for the row engine):

| engine | vs `alpha-beta` | vs `muct` | vs `tiny-tactician` |
|---|---|---|---|
| `alpha-beta` | - | 100% | 100% / 75% |
| `tiny-tactician` (search only) | 25% | 100% | - |
| `tiny-tactician` (with model) | 0% | 100% | 50% |
| `muct` | 0% | - | 0% |

### Reading the results honestly

- **Alpha-beta beats our Monte-Carlo search** (100% and 75% of the games).
  This is the expected result and it is the most useful thing in this
  section: Othello has a low branching factor and a genuinely informative
  evaluation, so alpha-beta with a good evaluation is very hard to beat. Our
  search is a competent player, not a state-of-the-art one, and we would
  rather say so than hide it.
- **Our PUCT search beats MUCT 100%** of the time at equal wall-clock with
  the same play-out policy. Prior-driven expansion is doing real work here.
- **The model changes nothing** (50% head to head). This matches the
  ablation above: a 360M model cannot read the board, so it has nothing to
  contribute to move choice. We keep the model because it *explains* the
  moves, not because it picks them.
- **Greedy is worse than random.** One ply of evaluation is a bad player in
  Othello: it grabs whatever the evaluation likes at that instant and walks
  straight into a corner trap.
- The sample is small (6 games per pairing, 30 games per engine). The
  ordering is stable and matches the theory, but individual percentages
  carry wide error bars.

### How this compares to published engines

The strongest Othello programs - Ed Gilbert's engine, Hector, the computer
Othello tournament winners - are far stronger than anything in the table
above. We did not run them and make no claim about the gap. What we do claim
is that these numbers come from a from-scratch bitboard implementation in
pure Python with no native extensions, and that the whole measurement is
reproducible with one command.

### Speed

The engine is bitboard-based: a position is two 64-bit integers, so every
operation is an integer bit-op. Measured on this machine (12 CPU threads,
Python 3.14):

| | |
|---|---|
| legal-move queries | **545,586 /s** |
| move application | **819,089 /s** |
| PUCT simulations | **~850 /s** |

### The model on CPU vs GPU

The CPU build of llama.cpp and the CUDA build (RTX 3050, all 99 layers
offloaded) were measured on the same prompt:

| | CPU, 12 threads | RTX 3050 | speed-up |
|---|---|---|---|
| seconds per call | 0.183 | **0.043** | **4.3x** |
| prompt processing | 1,370 tok/s | **7,191 tok/s** | **5.2x** |
| generation | 33.4 tok/s | **158.3 tok/s** | **4.7x** |

The GPU build is one environment variable away:

```bash
# a CUDA build of llama.cpp
LLAMA_BIN=~/.local/llama-cuda/llama-b11799/llama \
  tiny-tactician play --gpu-layers 99
```

## How it works

```
   ┌──────────────────────┐
   │  SmolLM2-360M-Instruct│
   │  (local GGUF, 4-bit)  │  → explains the move in Othello terms
   └──────────┬───────────┘
              │ commentary
              ▼
   ┌──────────────────────┐
   │   MCTS (PUCT)         │  → 600 simulations, bitboard engine
   │   bitboard Othello    │  → picks the move
   └──────────────────────┘
```

The search is a plain PUCT MCTS over a bitboard engine: a position is two
64-bit integers, so legal-move generation, flips and rollouts run thousands of
times per second in pure Python, with no numpy and no compiled extensions.

## The model

| | |
|---|---|
| **Model** | [`SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) |
| **Parameters** | **361,821,120** (budget: 500,000,000) ✓ |
| **Quantisation** | Q4_K_M GGUF (~270 MB on disk) |
| **Runtime** | [`llama.cpp`](https://github.com/ggml-org/llama.cpp), local process on `127.0.0.1` |
| **License** | Apache-2.0 |

`tiny-tactician doctor` prints the parameter count read directly out of the
GGUF container and checks it against the 500M budget. Nothing is hard-coded.

### Talking to the model

The model is reached through llama.cpp's native `/completion` endpoint, with
the prompt templated in Python. Two techniques do the heavy lifting:

- **GBNF grammar.** When we need the model to *name* a move, we constrain it
  with a grammar that only allows legal squares. A small model then cannot
  hallucinate an illegal move at all.
- **Token log-probabilities.** When we need the model's *opinion* rather than
  its text, we read the log-probabilities over the next token. That is how the
  numbered-option prior in the ablation was produced.

## Install

Requires Python 3.10+ and the `llama` binary from llama.cpp.

```bash
# 1. the local inference runtime
curl -LsSf https://llama.app/install.sh | sh

# 2. this project
git clone https://github.com/sunburnedasian/tiny-tactician
cd tiny-tactician
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. the model weights (~270 MB)
tiny-tactician download
```

## Use

```bash
# play the agent (you are light by default)
tiny-tactician play

# play as dark, with a deeper search
tiny-tactician play --side dark --sims 1200

# watch the agent play itself
tiny-tactician selfplay --games 10

# ask the model what it sees in a position
tiny-tactician explain --moves e3,f4

# measure what the model actually contributes
tiny-tactician ablate --games 12

# benchmark against reference engines (alpha-beta, MUCT, greedy, random)
tiny-tactician bench --games 6 --time-budget 0.3

# measure the model's throughput, on CPU or on the GPU
tiny-tactician bench-model
LLAMA_BIN=~/.local/llama-cuda/llama-b11799/llama tiny-tactician bench-model --gpu-layers 99

# verify the runtime and the parameter budget
tiny-tactician doctor
```

During a game the agent shows its reasoning:

```
   a b c d e f g h
8  . . . . . . . .
7  . . . . . . . .
6  . . . . . . . .
5  . . . O X . . .
4  . . . X O . . .
3  . . . . . . . .
2  . . . . . . . .
1  . . . . . . . .
   a b c d e f g h
  Current: X  (X=dark, O=light, *=legal, #=last)

Agent played e6 (2.9s)
╭───────────────────────────── model's reasoning ──────────────────────────────╮
│ This is a strong move because it allows you to control the center and create │
│ opportunities for other players on the board. You can also use your          │
│ opponent's weaknesses to your advantage by exploiting their weakness in the  │
│ corners of the game. Additionally, this move gives you an option to try your │
│ hand at the corner or edge, which is often considered a high-risk but        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Layout

```
src/tiny_tactician/
├── games/
│   ├── reversi.py      bitboard Othello engine (pure Python, no deps)
│   └── mcts.py         PUCT Monte-Carlo Tree Search
├── models/
│   ├── llamacpp.py     local llama.cpp server process
│   ├── strategy.py     prompting, grammar and log-probability handling
│   └── gguf.py         GGUF reader (parameter counting)
├── agents/
│   └── board_agent.py  model prior + search + commentary
├── environments/
│   └── reversi_env.py  game state and move loop
├── bench/
│   ├── evaluate.py     the standard Othello feature evaluation
│   ├── baselines.py    random, greedy, alpha-beta and MUCT engines
│   ├── tournament.py   round-robin harness with equal time budgets
│   └── speed.py        engine, search and model throughput
└── run/
    └── cli.py          the command line interface
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/            # everything
pytest -m "not slow"     # skip the long strength benchmark
```

The engine is verified against an independent, deliberately naive array-based
reference implementation over thousands of random positions, so the bitboard
trickery cannot silently drift. The GGUF reader is checked against a
hand-written GGUF file. The model's output handling is tested without loading
the model, so the test suite runs in seconds and needs no weights.

## License

Apache License 2.0 for the code (see `LICENSE.md`, and `NOTICE` for
attribution).

The model weights are **not** vendored into this repository; they are
downloaded separately and are licensed by their authors under the Apache
License 2.0:

| Model | License | Link |
|---|---|---|
| `SmolLM2-360M-Instruct` | Apache-2.0 | [Hugging Face](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) |

Portions of this project derive from
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (MIT License);
see `NOTICE` for the full attribution.
