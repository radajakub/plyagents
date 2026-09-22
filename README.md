# game_agents

Agentic and LLM approaches to solving simple games, trained and compared on
[PlyBench](https://plybench.jakubrada.com)'s training harness.

## `reasoning_bank`

A port of [ReasoningBank](https://arxiv.org/abs/2509.25140) to turn-based games, kept faithful to the
paper's own implementation: every move retrieves the memory items whose state most resembles the one
on the board and appends them to the system message; every finished training game is judged,
distilled into at most three items, and filed once per state the agent acted in. The judge prompt and
its two-line contract, the success/failure induction prompts, the item cap and split, the
instruct-wrapped symmetric embedding retrieval and the induction temperature are upstream's — only
the domain wording changes (web navigation → turn-based games).

```
src/game_agents/reasoning_bank/
  params.py      the config string
  player.py      the LearnablePlayer: retrieval on every move, induction after every training game
  judge.py       LLM-as-a-judge over a finished game (WebArena/autoeval)
  trajectory.py  a plybench transcript as <observation>/<think>/<action>
  derivation.py  per-epoch cache that keeps a resumed epoch reproducible
  prompts.py     the retrieved-experience header
  memory/        item.py · store.py · retrieval.py · induction.py · prompts.py
scripts/         train.py, analyse.py
experiments/training/*.json
```

What is *not* ported is the browsergym/langchain scaffold the original wrapped its agent in: moves go
through plybench's own prompt adapter and output strategy, so an arm differs from the benchmark's
plain `llm` player by exactly the retrieved memory block and nothing else. Two smaller consequences of
the domain change: a record's retrieval key is the game's objective plus the current observation (the
closest analogue of the paper's intent-plus-page key), and records keep the induced items and their
provenance but not a copy of the trajectory, which retrieval never read.

### Config string

```
reasoning_bank:<observation>:<strategy>:<provider>:<model>:<options>
```

| option | default | meaning |
| --- | --- | --- |
| `memory` | `on` | `off` freezes the bank empty — the baseline arm, identical prompt otherwise |
| `top_n` | `1` | memory records retrieved per move (the paper's implementation uses 1) |
| `criteria` | `autoeval` | success label: the judge's own call, or the engine's outcome (`gt`) |
| `draw_is_success` | `False` | count a draw as success under `gt` |
| `asymmetric` | `False` | embed stored entries as documents and the live query as a query |
| `embed` | `gemini/gemini-embedding-2` | retrieval embedding model |
| `judge` / `induce` | the agent's model | override the judging / induction model |

`checkpoint=<path>` is appended by the training harness; a trained arm is therefore an ordinary player
config that the benchmark can play afterwards.

## Running

Provider keys come from `.env` (see plybench's configuration section).

```bash
uv run python scripts/train.py --experiment smoke          # tic-tac-toe, 2 epochs, tiny
uv run python scripts/train.py --experiment connect_four --concurrency 4
uv run python scripts/analyse.py --experiment connect_four
```

`--notify` pushes a [clankers](https://pypi.org/project/clankers/) notification when the run starts,
as each epoch finishes (with an ETA derived from round throughput) and on success or crash. It needs
`NTFY_URL` and `NTFY_TOPIC` in the environment; an unconfigured or unreachable server is warned about
and never interrupts the run.

Runs are resumable: completed epochs and rounds are skipped, and an interrupted epoch replays its
recorded games through the learner, serving the judge and induction results from that epoch's cache so
it lands on exactly the checkpoint an uninterrupted run would.

```bash
uv run pytest        # offline: drives the whole harness with a stub LLM
```
