# TextRL config strings

TextRL is reinforcement learning of a frozen LLM through textual memories: the memories are the parameters, the
losses' feedback is the gradient, and Apply is the optimizer. The text actor-critic (TAC) of the paper is one of its
instances ([§5.3](#53-policy-gradient-analogues)). This document is how to write a TextRL player spec: every component
with its parameters, how the components fit together, and how to build known algorithms from them. The code is the authority:
`src/plyagents/textrl/params.py` (the TextRL options), each component's `build.py` (its types) and its params class (its
parameters and defaults). Every recipe in this document parses and passes the build checks (checked 2026-11-13).

Contents:

1. [Syntax](#1-syntax)
2. [The TextRL options](#2-the-textrl-options)
3. [Components](#3-components): actor, critic, losses, backward, Apply, scheduler, sweep, selector, reward, target
4. [Composing a TextRL player](#4-composing-a-textrl-player): the data flow, the build rules, what is ignored
5. [Algorithms from the blocks](#5-algorithms-from-the-blocks)

## 1. Syntax

A TextRL player is written `textrl:<options>` in an experiment's `players` list, e.g.

```
textrl:model=metacentrum:gpt-oss-120b,actor=llm,critic=llm_q,actor_loss=llm,critic_loss=llm[target=critic],sweep=parallel
```

- **Options** are `name=value`, separated by commas. Each option is named once.
- **A component** is `type` or `type[params]`. Its params are again `name=value` separated by commas. A comma inside
  brackets belongs to the value, so components nest: `actor=llm[budget=4000,apply=compress[reduce=mean]]`.
- **A model** is `provider:model` or `provider:model[options]`, e.g. `metacentrum:gpt-oss-120b[reasoning_effort=low]`.
  The options are plyllm's model options, bracketed so their commas do not split the component's own options.
- **Defaults are left out.** A config is written back with only the values that differ from the defaults, so a new
  option never changes existing specs or paths. `TextRLParams.from_string(spec).to_string()` gives the canonical form.
- **Booleans** are `true`/`false` (also `True`, `1`, `yes`, `0`, `no`). Numbers are written as Python reads them.
- **Unknown options are errors**, with the list of known ones. So are unknown types and unbalanced brackets.

Where the spec goes:

```json
"players": [{ "value": "textrl:preset=tac,model=metacentrum:gpt-oss-120b", "label": "TAC", "enabled": true }],
"training_configs": [{ "value": "num_epochs=10,num_training_games=10,num_test_games=30", "enabled": true }]
```

The training config is plybench's: `num_epochs`, `num_training_games`, `num_test_games`, and optionally
`update_every=K` (needed by `batch=every`, [§2](#2-the-textrl-options)).

**Paths.** The player's results folder is named from its spec. A suffix over 160 characters keeps its start and adds a
hash of the whole. A preset's path is `preset-<name>-<hash of the whole config>`. `run.json` and `metadata.json` keep
the full spec.

## 2. The TextRL options

| Option | Default | Values | What it does |
|---|---|---|---|
| `actor` | (required) | `llm`, `tabular`, `oracle` | The policy. Every spec names it. [§3.1](#31-actor) |
| `critic` | `none` | `none`, `llm_v`, `llm_q`, `tabular`, `oracle` | The value estimate. [§3.2](#32-critic) |
| `feedback` | `none` | `none`, `llm`, `oracle`, `log` | The joint loss: one loss that teaches both blocks. [§3.3](#33-losses) |
| `actor_loss` | `none` | `none`, `llm`, `oracle` | The actor's own loss $\mathcal{F}_\pi$. [§3.3](#33-losses) |
| `critic_loss` | `none` | `none`, `llm`, `oracle` | The critic's own loss $\mathcal{F}_Q$ (or $\mathcal{F}_V$). [§3.3](#33-losses) |
| `sweep` | `episode` | `episode`, `parallel`, `backward` | The order in which the losses take the steps. [§3.7](#37-sweep) |
| `train_selector` | `greedy` | `greedy`, `eps_greedy`, `ucb` | How a tabular or oracle actor picks in training. [§3.8](#38-selector) |
| `eval_selector` | `greedy` | as above | The same in evaluation. |
| `reward` | `result` | `result`, `length_penalty` | How the game's result becomes the outcome $G$. [§3.9](#39-reward) |
| `key` | `actions` | `actions`, `state` | The observation formalism the memory key is computed in: `actions`, the move history as text; `state`, the state description. Only for retrieval; what an LLM is shown is set by the actor's `observation`. |
| `reference` | `solved` | `solved`, `counts` | What the reference record holds beside the counts per key and move: `solved` adds $V^*$ and $Q^*$ when the game is solvable. Recorded for analysis, shown to no component (except the actor's `counts`, [§3.1](#31-actor)). |
| `batch` | `game` | `game`, `epoch`, `every` | The games one update of the memories learns from: each game alone; all training games of the epoch; or every `update_every` games (plybench `TrainingConfig.update_every`). With `epoch` and `every`, the counts are still committed after every game. |
| `gamma` | `1.0` | $(0, 1]$ | The discount per move of the agent. MC targets are $\gamma^{T-1-t}G$; TD targets use $\gamma$ times the next value. A text target gets a sentence ("it counts gamma times …"), not a product. The oracle components keep undiscounted values. |
| `model` | none | a model | The default model: given to every LLM component that names none, nested ones included (backward, Apply). Resolved when parsing, so the spec equals the same spec written in full. |
| `preset` | none | a preset name | Start from a named config in `src/plyagents/textrl/presets/`. The other options replace its roles. [§2.1](#21-presets) |
| `checkpoint` | none | a path | Set by the harness. Do not write it. |
| `debug` | `false` | bool | Print every component's one-line description when the player is built. Not part of the identity. |

### 2.1 Presets

A preset is a named algorithm: a spec, a description, and what defines the algorithm (`defines`: per role, the type and
the params it must keep). Every other role and param is open: the model, budgets, Apply, step sizes, the backward pass,
$\gamma$, the reward, the batch. `[…]` is `budget=4000,apply=compress[reduce=mean]`.

| Preset | Algorithm | Spec (without `model=`) | Defines |
|---|---|---|---|
| `tac` | The paper's text actor-critic, SAC-Discrete-like ([§5.3](#53-policy-gradient-analogues)) | `actor=llm[…],critic=llm_q[…],actor_loss=llm[lesson_chars=200],critic_loss=llm[target=critic,note_chars=200],sweep=parallel` | `actor=llm[memory=true]`, `critic=llm_q`, `feedback=none`, `actor_loss=llm[inputs=none]`, `critic_loss=llm[target=critic,inputs=none]`, `sweep=parallel` |
| `a2c` | A2C-like | as `tac` with `critic=llm_v[…]` | as `tac` with `critic=llm_v` |
| `mc` | Monte Carlo actor-critic | `critic=llm_q[…]`, `actor_loss=llm[lesson_chars=200,inputs=outcome]`, `critic_loss=llm[target=mc,note_chars=200]`, `sweep=parallel` | `actor=llm[memory=true]`, `critic=llm_q`, `feedback=none`, `actor_loss=llm[inputs=outcome]`, `critic_loss=llm[target=mc,inputs=none]`, `sweep=parallel` |
| `hindsight` | Both losses see the whole game | `critic=llm_q[…]`, `actor_loss=llm[lesson_chars=200,inputs=outcome+trajectory]`, `critic_loss=llm[target=mc,inputs=trajectory,note_chars=200]` (sweep `episode`) | `actor=llm[memory=true]`, `critic=llm_q`, `feedback=none`, `actor_loss=llm[inputs=outcome+trajectory]`, `critic_loss=llm[target=mc,inputs=trajectory]`, `sweep=episode` |
| `reinforce` | REINFORCE-like, no critic | `actor=llm[…],actor_loss=llm[lesson_chars=200,inputs=outcome],sweep=parallel` | `actor=llm[memory=true]`, `critic=none`, `feedback=none`, `actor_loss=llm[inputs=outcome]`, `critic_loss=none`, `sweep=parallel` |
| `joint` | The first TAC design: one joint loss per game | `actor=llm[…],critic=llm_v[…],feedback=llm[lesson_chars=200]` | `actor=llm[memory=true]`, `critic=llm_v`, `feedback=llm`, `actor_loss=none`, `critic_loss=none`, `sweep=episode` |
| `greedy` | Sample mean, greedy | `actor=tabular[scheduler=harmonic]` | `actor=tabular[target=mc,scheduler=harmonic]`, `train_selector=greedy` |
| `eps_greedy` | Sample mean, ε-greedy | `actor=tabular[scheduler=harmonic],train_selector=eps_greedy[eps=0.1]` | as `greedy` with `train_selector=eps_greedy` (`eps` open) |
| `ucb` | UCB1 | `actor=tabular[scheduler=harmonic],train_selector=ucb[c=2.83]` | as `greedy` with `train_selector=ucb` (`c` open) |

The LLM presets name no model; give it with `model=`. Other options replace roles of the preset:
`textrl:preset=tac,model=…,critic=llm_q[budget=8000,apply=compress[reduce=mean],backward=llm],gamma=0.9` is still `tac`.
**An override that changes what the preset defines is refused when the player is built**, e.g. `preset=tac,critic=llm_v`:
`critic=llm_v… is not the preset tac (it defines critic=llm_q); write the spec in full without preset=`. So a preset name
in a path or a plot always means its algorithm. The check runs at build time, not when parsing: a stored spec still reads
back if a preset's definition changes later.

A preset spec is stored with every role written out, so it reads back the same whatever the preset file says later. A
preset and the same spec in full are different players with different paths (and game seeds, which plybench derives
from the path).

## 3. Components

Each subsection lists the types of one role, and for each type its parameters with their defaults.

### 3.1 Actor

The actor chooses the move during the game (its forward pass). It is the only component used in evaluation.

**`llm`**: an LLM plays the move with its memory entry for the key as context. It learns through a textual memory $M$,
one entry per key. It ignores `train_selector` and `eval_selector` on purpose: the LLM chooses the move itself, and no
external selector is injected into an LLM component.

| Param | Default | What it does |
|---|---|---|
| `model` | TextRL `model` | The model that plays. |
| `observation` | `actions` | What the LLM is shown: `actions` (the move history) or `state` (the state description). Independent of `key`. |
| `strategy` | `text` | plybench's output strategy: `text` or `structured`. |
| `memory` | `true` | `false`: reads and writes nothing (the prior). |
| `budget` | none | The entry budget $B$ in characters. None: lessons are only appended, never shortened or cut. |
| `scheduler` | `constant[beta=0.25]` | The step size $\alpha$ of a write ([§3.6](#36-scheduler)). |
| `apply` | `truncate` | How a gradient is written into the entry ([§3.5](#35-apply)). |
| `explore` | `off` | An exploration instruction in training: `off`; `generic` ("trying an action you are unsure about can teach you more…"); `memory` ("judge from your notes … which actions you are still unsure about"). |
| `counts` | `off` | The reference counts at the key in the prompt: `off`; `counts` (how often each move was tried, training only); `request` (the counts and "consider the actions tried rarely or not yet", training only); `stats` (the counts and mean outcome per move, in training and evaluation). |
| `backward` | `identity` | The backward pass ([§3.4](#34-backward)). |

**`tabular`**: a table $Q(z, a)$ per key, starting at 0; the selector picks from it. Learns
$Q(z_t,a_t) \leftarrow Q(z_t,a_t) + \beta\,(y_t - Q(z_t,a_t))$ for every step with a move.

| Param | Default | What it does |
|---|---|---|
| `target` | `mc` | The target $y_t$ ([§3.10](#310-target)): `mc` ($\gamma^{T-1-t}G$) or `critic` ($\gamma\,V(z_{t+1})$, $G$ after the last move). |
| `scheduler` | `constant` (β = 0.1) | The step $\beta$ from the number of updates of the estimate. `harmonic` gives the sample mean. |

**`oracle`**: $Q^*(s,a)$ of the solved game through the selector. Learns nothing. Privileged: it reads the true state.
With `greedy` it plays an optimal move, ties at random.

### 3.2 Critic

The critic assesses the visited steps after the game (its forward pass), with its state from before the update. Its
assessments go to the losses, never to the actor directly.

**`llm_v`**: an LLM assesses each visited situation from the situation and its memory entry $m^V_{s}$, one call per
distinct key of the game. It learns through a textual memory, one entry per key.

**`llm_q`**: an LLM assesses every legal move of each visited situation in one call. Its memory holds one entry per
state and move, key `<state>|<move>`, and it reads the entries of all legal moves. Only the played move's entry is
written. A failed move has no entry, so its note is dropped. An unvisited move's assessment is the LLM's prior.

Both LLM critics take:

| Param | Default | What it does |
|---|---|---|
| `model` | TextRL `model` | The model that assesses. |
| `budget` | none | The entry budget $B$. None: notes are only appended. |
| `scheduler` | `constant[beta=0.4]` | $\alpha_V$, above the actor's 0.25: the critic tracks a policy that changes. |
| `apply` | `truncate` | How a note is written ([§3.5](#35-apply)). |
| `assessment_chars` | none | The length the assessments are asked to keep to (per move for `llm_q`). A request, never a cut. |
| `value` | `false` | Also ask for a number in $[-1, 1]$ (per move for `llm_q`). For diagnostics, and for the loss inputs `delta` (`llm_v`) and `q` (`llm_q`). Never a target for a tabular actor. |
| `backward` | `identity` | The backward pass ([§3.4](#34-backward)). The critic's backward pass is shown the move played, $a_t$; for `llm_q` only the played move's entry and assessment $q_t(a_t)$. |

**`tabular`**: Monte Carlo tables: $V(z)$ for every visited key and $Q(z,a)$ for the move played, both toward
$G_t = \gamma^{T-1-t}G$. Its numbers can be learned toward (`actor=tabular[target=critic]`) and shown as `delta` or `q`.

| Param | Default | What it does |
|---|---|---|
| `scheduler` | `constant` (β = 0.1) | The step from the number of updates. `harmonic`: the sample mean. |

**`oracle`**: $V^*(s)$ and $Q^*(s,a)$ of the solved game (best play by both sides, not the agent's policy). Learns
nothing. Privileged.

**`none`**: no assessment. The losses then see the moves and the outcome only.

### 3.3 Losses

A loss turns the game into a verdict per step: a grade, a reason, a lesson for the actor's memory and/or a note for
the critic's memory. The text of a verdict is the block's feedback ($g_t$ for the actor, $e_t$ for the critic).
**Use either the joint loss (`feedback`) or the own losses (`actor_loss`, `critic_loss`), not both.**

#### Joint loss (`feedback`): one loss for both blocks

**`llm`**: the feedback model $\mathcal{F}$ of the first TAC design. One call per game, the steps numbered, the outcome
shown. It returns per step a grade (`mistake`, `dubious`, `neutral`, `good`, `strong`), a reason, a lesson and a critic
note. Needs `sweep=episode`.

| Param | Default | What it does |
|---|---|---|
| `model` | TextRL `model` | The judging model. |
| `lesson_chars` | none | The length the lessons are asked to keep to. A request, never a cut. |
| `inputs` | `value` | What it is shown besides the moves and the outcome, joined by `+` in any order, or `none`: `value` (the critic's text), `memory` (the entry the actor read), `delta` (the critic's value before and after the move; needs a numeric critic), `q` (the critic's value of every legal move; needs action values), `history` (the moves played in the same situation in earlier games and their results), `reasoning` (the actor's reasoning, and an LLM critic's notes and reasoning). |
| `track` | `false` | Ask the lessons to keep track of which moves were tried and how sure the lesson is. |

**`oracle`**: the verdict of the solved game, no call. A move with $Q^*(s,a) = V^*(s)$ is `good`; one that gives up
value but does not lose is `dubious`; one that loses is a `mistake`. The lesson names the best moves; the critic note
the value under best play. Privileged. Needs `sweep=episode`.

**`log`**: no grade and no call. The lesson of every step is "Played <move>: <result>." With no budget, the actor's
entry becomes the full history of its moves and results (the raw history in the prompt). Needs `sweep=episode`.

#### Actor loss (`actor_loss`): $\mathcal{F}_\pi$

**`llm`**: grades every move by what the critic says of it, and writes a lesson. It gives the critic nothing (the
stop-gradient of actor-critic RL). With `llm_q` it is shown the critic's assessment of every legal move (SAC-like);
with `llm_v` the assessment of the situation before and of the next situation after the move, or the outcome after the
last move (the TD error of A2C). Without `trajectory`, one call per step, shown that step alone; with it, one call per
game.

| Param | Default | What it does |
|---|---|---|
| `model` | TextRL `model` | The judging model. |
| `lesson_chars` | none | The requested lesson length. |
| `inputs` | `none` | Besides the step and the critic's assessment, joined by `+`: `outcome` (the outcome, discounted to the step: Monte Carlo), `trajectory` (the whole game in one call: hindsight; needs `sweep=episode`), `memory`, `delta`, `q`, `history`, `reasoning` (as for the joint loss). `none` is the strict loss. |
| `track` | `false` | As for the joint loss. |

Without a critic it needs `outcome` or `trajectory`, since it has nothing else to judge by.

**`oracle`**: the grade, reason and lesson of the oracle judgement. Privileged.

#### Critic loss (`critic_loss`): $\mathcal{F}_Q$ or $\mathcal{F}_V$

**`llm`**: checks the critic's assessment of every step against its target and writes a critic note where it was
wrong. For `llm_q` it checks the assessment of the move played. A step whose move the Q critic did not assess, or a
failed move, is not checked. Needs a critic.

| Param | Default | What it does |
|---|---|---|
| `model` | TextRL `model` | The checking model. |
| `note_chars` | none | The requested note length. |
| `target` | `mc` | `mc`: the outcome, $\gamma^{T-1-t}G$. `critic`: temporal difference, the critic's own assessment of the next step ($q'_{t+1}(a_{t+1})$ of the move played next, SARSA, for `llm_q`; $v'_{t+1}$ for `llm_v`), and the outcome after the last step. |
| `inputs` | `none` | `outcome` (also show the outcome beside a TD target), `trajectory` (the whole game in one call; needs `sweep=episode`). |

**`oracle`**: the value under best play of the position (V critic) or of the move played (Q critic). Privileged.

### 3.4 Backward

A block's backward pass turns its feedback into the local gradient: the text written to the entry it read. Set per
block: `actor=llm[backward=…]`, `critic=llm_v[backward=…]`, `critic=llm_q[backward=…]`.

**`identity`** (default): the feedback is the gradient, written unchanged. No call.

**`llm`**: an LLM is shown the block's tape of the step and the feedback, and writes the text to add to the entry,
within the share $\alpha B$ the write leaves it. The tape is: the situation, the entry the block read, what it
produced (the actor's move; the critic's assessment, for `llm_q` of the move played), its reasoning, and for a critic
the move played. One call per step with feedback; a step without feedback gets no call. An empty answer writes nothing.

| Param | Default | What it does |
|---|---|---|
| `model` | TextRL `model` | The model of the backward pass. |
| `reasoning` | `true` | Show the block's reasoning for the step. |

### 3.5 Apply

Apply is the optimizer step: it writes a key's gradients into its entry, once per key per update (per game, or per
batch). With a budget $B$: the old entry is shortened to $(1-\alpha)B$ and the new text gets the share $\alpha B$; the
result is kept within $B$ (an over-long new text is cut and flagged `cut_lesson`). Without a budget the new text is only
appended. Set per block: `actor=llm[apply=…]`, `critic=llm_v[apply=…]`.

**`truncate`** (default): keeps the most recent $(1-\alpha)B$ characters of the old entry, cut wherever that falls. No
call. Several gradients of a key are joined.

**`compress`**: an LLM shortens the old entry to $(1-\alpha)B$, keeping every point in proportion and any count of the
lessons that support a point. One call, only when the entry does not fit its share.

| Param | Default | What it does |
|---|---|---|
| `model` | TextRL `model` | The shortening model. |
| `reduce` | `join` | What one write makes of several gradients of a key. `join`: appended in step order (their sum). `mean`: one call combines them, each point with how many of the $n$ gradients support it ("k of n lessons: …"); when the entry must also be shortened, both in one call. |

**`rewrite`**: an LLM rewrites the whole entry with the gradients, told to change about $\alpha$ of it, within $B$, and
with several gradients to state the support of each point. Nothing enforces the step; every write records the realised
change (`changed`: 1 − the similarity of the new entry to the old). An empty entry with one gradient takes it without
a call.

| Param | Default | What it does |
|---|---|---|
| `model` | TextRL `model` | The rewriting model. |

**Batches.** With `batch=epoch|every`, a key visited in several games gets all their gradients in one write. With a
budget, use `compress[reduce=mean]` or `rewrite`: the build refuses a budget with `truncate` or `compress[reduce=join]`,
because the joined text would be cut to the budget and all but the first gradient lost. Without a budget, `join` keeps
every gradient.

### 3.6 Scheduler

The step size from the number of updates so far. For a memory entry it is $\alpha$ of a write; for a table, $\beta$ of
an update.

**`constant`**: `beta` (default 0.1) at every update.

**`harmonic`**: $1/(n+1)$ at the $n$-th update: an estimate that starts at 0 is then the sample mean of its targets.

Both take two options that only affect memory writes (the tabular blocks use the step alone):

| Param | Default | What it does |
|---|---|---|
| `per` | `write` | The step of one write of $n$ gradients. `write`: one step per write (the mean gradient). `sample`: as if the $n$ gradients were applied one after another: $1-(1-\beta)^n$ for `constant`, $n/(k+n)$ for `harmonic` ($k$ gradients before). Use `sample` to compare batch sizes. |
| `cap` | `1.0` | An upper bound on the step, in $(0, 1]$. With `constant`, `per=sample` and a large batch, the step tends to 1; a cap keeps part of the old entry. |

Example: `scheduler=constant[beta=0.25,per=sample,cap=0.8]`.

### 3.7 Sweep

The order in which the critic's forward pass and the losses take the steps of a game. The memories are written after
the sweep, once per key.

| Type | Order | Use with |
|---|---|---|
| `episode` (default) | The critic assesses every step, then each loss judges the game. An own loss without `trajectory` makes one call per step. | The joint loss; a loss with `trajectory`. |
| `parallel` | Every step on its own and at once, every target from the critic before the update (as SAC's target network). No step sees another's verdict. | The strict TD and MC losses. |
| `backward` | From the last step to the first. At step $t$ the critic loss is shown the next assessment together with what its check found; the actor loss is shown the critic's assessment with its check. The outcome reaches the first step in one pass (the text version of episodic backward update). | The own losses, TD. |

`parallel` and `backward` refuse the joint loss and losses with `trajectory`.

### 3.8 Selector

Picks a move from the estimates of a `tabular` or `oracle` actor. The LLM actor ignores it.

| Type | Params | What it does |
|---|---|---|
| `greedy` | none | The best estimate, ties at random. |
| `eps_greedy` | `eps` (0.1) | A random move with probability `eps`, else greedy. |
| `ucb` | `c` (1.0) | UCB1: untried moves first, then estimate + $c\sqrt{\ln N / n}$. For outcomes in $[-1, 1]$, $c = 2.83$ is $\sqrt 2$ rescaled. |

### 3.9 Reward

How plybench's game result becomes the outcome $G$, the one reward, after the agent's last move.

| Type | Params | What it does |
|---|---|---|
| `result` | `my_fail` (−1), `opponent_fail` (+1) | Win +1, draw 0, loss −1, on the scale of the solved values; a failed move of the agent or the opponent gets its own value. |
| `length_penalty` | as `result`, and `penalty` (0.01) | The result minus `penalty` per move of the agent. It leaves the scale of $V^*$, so a target from the oracle critic mixes two scales. |

### 3.10 Target

What a `tabular` actor learns toward (`actor=tabular[target=…]`).

| Type | Target $y_t$ |
|---|---|
| `mc` | $\gamma^{T-1-t}G$, the outcome discounted over the agent's later moves. |
| `critic` | $\gamma\,V(z_{t+1})$, the critic's value of the agent's next situation; $G$ after the last move. Needs `critic=tabular` or `critic=oracle`. |

## 4. Composing a TextRL player

### 4.1 The data flow

1. **During the game** the actor reads its entry $c_t$ for the key $z_t$ and plays $a_t$. The step record keeps the
   key, the move, the entry read and the reasoning.
2. **After each game** (or after the batch), in the order of the sweep:
   - the critic assesses every visited step: $q_t(\cdot)$ (`llm_q`), $v_t$ (`llm_v`), numbers (`tabular`, `oracle`);
   - the losses judge each step: the joint loss gives both blocks' feedback; or $\mathcal{F}_\pi$ gives the actor's
     $g_t$ from the critic's assessment and its inputs, and $\mathcal{F}_Q$ gives the critic's $e_t$ from the
     assessment and its target.
3. **Backward**: each LLM block turns its feedback into a local gradient per step ($\Delta_t$).
4. **Apply**: each LLM block writes, per key, all its gradients of the update in one write. A tabular block applies its
   samples toward its targets.
5. **Evaluation** uses the actor alone, everything frozen.

A block learns only from its own feedback: an LLM actor needs an actor loss or the joint loss; an LLM critic needs a
critic loss or the joint loss. A tabular block learns from the outcome or its target, not from text.

### 4.2 Build rules

Checked when the player is built; the error names what is missing.

| Rule | Error |
|---|---|
| The joint loss or the own losses, not both. | `feedback (the joint loss) or actor_loss and critic_loss, not both` |
| The joint loss (`llm`, `oracle`, `log`) and a loss with `inputs=trajectory` need `sweep=episode`. | `… needs a sweep that shows the losses the whole game at once (sweep=episode)` |
| `inputs=delta` needs a critic with numbers: `oracle`, `tabular`, `llm_v[value=true]`. | `… needs a critic that gives numbers …` |
| `inputs=q` needs action values: `oracle`, `tabular`, `llm_q[value=true]`. | `… needs a critic with action values …` |
| `actor_loss=llm` without `outcome` or `trajectory` needs a critic. | `actor_loss without inputs=outcome or trajectory needs a critic that assesses the steps` |
| `critic_loss=llm` or `oracle` needs a critic. | `critic_loss needs a critic that assesses the steps` |
| `actor=tabular[target=critic]` needs `critic=tabular` or `oracle` (an LLM critic's numbers are for diagnostics). | `actor target=critic needs a critic that gives values to learn toward …` |
| `batch=epoch|every` with a budget needs `compress[reduce=mean]` or `rewrite`. | `batch=… writes all of a key's gradients of the batch at once …` |
| `batch=every` needs `update_every` in the training config. | `batch=every learns from the batches of the harness: set update_every …` (at the end of the first epoch) |

### 4.3 Accepted but without effect

These build, and the part named does nothing. Avoid them unless that is the point.

- `train_selector` / `eval_selector` with `actor=llm`.
- Lessons for `actor=tabular` or `oracle`; critic notes for `critic=tabular`, `oracle` or `none` (an LLM loss still
  makes its calls).
- `per` and `cap` on the scheduler of a tabular block.
- Every write option of `actor=llm[memory=false]`.
- A budget with `apply=compress` that the entry never reaches (the shortening never runs).

### 4.4 Cost per game

$T$ steps of the agent, $K \le T$ keys with feedback. Worst case, all feedback non-empty:

- actor: $T$ calls in the game;
- critic: 1 call per distinct key;
- own losses without `trajectory`: up to $T$ calls each; with `trajectory` or the joint loss: 1 per game per loss;
- LLM backward: up to $T$ per block;
- Apply: up to $K$ per block (`truncate`: none; `compress`: when the entry must be shortened, and with `reduce=mean` when a
  key has several gradients; `rewrite`: every write except an empty entry with one gradient).

## 5. Algorithms from the blocks

Each recipe names the algorithm it follows and where it differs. The model is left out; add `model=…`. References
marked *verify* are from memory and must be checked before they are cited.

### 5.1 Numeric baselines (no LLM)

**Greedy, ε-greedy, UCB1 (bandits).** The sample mean of each arm's outcomes with a selector (presets `greedy`,
`eps_greedy`, `ucb`).

```
actor=tabular[scheduler=harmonic],train_selector=ucb[c=2.83]
```

**Every-visit Monte Carlo control, on-policy, ε-greedy** (Sutton & Barto 2018, ch. 5, *verify*). $Q(z,a)$ toward the
discounted outcome of every visit; `harmonic` gives the sample average, `constant` a recency-weighted one.

```
actor=tabular[scheduler=harmonic],train_selector=eps_greedy[eps=0.1],gamma=0.9
```

**One-step target from a Monte Carlo value.** $Q(z_t,a_t) \leftarrow \gamma V(z_{t+1})$ with $V$ learned by MC. This
is *not* SARSA or Q-learning: the bootstrap is from $V$ of the next situation, not from $Q(s',a')$ or $\max_{a'}$.
Neither of those is implemented for the tabular blocks.

```
actor=tabular[target=critic],critic=tabular[scheduler=harmonic],train_selector=eps_greedy[eps=0.1]
```

**Oracle rungs** (debugging). `actor=oracle` plays optimally. `critic=oracle`, `feedback=oracle`, `actor_loss=oracle`,
`critic_loss=oracle` give the true values and grades, so a failure can be located in one block:

```
actor=llm,critic=oracle,feedback=oracle
```

### 5.2 LLM baselines without a learned critic

**The prior** (zero-shot, no learning):

```
actor=llm[memory=false]
```

**Raw history in the prompt** (H-raw; the in-context bandit setting of Krishnamurthy et al. 2024, *verify*). Every move
and its result appended to the entry, no judgement:

```
actor=llm,feedback=log
```

**Summary statistics in the prompt** (H-stats): the counts and the mean outcome per move, no memory:

```
actor=llm[memory=false,counts=stats]
```

**Verbal reflection after each episode** (in the spirit of Reflexion, Shinn et al. 2023, *verify*). One LLM call per
game sees the moves and the outcome and writes a lesson per step. Differences: Reflexion keeps episode-level
reflections for a retry of the same task; here the lessons are per key and the joint loss also grades.

```
actor=llm,feedback=llm[inputs=none]
```

### 5.3 Policy-gradient analogues

The actor loss plays the role of $\nabla_\theta \log \pi(a_t \mid s_t)\,\hat{A}_t$: it says which move to prefer and
how strongly. What it is shown sets which estimator it imitates.

**REINFORCE** (Williams 1992, *verify*): Monte Carlo, no baseline. The actor loss sees each step and the discounted
outcome; there is no critic. Preset `reinforce`, or in full:

```
actor=llm,actor_loss=llm[inputs=outcome],sweep=parallel
```

**REINFORCE with a learned baseline.** A V critic learned toward the outcome; the actor loss sees $v_t$, $v_{t+1}$ and
the outcome. Close to REINFORCE with baseline, but the loss also sees the next value, so it sits between this and A2C.

```
actor=llm,critic=llm_v,actor_loss=llm[inputs=outcome],critic_loss=llm[target=mc],sweep=parallel
```

**A2C** (advantage actor-critic with a TD target; Mnih et al. 2016 for the asynchronous version, *verify*). Preset
`a2c`: a V critic, a TD critic loss, the strict actor loss shown $v_t$ and $v_{t+1}$ (the TD error), every step on
its own.

```
preset=a2c
```

With a numeric critic, the TD error itself can be shown as a number:

```
actor=llm,critic=tabular,actor_loss=llm[inputs=delta],sweep=parallel
```

**SAC-Discrete-like** (Christodoulou 2019, *verify*). Preset `tac`, the paper's main instance: a Q critic per state and move; the strict actor
loss judges the move against the critic's assessment of every legal move, as SAC-Discrete's actor loss uses
$Q(s, \cdot)$; a TD critic loss; the critic frozen during the sweep (`parallel`, as a target network). Differences from
SAC: no entropy term and no temperature; the TD target is SARSA's $q(s', a')$ of the move played next, not
$\mathbb{E}_{a' \sim \pi} Q(s', a')$ (the LLM actor gives no $\pi(a' \mid s')$); one critic, not two.

```
preset=tac
```

With the critic's numbers shown as well (`llm_q` asked for a value per move):

```
actor=llm,critic=llm_q[value=true],actor_loss=llm[inputs=q],critic_loss=llm[target=critic],sweep=parallel
```

**Monte Carlo actor-critic.** Preset `mc`: the Q critic checked against the outcome, the actor loss shown the outcome
too.

**Hindsight.** Preset `hindsight`: both losses see the whole game and the outcome in one call. Valid, but no longer
SAC: the actor's signal is the trajectory in hindsight, not the critic alone.

### 5.4 Variants of the update

**Episodic backward update** (Lee et al. 2019, *verify*): the steps from last to first, so the outcome reaches the first
step's target in one pass.

```
actor=llm,critic=llm_q,actor_loss=llm,critic_loss=llm[target=critic],sweep=backward
```

**A local backward pass** (as TextGrad's backward through each component, Yuksekgonul et al. 2024, *verify*): each
block rewrites its feedback in terms of the entry it read. `backward=identity` is the ablation.

```
actor=llm[backward=llm],critic=llm_q[backward=llm],actor_loss=llm,critic_loss=llm[target=critic],sweep=parallel
```

**Batches (minibatch updates).** One write per key per epoch, the gradients combined into their mean:

```
actor=llm[budget=4000,apply=compress[reduce=mean]],critic=llm_q[budget=4000,apply=compress[reduce=mean]],actor_loss=llm,critic_loss=llm[target=critic],sweep=parallel,batch=epoch
```

Every K games, with a step that is the same per sample whatever the batch size (`per=sample`):

```
actor=llm[budget=4000,apply=rewrite,scheduler=harmonic[per=sample]],critic=llm_v[budget=4000,apply=rewrite],actor_loss=llm,critic_loss=llm[target=critic],sweep=parallel,batch=every
```

and `update_every=K` in the training config.

**The first TAC design** (one joint loss per game). Preset `joint`, kept to compare with the earlier results.

**Exploration by text.** An instruction and the counts in the actor's prompt in training, on any of the designs:

```
actor=llm[explore=memory,counts=request],critic=llm_q,actor_loss=llm,critic_loss=llm
```

### 5.5 What cannot be built yet

- **Off-policy learning** (Q-learning, a replay buffer): every update is on-policy, from the games just played.
- **Tabular SARSA or Q-learning**: the tabular targets are MC or $\gamma V(z_{t+1})$ only.
- **A tabular actor that learns from an LLM critic**: an LLM critic's numbers are diagnostics, not targets.
- **Entropy regularisation, twin critics, policy probabilities**: the LLM actor gives no $\pi(a \mid s)$.
- **Keys that generalise across similar states**: memories are read by exact key.
