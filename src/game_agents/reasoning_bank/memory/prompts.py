# Ported from WebArena/prompts/memory_instruction.py. Every deviation from the original is a
# re-domaining of the task ("web navigation" -> turn-based games); the structure, the item cap,
# the "think first" instruction and the output format are kept verbatim so that results stay
# comparable to the ReasoningBank paper.

SUCCESSFUL_SI = """
You are an expert in turn-based strategy games. You will be given a description of a game, the corresponding trajectory that represents **how an agent successfully won the game**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar games.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable procedures over abstract principles. Do not embed specific board coordinates, move labels, or literal state strings from this game.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing similar games in the future>
```
"""

FAILED_SI = """
You are an expert in turn-based strategy games. You will be given a description of a game, the corresponding trajectory that represents **how an agent attempted to win the game but failed**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar games.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable recovery procedures over abstract principles. Do not embed specific board coordinates, move labels, or literal state strings from this game.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to avoid such failures and successfully accomplishing similar games in the future>
```
"""


# task description wrapped around the query before scoring, as in memory_management.screening
RETRIEVAL_INSTRUCTION = "Given the prior game-playing queries, your task is to analyze a current query's intent and select relevant prior queries that could help resolve it."
