from __future__ import annotations

from plybench.utils.enums import ExtendedEnum

# the type of each TAC component, as its config string names it


class ActorType(ExtendedEnum):
    ORACLE = "oracle"
    TABULAR = "tabular"
    LLM = "llm"


class CriticType(ExtendedEnum):
    NONE = "none"
    ORACLE = "oracle"
    TABULAR = "tabular"
    # the LLM critics, named by what they assess: the situation (V) or each legal move (Q)
    LLM_V = "llm_v"
    LLM_Q = "llm_q"


class NotesType(ExtendedEnum):
    # the entries of the LLM Q critic's memory: one per situation, covering all its moves, or one per move of a situation
    SITUATION = "situation"
    MOVE = "move"


class SelectorType(ExtendedEnum):
    GREEDY = "greedy"
    EPS_GREEDY = "eps_greedy"
    UCB = "ucb"


class TargetType(ExtendedEnum):
    MC = "mc"
    CRITIC = "critic"


class SchedulerType(ExtendedEnum):
    CONSTANT = "constant"


class ApplyType(ExtendedEnum):
    TRUNCATE = "truncate"
    COMPRESS = "compress"


class FeedbackType(ExtendedEnum):
    NONE = "none"
    ORACLE = "oracle"
    LLM = "llm"


class ReferenceType(ExtendedEnum):
    # what the player's reference record holds besides the counts: SOLVED adds V* and Q* when the game is solvable
    SOLVED = "solved"
    COUNTS = "counts"


class RewardType(ExtendedEnum):
    RESULT = "result"
    LENGTH_PENALTY = "length_penalty"
