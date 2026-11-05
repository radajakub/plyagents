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
    LLM = "llm"


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


class RewardType(ExtendedEnum):
    RESULT = "result"
    LENGTH_PENALTY = "length_penalty"
