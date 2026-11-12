from __future__ import annotations

from plybench.utils.enums import ExtendedEnum

# the type of each TextRL component, as its config string names it


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


class SelectorType(ExtendedEnum):
    GREEDY = "greedy"
    EPS_GREEDY = "eps_greedy"
    UCB = "ucb"


class TargetType(ExtendedEnum):
    MC = "mc"
    CRITIC = "critic"


class SchedulerType(ExtendedEnum):
    CONSTANT = "constant"
    HARMONIC = "harmonic"


class ApplyType(ExtendedEnum):
    TRUNCATE = "truncate"
    COMPRESS = "compress"
    REWRITE = "rewrite"


class ReduceType(ExtendedEnum):
    # what one write makes of several gradients of a key: joined in step order (their sum), or combined by an LLM into one (their mean)
    JOIN = "join"
    MEAN = "mean"


class ExploreLevel(ExtendedEnum):
    # the exploration instruction in the LLM actor's prompt in training: none, a generic one, or one that points to its notes and knowledge
    OFF = "off"
    GENERIC = "generic"
    MEMORY = "memory"


class CountsLevel(ExtendedEnum):
    # what the LLM actor is shown of the reference counts at its key: nothing; the counts in training; the counts and a request to try rare
    # actions in training; the counts and mean outcomes, in training and evaluation (the statistics as the context, H-stats)
    OFF = "off"
    COUNTS = "counts"
    REQUEST = "request"
    STATS = "stats"


class JointLossType(ExtendedEnum):
    # the joint loss (the feedback model F), configured by the TextRL option `feedback`
    NONE = "none"
    ORACLE = "oracle"
    LLM = "llm"
    LOG = "log"


class ReferenceType(ExtendedEnum):
    # what the player's reference record holds besides the counts: SOLVED adds V* and Q* when the game is solvable
    SOLVED = "solved"
    COUNTS = "counts"


class RewardType(ExtendedEnum):
    RESULT = "result"
    LENGTH_PENALTY = "length_penalty"


class BackwardType(ExtendedEnum):
    # a block's backward pass: turns the feedback it receives into the local gradient of the entries it read
    IDENTITY = "identity"
    LLM = "llm"


class SweepType(ExtendedEnum):
    # the order in which the losses are taken over the steps of a game
    EPISODE = "episode"
    PARALLEL = "parallel"
    BACKWARD = "backward"


class ActorLossType(ExtendedEnum):
    # the actor's own loss F_pi, configured by the TextRL option `actor_loss`
    NONE = "none"
    ORACLE = "oracle"
    LLM = "llm"


class CriticLossType(ExtendedEnum):
    # the critic's own loss F_Q (or F_V), configured by the TextRL option `critic_loss`
    NONE = "none"
    ORACLE = "oracle"
    LLM = "llm"


class ShareRule(ExtendedEnum):
    # the step of one write of n gradients to an entry (scheduler option `per`): one step per write (the mean gradient), or the share the n gradients
    # would keep if applied one after another
    WRITE = "write"
    SAMPLE = "sample"


class BatchType(ExtendedEnum):
    # the games one update of the memories learns from (TextRL option `batch`): each game, every training game of an epoch, or the batches the harness
    # hands over every update_every games (TrainingConfig.update_every)
    GAME = "game"
    EPOCH = "epoch"
    EVERY = "every"


class Capability(ExtendedEnum):
    # what a component gives the components of other roles (ComponentSpec.gives, needs), checked when the player is built: a critic's assessment of
    # the steps, its value of the situation and of every legal move as numbers, values a tabular target may learn toward (not an LLM's diagnostic
    # numbers); a sweep that shows the losses the whole game at once
    ASSESSMENT = "assessment"
    VALUE = "value"
    ACTION_VALUES = "action_values"
    TARGET = "target"
    WHOLE_GAME = "whole_game"
