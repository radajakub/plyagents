from __future__ import annotations

from plyagents.tac.configs.components import Collection, ComponentSpec, NoParams, require_model
from plyagents.tac.enums import FeedbackType
from plyagents.tac.feedback.base import Feedback
from plyagents.tac.feedback.llm import LLMFeedback, LLMFeedbackParams
from plyagents.tac.feedback.none import NoFeedback
from plyagents.tac.feedback.oracle import OracleFeedback

FEEDBACKS: Collection[FeedbackType, Feedback] = Collection(
    "feedback",
    FeedbackType,
    {
        FeedbackType.NONE: ComponentSpec(NoParams, lambda params, services: NoFeedback()),
        FeedbackType.ORACLE: ComponentSpec(NoParams, lambda params, services: OracleFeedback()),
        FeedbackType.LLM: ComponentSpec(
            LLMFeedbackParams, lambda params, services: LLMFeedback(require_model(params.model, "llm feedback", services), params.lesson_chars, params.shown)
        ),
    },
)
