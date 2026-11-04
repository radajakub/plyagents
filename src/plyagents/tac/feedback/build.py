from __future__ import annotations

from plyagents.tac.configs.components import Collection, ComponentSpec, NoParams
from plyagents.tac.enums import FeedbackType
from plyagents.tac.feedback.base import Feedback
from plyagents.tac.feedback.none import NoFeedback

FEEDBACKS: Collection[FeedbackType, Feedback] = Collection(
    "feedback",
    FeedbackType,
    {
        FeedbackType.NONE: ComponentSpec(NoParams, lambda params, services: NoFeedback()),
    },
)
