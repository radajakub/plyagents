from __future__ import annotations

from plyagents.textrl.configs.components import Collection, ComponentSpec, Need, NoParams, require_model
from plyagents.textrl.enums import ActorLossType, Capability, CriticLossType, JointLossType
from plyagents.textrl.losses.base import ActorLoss, CriticLoss, JointLoss
from plyagents.textrl.losses.inputs import ShownInputs
from plyagents.textrl.losses.llm_actor import LLMActorLoss, LLMActorLossParams
from plyagents.textrl.losses.llm_critic import LLMCriticLoss, LLMCriticLossParams
from plyagents.textrl.losses.llm_joint import LLMJointLoss, LLMJointLossParams
from plyagents.textrl.losses.log import LogJointLoss
from plyagents.textrl.losses.none import NoActorLoss, NoCriticLoss, NoJointLoss
from plyagents.textrl.losses.oracle import OracleActorLoss, OracleCriticLoss, OracleJointLoss

# a joint loss judges the whole game at once, so it needs the sweep that shows it the whole game
_WHOLE_GAME = Need("sweep", Capability.WHOLE_GAME)
# a loss's own judgement needs the critic's assessments to judge by
_ASSESSMENTS = Need("critic", Capability.ASSESSMENT)


def _input_needs(params: ShownInputs) -> tuple[Need, ...]:
    # the inputs of an LLM loss that need another component: the critic's numbers (delta, q), the whole game at once (trajectory)
    needs = {
        "trajectory": Need("sweep", Capability.WHOLE_GAME, "inputs=trajectory"),
        "delta": Need("critic", Capability.VALUE, "inputs=delta"),
        "q": Need("critic", Capability.ACTION_VALUES, "inputs=q"),
    }
    return tuple(need for name, need in needs.items() if name in params.shown)


def _llm_actor_loss_needs(params: LLMActorLossParams) -> tuple[Need, ...]:
    # without a critic it has nothing to judge the moves by, unless it is shown the outcome or the whole game
    judged_alone = () if params.shown & {"outcome", "trajectory"} else (Need("critic", Capability.ASSESSMENT, "without inputs=outcome or trajectory"),)
    return judged_alone + _input_needs(params)


JOINT_LOSSES: Collection[JointLossType, JointLoss] = Collection(
    "feedback",
    JointLossType,
    {
        JointLossType.NONE: ComponentSpec(NoParams, lambda params, services: NoJointLoss()),
        JointLossType.ORACLE: ComponentSpec(NoParams, lambda params, services: OracleJointLoss(), needs=lambda params: (_WHOLE_GAME,)),
        JointLossType.LLM: ComponentSpec(
            LLMJointLossParams,
            lambda params, services: LLMJointLoss(require_model(params.model, "llm feedback", services), params.lesson_chars, params.shown, params.track, services.gamma),
            needs=lambda params: (_WHOLE_GAME, *_input_needs(params)),
        ),
        JointLossType.LOG: ComponentSpec(NoParams, lambda params, services: LogJointLoss(), needs=lambda params: (_WHOLE_GAME,)),
    },
)

ACTOR_LOSSES: Collection[ActorLossType, ActorLoss] = Collection(
    "actor_loss",
    ActorLossType,
    {
        ActorLossType.NONE: ComponentSpec(NoParams, lambda params, services: NoActorLoss()),
        ActorLossType.ORACLE: ComponentSpec(NoParams, lambda params, services: OracleActorLoss()),
        ActorLossType.LLM: ComponentSpec(
            LLMActorLossParams,
            lambda params, services: LLMActorLoss(require_model(params.model, "llm actor_loss", services), params.lesson_chars, params.shown, params.track, services.gamma),
            needs=_llm_actor_loss_needs,
        ),
    },
)

CRITIC_LOSSES: Collection[CriticLossType, CriticLoss] = Collection(
    "critic_loss",
    CriticLossType,
    {
        CriticLossType.NONE: ComponentSpec(NoParams, lambda params, services: NoCriticLoss()),
        CriticLossType.ORACLE: ComponentSpec(NoParams, lambda params, services: OracleCriticLoss(), needs=lambda params: (_ASSESSMENTS,)),
        CriticLossType.LLM: ComponentSpec(
            LLMCriticLossParams,
            lambda params, services: LLMCriticLoss(require_model(params.model, "llm critic_loss", services), params.note_chars, params.shown, params.target, services.gamma),
            needs=lambda params: (_ASSESSMENTS, *_input_needs(params)),
        ),
    },
)
