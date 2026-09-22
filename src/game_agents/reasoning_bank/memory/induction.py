from __future__ import annotations

from dataclasses import replace

from plybench.llm import LLM, LLMMessage, ModelConfig

from game_agents.reasoning_bank.memory.item import split_memory_items
from game_agents.reasoning_bank.memory.prompts import FAILED_SI, SUCCESSFUL_SI
from game_agents.reasoning_bank.trajectory import Trajectory

# the paper induces at temperature 1.0 for ReasoningBank items
INDUCTION_TEMPERATURE = 1.0


class MemoryInducer:
    def __init__(self, llm: LLM, model: ModelConfig) -> None:
        self._llm = llm
        self._model = replace(model, options=replace(model.options, temperature=INDUCTION_TEMPERATURE))

    async def induce(self, trajectory: Trajectory, success: bool, judge_thoughts: str) -> list[str]:
        content = trajectory.to_prompt()
        if judge_thoughts:
            content += f"\n\nThe task {'succeeded' if success else 'failed'} because: {judge_thoughts}"

        instruction = SUCCESSFUL_SI if success else FAILED_SI
        response = await self._llm.generate(self._model, LLMMessage.system(instruction), [LLMMessage.user(content)])
        return split_memory_items(response.output_text)
