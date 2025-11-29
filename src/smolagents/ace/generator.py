# coding=utf-8
# Copyright 2024 HuggingFace Inc. team and ACE contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generator role for ACE (Agentic Context Engineering).

The Generator produces reasoning trajectories for tasks using
the current Playbook as context augmentation.
"""

import re
from typing import Any, List, Optional

from ..agents import CodeAgent
from .playbook import Playbook


class ACEGenerator:
    """
    Generator: Produces reasoning trajectories for new queries.

    The Generator role in ACE is responsible for:
    1. Injecting the current Playbook into the system prompt
    2. Executing tasks and capturing the reasoning trajectory
    3. Tracking which strategies were used during execution

    Attributes:
        playbook: The current Playbook for context augmentation
        agent: The underlying CodeAgent for task execution
        last_trajectory: The most recent reasoning trajectory
        last_used_strategies: Strategy IDs used in the last run
    """

    DEFAULT_ACE_PROMPT = """You are an expert agent with access to proven strategies and knowledge.

Follow these proven strategies and insights when solving tasks:

{playbook}

When you use a strategy from the playbook, reference it by its ID (e.g., [str-00001]).
This helps track which strategies are most effective.

Think step by step and apply relevant strategies from the playbook when applicable.
"""

    def __init__(
        self,
        model: Any,
        tools: Optional[List] = None,
        playbook: Optional[Playbook] = None,
        max_steps: int = 10,
        verbosity_level: int = 1,
        **agent_kwargs,
    ):
        """
        Initialize the ACE Generator.

        Args:
            model: The LLM model to use
            tools: List of tools available to the agent
            playbook: Initial Playbook (creates empty one if not provided)
            max_steps: Maximum steps for agent execution
            verbosity_level: Verbosity level for agent output
            **agent_kwargs: Additional arguments passed to CodeAgent
        """
        self.playbook = playbook or Playbook()
        self.model = model
        self.tools = tools or []
        self.max_steps = max_steps
        self.verbosity_level = verbosity_level
        self.agent_kwargs = agent_kwargs

        self.last_trajectory: List[dict] = []
        self.last_used_strategies: List[str] = []

        self._build_agent()

    def _build_agent(self) -> None:
        """Build or rebuild the agent with current playbook."""
        system_prompt = self._build_prompt()

        self.agent = CodeAgent(
            model=self.model,
            tools=self.tools,
            system_prompt=system_prompt,
            max_steps=self.max_steps,
            verbosity_level=self.verbosity_level,
            **self.agent_kwargs,
        )

    def _build_prompt(self) -> str:
        """Build system prompt with Playbook injection."""
        playbook_content = self.playbook.render()
        if not playbook_content.strip():
            playbook_content = "(No strategies accumulated yet - this will grow as you complete tasks)"

        return self.DEFAULT_ACE_PROMPT.format(playbook=playbook_content)

    def update_playbook(self, playbook: Playbook) -> None:
        """
        Update the playbook and rebuild the agent.

        Args:
            playbook: The new Playbook to use
        """
        self.playbook = playbook
        self._build_agent()

    def run(self, task: str) -> dict:
        """
        Execute a task and capture the reasoning trajectory.

        Args:
            task: The task description to execute

        Returns:
            Dictionary containing:
            - task: The original task
            - result: The final result
            - trajectory: List of reasoning steps
            - used_strategies: List of strategy IDs referenced
        """
        # Execute the task
        result = self.agent.run(task)

        # Capture trajectory from memory
        trajectory = self._extract_trajectory()
        self.last_trajectory = trajectory

        # Find which strategies were referenced
        used_strategies = self._extract_used_strategies(trajectory)
        self.last_used_strategies = used_strategies

        return {
            "task": task,
            "result": result,
            "trajectory": trajectory,
            "used_strategies": used_strategies,
        }

    def _extract_trajectory(self) -> List[dict]:
        """Extract reasoning trajectory from agent memory."""
        trajectory = []

        if hasattr(self.agent, 'memory') and self.agent.memory:
            for step in self.agent.memory.steps:
                step_dict = {
                    "type": type(step).__name__,
                }

                # Extract relevant fields based on step type
                if hasattr(step, 'model_output'):
                    step_dict["model_output"] = str(step.model_output) if step.model_output else None
                if hasattr(step, 'tool_calls'):
                    step_dict["tool_calls"] = [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in (step.tool_calls or [])
                    ]
                if hasattr(step, 'observations'):
                    step_dict["observations"] = step.observations
                if hasattr(step, 'error'):
                    step_dict["error"] = str(step.error) if step.error else None

                trajectory.append(step_dict)

        return trajectory

    def _extract_used_strategies(self, trajectory: List[dict]) -> List[str]:
        """
        Extract strategy IDs referenced in the trajectory.

        Looks for patterns like [str-00001], [cal-00002], [mis-00003]
        in the model outputs.

        Args:
            trajectory: The reasoning trajectory

        Returns:
            List of unique strategy IDs that were referenced
        """
        strategy_pattern = re.compile(r'\[(str|cal|mis)-\d{5}\]')
        used_strategies = set()

        for step in trajectory:
            model_output = step.get("model_output", "")
            if model_output:
                matches = strategy_pattern.findall(model_output)
                for match in matches:
                    # Reconstruct full ID from the match
                    full_matches = re.findall(r'\[(str|cal|mis)-\d{5}\]', model_output)
                    for full_match in full_matches:
                        used_strategies.add(full_match.strip('[]'))

        return list(used_strategies)

    def get_agent(self) -> CodeAgent:
        """Get the underlying CodeAgent instance."""
        return self.agent

    def __repr__(self) -> str:
        return f"ACEGenerator(playbook={self.playbook}, tools={len(self.tools)})"
