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

Orchestrates task execution with any smolagents agent, injecting
the Playbook as context and extracting reasoning trajectories.
"""

import re
from typing import Any, List, Optional

from ..agents import MultiStepAgent
from .playbook import Playbook


class ACEGenerator:
    """
    Generator: Orchestrates task execution with Playbook context.

    The Generator role in ACE is responsible for:
    1. Injecting the current Playbook into any smolagents agent
    2. Executing tasks and capturing the reasoning trajectory
    3. Tracking which strategies were used during execution

    Uses existing smolagents agents (CodeAgent, ToolCallingAgent) - no wrappers.

    Attributes:
        agent: The smolagents agent to use for execution
        playbook: The current Playbook for context augmentation
        last_trajectory: The most recent reasoning trajectory
        last_used_strategies: Strategy IDs used in the last run
    """

    ACE_PROMPT_PREFIX = """You have access to proven strategies and knowledge from previous tasks.

{playbook}

When you use a strategy from above, reference it by its ID (e.g., [str-00001]).
This helps track which strategies are most effective.

---

"""

    def __init__(
        self,
        agent: MultiStepAgent,
        playbook: Optional[Playbook] = None,
    ):
        """
        Initialize the ACE Generator.

        Args:
            agent: Any smolagents agent (CodeAgent, ToolCallingAgent, etc.)
            playbook: Initial Playbook (creates empty one if not provided)
        """
        self.agent = agent
        self.playbook = playbook or Playbook()

        # Store original system prompt for playbook injection
        self._original_prompt = getattr(agent, 'system_prompt', None)

        self.last_trajectory: List[dict] = []
        self.last_used_strategies: List[str] = []

        # Inject playbook
        self._inject_playbook()

    def _inject_playbook(self) -> None:
        """Inject current playbook into agent's system prompt."""
        playbook_content = self.playbook.render()

        if playbook_content.strip():
            prefix = self.ACE_PROMPT_PREFIX.format(playbook=playbook_content)
        else:
            prefix = ""

        if self._original_prompt:
            self.agent.system_prompt = prefix + self._original_prompt
        elif prefix:
            self.agent.system_prompt = prefix.rstrip("\n-")

    def update_playbook(self, playbook: Playbook) -> None:
        """
        Update the playbook and re-inject into agent.

        Args:
            playbook: The new Playbook to use
        """
        self.playbook = playbook
        self._inject_playbook()

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
        # Execute the task with the agent
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
                full_matches = strategy_pattern.findall(model_output)
                for _ in full_matches:
                    all_ids = re.findall(r'\[(str|cal|mis)-\d{5}\]', model_output)
                    for match in all_ids:
                        used_strategies.add(match.strip('[]'))

        return list(used_strategies)

    def get_agent(self) -> MultiStepAgent:
        """Get the underlying smolagents agent instance."""
        return self.agent

    def __repr__(self) -> str:
        return f"ACEGenerator(agent={type(self.agent).__name__}, playbook={self.playbook})"
