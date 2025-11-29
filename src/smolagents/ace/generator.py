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
the Playbook as context via smolagents' native `instructions` mechanism.
"""

import re
from typing import Any, Callable, List, Optional, Union

from ..agents import MultiStepAgent
from ..memory import ActionStep
from .playbook import Playbook


class ACEGenerator:
    """
    Generator: Orchestrates task execution with Playbook context.

    The Generator role in ACE is responsible for:
    1. Injecting the current Playbook via smolagents' `instructions`
    2. Optionally using step_callbacks for real-time trajectory capture
    3. Tracking which strategies were used during execution

    Uses smolagents' native mechanisms - no monkey-patching.

    Attributes:
        agent: The smolagents agent to use for execution
        playbook: The current Playbook for context augmentation
        use_step_callbacks: Whether to use real-time step callbacks
    """

    PLAYBOOK_INSTRUCTIONS = """
## Proven Strategies & Knowledge

You have access to proven strategies from previous successful tasks:

{playbook}

**Important:** When you use a strategy from above, reference it by its ID (e.g., [str-00001]).
This helps track which strategies are most effective.
"""

    def __init__(
        self,
        agent: MultiStepAgent,
        playbook: Optional[Playbook] = None,
        use_step_callbacks: bool = True,
    ):
        """
        Initialize the ACE Generator.

        Args:
            agent: Any smolagents agent (CodeAgent, ToolCallingAgent, etc.)
            playbook: Initial Playbook (creates empty one if not provided)
            use_step_callbacks: Use step_callbacks for real-time capture
        """
        self.agent = agent
        self.playbook = playbook or Playbook()
        self.use_step_callbacks = use_step_callbacks

        # Store original instructions to preserve them
        self._original_instructions = getattr(agent, 'instructions', None)

        # Trajectory capture
        self._current_trajectory: List[dict] = []
        self._step_callback_registered = False

        # Inject playbook
        self._inject_playbook()

        # Register step callback if enabled
        if use_step_callbacks:
            self._register_step_callback()

    def _inject_playbook(self) -> None:
        """Inject current playbook via smolagents' instructions mechanism."""
        playbook_content = self.playbook.render()

        if playbook_content.strip():
            playbook_instructions = self.PLAYBOOK_INSTRUCTIONS.format(
                playbook=playbook_content
            )
        else:
            playbook_instructions = ""

        # Combine with original instructions
        if self._original_instructions:
            new_instructions = playbook_instructions + "\n\n" + self._original_instructions
        else:
            new_instructions = playbook_instructions

        # Use smolagents' native instructions attribute
        self.agent.instructions = new_instructions.strip() if new_instructions.strip() else None

    def _register_step_callback(self) -> None:
        """Register step callback for real-time trajectory capture."""
        if self._step_callback_registered:
            return

        # Get existing callbacks
        existing_callbacks = getattr(self.agent, 'step_callbacks', None) or []

        # Handle both list and dict formats
        if isinstance(existing_callbacks, dict):
            # Add to ActionStep callbacks
            action_callbacks = existing_callbacks.get(ActionStep, [])
            if not isinstance(action_callbacks, list):
                action_callbacks = [action_callbacks]
            action_callbacks.append(self._step_callback)
            existing_callbacks[ActionStep] = action_callbacks
            self.agent.step_callbacks = existing_callbacks
        else:
            # List format
            if not isinstance(existing_callbacks, list):
                existing_callbacks = list(existing_callbacks) if existing_callbacks else []
            existing_callbacks.append(self._step_callback)
            self.agent.step_callbacks = existing_callbacks

        self._step_callback_registered = True

    def _step_callback(self, step: ActionStep, agent: MultiStepAgent) -> None:
        """Callback executed after each step for real-time capture."""
        step_dict = {
            "type": type(step).__name__,
            "step_number": getattr(step, 'step_number', None),
            "model_output": str(step.model_output) if hasattr(step, 'model_output') and step.model_output else None,
            "observations": getattr(step, 'observations', None),
            "error": str(step.error) if hasattr(step, 'error') and step.error else None,
        }

        if hasattr(step, 'tool_calls') and step.tool_calls:
            step_dict["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in step.tool_calls
            ]

        self._current_trajectory.append(step_dict)

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
        # Clear trajectory for new run
        self._current_trajectory = []

        # Execute the task with the agent
        result = self.agent.run(task)

        # Get trajectory - prefer real-time if available, else extract from memory
        if self.use_step_callbacks and self._current_trajectory:
            trajectory = self._current_trajectory.copy()
        else:
            trajectory = self._extract_trajectory_from_memory()

        # Find which strategies were referenced
        used_strategies = self._extract_used_strategies(trajectory)

        return {
            "task": task,
            "result": result,
            "trajectory": trajectory,
            "used_strategies": used_strategies,
        }

    def _extract_trajectory_from_memory(self) -> List[dict]:
        """Extract reasoning trajectory from agent memory (fallback)."""
        trajectory = []

        if hasattr(self.agent, 'memory') and self.agent.memory:
            for step in self.agent.memory.steps:
                step_dict = {"type": type(step).__name__}

                if hasattr(step, 'model_output'):
                    step_dict["model_output"] = str(step.model_output) if step.model_output else None
                if hasattr(step, 'tool_calls') and step.tool_calls:
                    step_dict["tool_calls"] = [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in step.tool_calls
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
        """
        strategy_pattern = re.compile(r'\[(str|cal|mis)-\d{5}\]')
        used_strategies = set()

        for step in trajectory:
            model_output = step.get("model_output", "")
            if model_output:
                matches = strategy_pattern.findall(model_output)
                if matches:
                    all_ids = re.findall(r'(str|cal|mis)-\d{5}', model_output)
                    used_strategies.update(all_ids)

        return list(used_strategies)

    @property
    def last_trajectory(self) -> List[dict]:
        """Get the last captured trajectory."""
        return self._current_trajectory.copy()

    def get_agent(self) -> MultiStepAgent:
        """Get the underlying smolagents agent instance."""
        return self.agent

    def __repr__(self) -> str:
        return f"ACEGenerator(agent={type(self.agent).__name__}, playbook={self.playbook})"
