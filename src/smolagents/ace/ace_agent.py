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
ACEAgent: Orchestrator for self-improving agents.

Orchestrates the ACE (Agentic Context Engineering) loop around any smolagents agent.
Uses smolagents' native mechanisms (instructions, step_callbacks, managed_agents).
"""

from typing import Any, Callable, List, Optional
import logging

from ..agents import MultiStepAgent
from ..memory import ActionStep
from .playbook import Playbook
from .generator import ACEGenerator
from .reflector import ACEReflector
from .curator import ACECurator

logger = logging.getLogger(__name__)


# Try to import PlanningStep (may not exist in all versions)
try:
    from ..memory import PlanningStep
    HAS_PLANNING_STEP = True
except ImportError:
    PlanningStep = None
    HAS_PLANNING_STEP = False


class ACEAgent:
    """
    ACE Orchestrator - coordinates self-improvement around any smolagents agent.

    Uses smolagents' native mechanisms:
    - `instructions` for playbook injection (not system_prompt hacking)
    - `step_callbacks` for real-time trajectory capture
    - `PlanningStep` integration for plan-aware reflection
    - Compatible with `managed_agents` for multi-agent systems

    Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                      ACEAgent (Orchestrator)                     │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
    │   │ ACEGenerator│ ──► │ACEReflector │ ──► │ ACECurator  │       │
    │   │             │     │             │     │             │       │
    │   │ instructions│     │ Analyze     │     │ Delta       │       │
    │   │ + callbacks │     │ trajectory  │     │ updates     │       │
    │   └──────┬──────┘     └─────────────┘     └─────────────┘       │
    │          │                                       │               │
    │          ▼                                       ▼               │
    │   ┌─────────────┐              ┌─────────────────────┐          │
    │   │ CodeAgent / │              │      PLAYBOOK       │          │
    │   │ToolCalling  │◄─────────────│ (via instructions)  │          │
    │   └─────────────┘              └─────────────────────┘          │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
    ```

    Can also be used as a managed_agent in multi-agent systems:
    ```python
    ace = ACEAgent(worker_agent, name="learner", description="Learns from tasks")
    manager = CodeAgent(managed_agents=[ace])
    ```

    Attributes:
        generator: The Generator role (task execution with playbook)
        reflector: The Reflector role (insight extraction)
        curator: The Curator role (playbook updates)
        playbook: The evolving Playbook with strategies and insights
        name: Name for use as managed_agent (optional)
        description: Description for use as managed_agent (optional)
    """

    def __init__(
        self,
        agent: MultiStepAgent,
        playbook: Optional[Playbook] = None,
        auto_improve: bool = True,
        reflect_on_planning: bool = True,
        similarity_threshold: float = 0.85,
        prune_threshold: int = -3,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize the ACE Orchestrator.

        Args:
            agent: Any smolagents agent (CodeAgent, ToolCallingAgent, etc.)
            playbook: Initial Playbook (creates empty one if not provided)
            auto_improve: Whether to run reflection/curation after each task
            reflect_on_planning: Whether to reflect after PlanningSteps
            similarity_threshold: Similarity threshold for deduplication
            prune_threshold: Score threshold for pruning harmful entries
            name: Name for use as managed_agent (optional)
            description: Description for use as managed_agent (optional)
        """
        self.playbook = playbook or Playbook()
        self.auto_improve = auto_improve
        self.reflect_on_planning = reflect_on_planning

        # For managed_agent compatibility
        self.name = name
        self.description = description

        # Initialize the three ACE roles
        self.generator = ACEGenerator(
            agent=agent,
            playbook=self.playbook,
            use_step_callbacks=True,
        )
        self.reflector = ACEReflector(model=agent.model)
        self.curator = ACECurator(
            playbook=self.playbook,
            similarity_threshold=similarity_threshold,
            prune_threshold=prune_threshold,
        )

        # Register PlanningStep callback if available and enabled
        if reflect_on_planning and HAS_PLANNING_STEP:
            self._register_planning_callback()

        # Statistics
        self.run_count = 0
        self.improvement_count = 0
        self._last_reflection = None
        self._last_curation_stats = None

    def _register_planning_callback(self) -> None:
        """Register callback to reflect after PlanningSteps."""
        existing_callbacks = getattr(self.agent, 'step_callbacks', None) or {}

        if isinstance(existing_callbacks, dict):
            planning_callbacks = existing_callbacks.get(PlanningStep, [])
            if not isinstance(planning_callbacks, list):
                planning_callbacks = [planning_callbacks]
            planning_callbacks.append(self._planning_step_callback)
            existing_callbacks[PlanningStep] = planning_callbacks
            self.agent.step_callbacks = existing_callbacks

    def _planning_step_callback(self, step, agent: MultiStepAgent) -> None:
        """Callback after PlanningStep - opportunity for plan-aware reflection."""
        logger.debug(f"ACE: PlanningStep detected, could inject plan-aware strategies")
        # Future: Could analyze the plan and suggest relevant strategies

    @property
    def agent(self) -> MultiStepAgent:
        """Access the underlying smolagents agent."""
        return self.generator.agent

    def run(self, task: str, improve: Optional[bool] = None, **kwargs) -> Any:
        """
        Execute a task with optional self-improvement.

        Args:
            task: The task description to execute
            improve: Override auto_improve for this run (optional)
            **kwargs: Additional arguments passed to agent.run()

        Returns:
            The result of the task execution
        """
        should_improve = improve if improve is not None else self.auto_improve

        # 1. Generator: Execute task with playbook context
        logger.info(f"ACE Generator executing task: {task[:100]}...")
        gen_result = self.generator.run(task)
        self.run_count += 1

        if should_improve:
            # 2. Reflector: Analyze the execution
            logger.info("ACE Reflector analyzing execution...")
            reflection = self.reflector.reflect(gen_result)
            self._last_reflection = reflection

            if reflection.get("parse_error"):
                logger.warning(f"Reflection parse error: {reflection['parse_error']}")
            else:
                # 3. Curator: Update the playbook
                logger.info("ACE Curator updating playbook...")
                _, curation_stats = self.curator.curate(reflection)
                self._last_curation_stats = curation_stats
                self.improvement_count += 1

                # 4. Update generator with new playbook
                self.generator.update_playbook(self.playbook)

                logger.info(
                    f"ACE improvement complete: "
                    f"+{curation_stats['insights_added']} insights, "
                    f"{curation_stats['counters_updated']} counters updated"
                )

        return gen_result["result"]

    def __call__(self, task: str, **kwargs) -> Any:
        """
        Make ACEAgent callable for managed_agent compatibility.

        This allows ACEAgent to be used as a managed_agent:
        ```python
        manager = CodeAgent(managed_agents=[ace_agent])
        ```
        """
        return self.run(task, **kwargs)

    def improve(self, reflection: Optional[dict] = None) -> dict:
        """
        Manually trigger an improvement cycle.

        Args:
            reflection: Optional custom reflection to use
                       (uses last reflection if not provided)

        Returns:
            Curation statistics
        """
        reflection = reflection or self._last_reflection

        if not reflection:
            raise ValueError("No reflection available. Run a task first or provide a reflection.")

        _, stats = self.curator.curate(reflection)
        self.generator.update_playbook(self.playbook)
        self.improvement_count += 1

        return stats

    def stats(self) -> dict:
        """Get comprehensive statistics about the ACE agent."""
        return {
            "runs": self.run_count,
            "improvements": self.improvement_count,
            "auto_improve": self.auto_improve,
            "playbook": self.playbook.stats(),
            "last_curation": self._last_curation_stats,
        }

    def show_playbook(self, max_entries_per_section: Optional[int] = None) -> str:
        """Get a formatted string representation of the current Playbook."""
        return self.playbook.render(max_entries_per_section)

    def save_playbook(self, path: str) -> None:
        """Save the current Playbook to a JSON file."""
        self.playbook.save(path)
        logger.info(f"Playbook saved to {path}")

    def load_playbook(self, path: str) -> None:
        """Load a Playbook from a JSON file."""
        self.playbook = Playbook.load(path)
        self.curator.playbook = self.playbook
        self.generator.update_playbook(self.playbook)
        logger.info(f"Playbook loaded from {path} (version {self.playbook.version})")

    @classmethod
    def from_playbook(
        cls,
        path: str,
        agent: MultiStepAgent,
        **kwargs,
    ) -> "ACEAgent":
        """
        Create an ACEAgent with a pre-existing Playbook.

        Args:
            path: Path to the playbook JSON file
            agent: Any smolagents agent
            **kwargs: Additional arguments passed to ACEAgent

        Returns:
            New ACEAgent instance with loaded playbook
        """
        playbook = Playbook.load(path)
        return cls(agent=agent, playbook=playbook, **kwargs)

    def reset_playbook(self) -> None:
        """Reset the Playbook to empty state."""
        self.playbook = Playbook()
        self.curator.playbook = self.playbook
        self.curator.clear_embedding_cache()
        self.generator.update_playbook(self.playbook)
        logger.info("Playbook reset to empty state")

    def get_last_reflection(self) -> Optional[dict]:
        """Get the last reflection result."""
        return self._last_reflection

    def get_last_curation_stats(self) -> Optional[dict]:
        """Get the last curation statistics."""
        return self._last_curation_stats

    def get_last_trajectory(self) -> List[dict]:
        """Get the last execution trajectory."""
        return self.generator.last_trajectory

    def merge_playbook(self, other_playbook: Playbook) -> int:
        """
        Merge another Playbook into this agent's Playbook.

        Args:
            other_playbook: The Playbook to merge from

        Returns:
            Number of new entries added
        """
        added = 0
        for entry in other_playbook.entries.values():
            if not self.curator._is_duplicate(entry.content):
                new_entry = self.playbook.add_entry(
                    section=entry.section,
                    content=entry.content,
                    metadata=entry.metadata,
                )
                new_entry.helpful_count = entry.helpful_count
                new_entry.harmful_count = entry.harmful_count
                added += 1

        if added > 0:
            self.generator.update_playbook(self.playbook)

        logger.info(f"Merged {added} entries from other playbook")
        return added

    def __repr__(self) -> str:
        name_str = f", name='{self.name}'" if self.name else ""
        return (
            f"ACEAgent("
            f"agent={type(self.agent).__name__}, "
            f"playbook={self.playbook}, "
            f"runs={self.run_count}"
            f"{name_str})"
        )
