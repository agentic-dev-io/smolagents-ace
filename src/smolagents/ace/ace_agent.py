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
Uses existing agents (CodeAgent, ToolCallingAgent) - no inheritance, no wrappers.
"""

from typing import Any, Optional
import logging

from ..agents import MultiStepAgent
from .playbook import Playbook
from .generator import ACEGenerator
from .reflector import ACEReflector
from .curator import ACECurator

logger = logging.getLogger(__name__)


class ACEAgent:
    """
    ACE Orchestrator - coordinates self-improvement around any smolagents agent.

    Uses existing smolagents agents (CodeAgent, ToolCallingAgent) directly.
    No inheritance, no wrappers - just orchestration.

    Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                      ACEAgent (Orchestrator)                     │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
    │   │    AGENT    │ ──► │  REFLECTOR  │ ──► │   CURATOR   │       │
    │   │ (CodeAgent/ │     │             │     │             │       │
    │   │ ToolCalling)│     │ Evaluate &  │     │ Delta       │       │
    │   │             │     │ Extract     │     │ Updates     │       │
    │   └─────────────┘     └─────────────┘     └─────────────┘       │
    │         │                   │                   │                │
    │         ▼                   ▼                   ▼                │
    │   ┌─────────────────────────────────────────────────────┐       │
    │   │                    PLAYBOOK                          │       │
    │   │  (injected into agent's system_prompt)               │       │
    │   └─────────────────────────────────────────────────────┘       │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
    ```

    Attributes:
        generator: The Generator role (task execution with playbook)
        reflector: The Reflector role (insight extraction)
        curator: The Curator role (playbook updates)
        playbook: The evolving Playbook with strategies and insights
        auto_improve: Whether to automatically improve after each run
    """

    def __init__(
        self,
        agent: MultiStepAgent,
        playbook: Optional[Playbook] = None,
        auto_improve: bool = True,
        similarity_threshold: float = 0.85,
        prune_threshold: int = -3,
    ):
        """
        Initialize the ACE Orchestrator.

        Args:
            agent: Any smolagents agent (CodeAgent, ToolCallingAgent, etc.)
            playbook: Initial Playbook (creates empty one if not provided)
            auto_improve: Whether to run reflection/curation after each task
            similarity_threshold: Similarity threshold for deduplication
            prune_threshold: Score threshold for pruning harmful entries
        """
        self.playbook = playbook or Playbook()
        self.auto_improve = auto_improve

        # Initialize the three ACE roles - all orchestrate the same agent
        self.generator = ACEGenerator(agent=agent, playbook=self.playbook)
        self.reflector = ACEReflector(model=agent.model)
        self.curator = ACECurator(
            playbook=self.playbook,
            similarity_threshold=similarity_threshold,
            prune_threshold=prune_threshold,
        )

        # Statistics
        self.run_count = 0
        self.improvement_count = 0
        self._last_reflection = None
        self._last_curation_stats = None

    def run(self, task: str, improve: Optional[bool] = None) -> Any:
        """
        Execute a task with optional self-improvement.

        Args:
            task: The task description to execute
            improve: Override auto_improve for this run (optional)

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

    @property
    def agent(self) -> MultiStepAgent:
        """Access the underlying smolagents agent."""
        return self.generator.agent

    def stats(self) -> dict:
        """
        Get comprehensive statistics about the ACE agent.

        Returns:
            Dictionary with run stats, improvement stats, and playbook stats
        """
        return {
            "runs": self.run_count,
            "improvements": self.improvement_count,
            "auto_improve": self.auto_improve,
            "playbook": self.playbook.stats(),
            "last_curation": self._last_curation_stats,
        }

    def show_playbook(self, max_entries_per_section: Optional[int] = None) -> str:
        """
        Get a formatted string representation of the current Playbook.

        Args:
            max_entries_per_section: Limit entries per section (optional)

        Returns:
            Formatted playbook string
        """
        return self.playbook.render(max_entries_per_section)

    def save_playbook(self, path: str) -> None:
        """
        Save the current Playbook to a JSON file.

        Args:
            path: File path to save to
        """
        self.playbook.save(path)
        logger.info(f"Playbook saved to {path}")

    def load_playbook(self, path: str) -> None:
        """
        Load a Playbook from a JSON file.

        Args:
            path: File path to load from
        """
        self.playbook = Playbook.load(path)
        self.curator.playbook = self.playbook
        self.generator.update_playbook(self.playbook)
        logger.info(f"Playbook loaded from {path} (version {self.playbook.version})")

    @classmethod
    def from_playbook(
        cls,
        path: str,
        model: Any,
        tools: Optional[List] = None,
        **kwargs,
    ) -> "ACEAgent":
        """
        Create an ACEAgent with a pre-existing Playbook.

        Args:
            path: Path to the playbook JSON file
            model: The LLM model to use
            tools: List of tools available to the agent
            **kwargs: Additional arguments passed to ACEAgent

        Returns:
            New ACEAgent instance with loaded playbook
        """
        playbook = Playbook.load(path)
        return cls(model=model, tools=tools, playbook=playbook, **kwargs)

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
            # Check for duplicates using curator's dedup
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
        return (
            f"ACEAgent("
            f"playbook={self.playbook}, "
            f"auto_improve={self.auto_improve}, "
            f"runs={self.run_count})"
        )
