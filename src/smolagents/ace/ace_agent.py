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
ACEAgent: Self-improving agent with Generator -> Reflector -> Curator loop.

Main entry point for using ACE (Agentic Context Engineering) with smolagents.
"""

from typing import Any, List, Optional
import logging

from .playbook import Playbook
from .generator import ACEGenerator
from .reflector import ACEReflector
from .curator import ACECurator

logger = logging.getLogger(__name__)


class ACEAgent:
    """
    Self-Improving Agent with Generator -> Reflector -> Curator Loop.

    ACEAgent wraps the three-role ACE architecture into a single unified interface.
    Each task execution can optionally trigger reflection and curation to improve
    the Playbook for future tasks.

    Architecture:
    ```
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  GENERATOR  │ ──► │  REFLECTOR  │ ──► │   CURATOR   │
    │             │     │             │     │             │
    │ Reasoning   │     │ Evaluate &  │     │ Delta       │
    │ Trajectories│     │ Extract     │     │ Updates     │
    │             │     │ Insights    │     │             │
    └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    ┌─────────────────────────────────────────────────────┐
    │                    PLAYBOOK                          │
    └─────────────────────────────────────────────────────┘
    ```

    Attributes:
        playbook: The evolving Playbook with strategies and insights
        generator: The Generator role (task execution)
        reflector: The Reflector role (insight extraction)
        curator: The Curator role (playbook updates)
        auto_improve: Whether to automatically improve after each run
        run_count: Total number of tasks executed
        improvement_count: Number of improvement cycles completed
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[List] = None,
        playbook: Optional[Playbook] = None,
        auto_improve: bool = True,
        similarity_threshold: float = 0.85,
        prune_threshold: int = -3,
        max_steps: int = 10,
        verbosity_level: int = 1,
        **agent_kwargs,
    ):
        """
        Initialize the ACE Agent.

        Args:
            model: The LLM model to use for all roles
            tools: List of tools available to the agent
            playbook: Initial Playbook (creates empty one if not provided)
            auto_improve: Whether to run reflection/curation after each task
            similarity_threshold: Similarity threshold for deduplication
            prune_threshold: Score threshold for pruning harmful entries
            max_steps: Maximum steps for agent execution
            verbosity_level: Verbosity level for agent output
            **agent_kwargs: Additional arguments passed to the underlying CodeAgent
        """
        self.model = model
        self.tools = tools or []
        self.playbook = playbook or Playbook()
        self.auto_improve = auto_improve

        # Initialize the three roles
        self.generator = ACEGenerator(
            model=model,
            tools=tools,
            playbook=self.playbook,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            **agent_kwargs,
        )
        self.reflector = ACEReflector(model=model)
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

        # 1. Generate: Execute the task
        logger.info(f"ACE Generator executing task: {task[:100]}...")
        gen_result = self.generator.run(task)
        self.run_count += 1

        if should_improve:
            # 2. Reflect: Analyze the execution
            logger.info("ACE Reflector analyzing execution...")
            reflection = self.reflector.reflect(gen_result)
            self._last_reflection = reflection

            if reflection.get("parse_error"):
                logger.warning(f"Reflection parse error: {reflection['parse_error']}")
            else:
                # 3. Curate: Update the playbook
                logger.info("ACE Curator updating playbook...")
                _, curation_stats = self.curator.curate(reflection)
                self._last_curation_stats = curation_stats
                self.improvement_count += 1

                # 4. Rebuild Generator with updated Playbook
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
