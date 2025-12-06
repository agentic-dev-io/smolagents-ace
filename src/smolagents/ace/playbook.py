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
Playbook data structures for ACE (Agentic Context Engineering).

The Playbook is the evolving context structure that stores strategies,
formulas, and common mistakes with helpful/harmful counters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import json


class PlaybookSection(Enum):
    """Sections in the Playbook for organizing different types of knowledge."""

    STRATEGIES = "STRATEGIES & INSIGHTS"
    FORMULAS = "FORMULAS & CALCULATIONS"
    MISTAKES = "COMMON MISTAKES TO AVOID"


@dataclass
class PlaybookEntry:
    """
    A single entry in the Playbook with helpful/harmful tracking.

    Attributes:
        id: Unique identifier (e.g., "str-00001", "cal-00002", "mis-00003")
        section: Which section this entry belongs to
        content: The actual strategy, formula, or mistake description
        helpful_count: Number of times this entry was helpful
        harmful_count: Number of times this entry was harmful
        metadata: Optional additional metadata
    """

    id: str
    section: PlaybookSection
    content: str
    helpful_count: int = 0
    harmful_count: int = 0
    metadata: Optional[Dict] = None

    def score(self) -> float:
        """
        Calculate net score for prioritization.

        Higher scores indicate more helpful entries.
        """
        return self.helpful_count - self.harmful_count

    def to_string(self) -> str:
        """Format entry for prompt injection."""
        return f"[{self.id}] helpful={self.helpful_count} harmful={self.harmful_count} :: {self.content}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "section": self.section.name,
            "content": self.content,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlaybookEntry":
        """Create entry from dictionary."""
        return cls(
            id=data["id"],
            section=PlaybookSection[data["section"]],
            content=data["content"],
            helpful_count=data.get("helpful_count", 0),
            harmful_count=data.get("harmful_count", 0),
            metadata=data.get("metadata"),
        )


@dataclass
class Playbook:
    """
    Evolving context structure for ACE agents.

    The Playbook stores accumulated knowledge organized into sections,
    with helpful/harmful counters for each entry to track effectiveness.

    Attributes:
        entries: Dictionary mapping entry IDs to PlaybookEntry objects
        version: Version number incremented on each update
        name: Optional name for the playbook
        description: Optional description of the playbook's purpose
    """

    entries: Dict[str, PlaybookEntry] = field(default_factory=dict)
    version: int = 0
    name: Optional[str] = None
    description: Optional[str] = None

    # Counter tracking for ID generation
    _counters: Dict[str, int] = field(default_factory=lambda: {"str": 0, "cal": 0, "mis": 0})

    def add_entry(self, section: PlaybookSection, content: str, metadata: Optional[Dict] = None) -> PlaybookEntry:
        """
        Add a new entry to the playbook.

        Args:
            section: Which section to add the entry to
            content: The content of the entry
            metadata: Optional additional metadata

        Returns:
            The created PlaybookEntry
        """
        prefix_map = {
            PlaybookSection.STRATEGIES: "str",
            PlaybookSection.FORMULAS: "cal",
            PlaybookSection.MISTAKES: "mis",
        }
        prefix = prefix_map[section]
        self._counters[prefix] += 1
        entry_id = f"{prefix}-{self._counters[prefix]:05d}"

        entry = PlaybookEntry(
            id=entry_id,
            section=section,
            content=content,
            metadata=metadata,
        )
        self.entries[entry_id] = entry
        return entry

    def update_counter(self, entry_id: str, helpful: bool) -> bool:
        """
        Update helpful/harmful counter for an entry.

        Args:
            entry_id: The ID of the entry to update
            helpful: True to increment helpful, False to increment harmful

        Returns:
            True if entry was found and updated, False otherwise
        """
        if entry_id not in self.entries:
            return False

        if helpful:
            self.entries[entry_id].helpful_count += 1
        else:
            self.entries[entry_id].harmful_count += 1
        return True

    def get_entry(self, entry_id: str) -> Optional[PlaybookEntry]:
        """Get an entry by ID."""
        return self.entries.get(entry_id)

    def get_entries_by_section(self, section: PlaybookSection) -> List[PlaybookEntry]:
        """Get all entries in a section, sorted by score (descending)."""
        entries = [e for e in self.entries.values() if e.section == section]
        return sorted(entries, key=lambda e: -e.score())

    def render(self, max_entries_per_section: Optional[int] = None) -> str:
        """
        Render the playbook as a string for prompt injection.

        Args:
            max_entries_per_section: Optional limit on entries per section

        Returns:
            Formatted string representation of the playbook
        """
        output = []

        for section in PlaybookSection:
            entries = self.get_entries_by_section(section)

            if not entries:
                continue

            if max_entries_per_section:
                entries = entries[:max_entries_per_section]

            output.append(f"## {section.value}")
            for entry in entries:
                output.append(entry.to_string())
            output.append("")

        return "\n".join(output)

    def stats(self) -> dict:
        """Get statistics about the playbook."""
        total = len(self.entries)
        by_section = {}
        total_helpful = 0
        total_harmful = 0

        for section in PlaybookSection:
            entries = self.get_entries_by_section(section)
            by_section[section.value] = len(entries)
            for e in entries:
                total_helpful += e.helpful_count
                total_harmful += e.harmful_count

        return {
            "total_entries": total,
            "by_section": by_section,
            "total_helpful": total_helpful,
            "total_harmful": total_harmful,
            "version": self.version,
        }

    def save(self, path: str) -> None:
        """
        Save playbook to JSON file.

        Args:
            path: File path to save to
        """
        data = {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "counters": self._counters,
            "entries": [e.to_dict() for e in self.entries.values()],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Playbook":
        """
        Load playbook from JSON file.

        Args:
            path: File path to load from

        Returns:
            Loaded Playbook instance
        """
        with open(path) as f:
            data = json.load(f)

        playbook = cls(
            version=data.get("version", 0),
            name=data.get("name"),
            description=data.get("description"),
        )
        playbook._counters = data.get("counters", {"str": 0, "cal": 0, "mis": 0})

        for entry_data in data.get("entries", []):
            entry = PlaybookEntry.from_dict(entry_data)
            playbook.entries[entry.id] = entry

        return playbook

    def __len__(self) -> int:
        """Return number of entries."""
        return len(self.entries)

    def __repr__(self) -> str:
        return f"Playbook(entries={len(self.entries)}, version={self.version})"
