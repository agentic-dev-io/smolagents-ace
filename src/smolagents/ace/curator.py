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
Curator role for ACE (Agentic Context Engineering).

The Curator converts insights into delta updates for the Playbook,
with semantic similarity deduplication and pruning of harmful entries.
"""

from typing import List, Optional, Tuple
import logging

from .playbook import Playbook, PlaybookSection, PlaybookEntry

logger = logging.getLogger(__name__)


class ACECurator:
    """
    Curator: Converts insights into delta updates with dedup/pruning.

    The Curator role in ACE is responsible for:
    1. Updating helpful/harmful counters for used strategies
    2. Adding new insights with semantic deduplication
    3. Pruning entries with consistently negative scores

    Attributes:
        playbook: The Playbook to curate
        similarity_threshold: Threshold for semantic similarity (0-1)
        prune_threshold: Score threshold below which entries are removed
        embedder: Sentence transformer model for semantic similarity
    """

    def __init__(
        self,
        playbook: Playbook,
        similarity_threshold: float = 0.85,
        prune_threshold: int = -3,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the ACE Curator.

        Args:
            playbook: The Playbook to curate
            similarity_threshold: Similarity threshold for deduplication (0-1)
            prune_threshold: Score threshold for pruning harmful entries
            embedding_model: Name of the sentence-transformers model to use
        """
        self.playbook = playbook
        self.similarity_threshold = similarity_threshold
        self.prune_threshold = prune_threshold
        self.embedding_model_name = embedding_model

        # Lazy-load embedder to avoid import overhead
        self._embedder = None
        self._embedding_cache = {}

    @property
    def embedder(self):
        """Lazy-load the sentence transformer embedder."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Semantic deduplication will be disabled. "
                    "Install with: pip install sentence-transformers"
                )
                self._embedder = False  # Mark as unavailable
        return self._embedder if self._embedder else None

    def curate(self, reflection: dict) -> Tuple[Playbook, dict]:
        """
        Apply delta updates to the Playbook based on reflection.

        Args:
            reflection: Output from ACEReflector.reflect() containing:
                - strategy_feedback: List of feedback for used strategies
                - new_insights: List of new insights to add

        Returns:
            Tuple of (updated Playbook, curation stats)
        """
        stats = {
            "counters_updated": 0,
            "insights_added": 0,
            "insights_skipped_duplicate": 0,
            "entries_pruned": 0,
        }

        # 1. Update counters for used strategies
        for feedback in reflection.get("strategy_feedback", []):
            entry_id = feedback.get("id")
            helpful = feedback.get("helpful", True)

            if self.playbook.update_counter(entry_id, helpful):
                stats["counters_updated"] += 1
                logger.debug(f"Updated counter for {entry_id}: helpful={helpful}")

        # 2. Add new insights with deduplication
        for insight in reflection.get("new_insights", []):
            section_name = insight.get("section", "").upper()
            content = insight.get("content", "")

            if not content or not section_name:
                continue

            try:
                section = PlaybookSection[section_name]
            except KeyError:
                logger.warning(f"Invalid section: {section_name}")
                continue

            # Check for duplicates
            if self._is_duplicate(content):
                stats["insights_skipped_duplicate"] += 1
                logger.debug(f"Skipped duplicate insight: {content[:50]}...")
                continue

            # Add the new entry
            entry = self.playbook.add_entry(section, content)
            stats["insights_added"] += 1
            logger.debug(f"Added new entry: {entry.id}")

            # Cache the embedding for future dedup
            if self.embedder:
                self._embedding_cache[entry.id] = self._get_embedding(content)

        # 3. Prune harmful entries
        pruned = self._prune_harmful()
        stats["entries_pruned"] = len(pruned)

        # 4. Increment version
        self.playbook.version += 1

        return self.playbook, stats

    def _is_duplicate(self, content: str) -> bool:
        """
        Check if content is semantically similar to existing entries.

        Uses cosine similarity with sentence embeddings when available,
        falls back to exact string matching otherwise.

        Args:
            content: The content to check

        Returns:
            True if content is a duplicate, False otherwise
        """
        if not self.playbook.entries:
            return False

        # Fall back to exact matching if embedder unavailable
        if not self.embedder:
            normalized_content = content.lower().strip()
            for entry in self.playbook.entries.values():
                if entry.content.lower().strip() == normalized_content:
                    return True
            return False

        # Semantic similarity check
        new_embedding = self._get_embedding(content)

        for entry_id, entry in self.playbook.entries.items():
            # Get or compute existing embedding
            if entry_id in self._embedding_cache:
                existing_embedding = self._embedding_cache[entry_id]
            else:
                existing_embedding = self._get_embedding(entry.content)
                self._embedding_cache[entry_id] = existing_embedding

            # Calculate cosine similarity
            similarity = self._cosine_similarity(new_embedding, existing_embedding)

            if similarity > self.similarity_threshold:
                logger.debug(
                    f"Found similar entry (similarity={similarity:.3f}): "
                    f"{entry.id} - {entry.content[:50]}..."
                )
                return True

        return False

    def _get_embedding(self, text: str):
        """Get embedding for text using the sentence transformer."""
        return self.embedder.encode(text, convert_to_numpy=True)

    def _cosine_similarity(self, a, b) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _prune_harmful(self) -> List[str]:
        """
        Remove entries with scores below the prune threshold.

        Returns:
            List of pruned entry IDs
        """
        to_remove = []

        for entry_id, entry in self.playbook.entries.items():
            if entry.score() < self.prune_threshold:
                to_remove.append(entry_id)
                logger.info(
                    f"Pruning entry {entry_id} with score {entry.score()}: "
                    f"{entry.content[:50]}..."
                )

        for entry_id in to_remove:
            del self.playbook.entries[entry_id]
            # Also remove from embedding cache
            self._embedding_cache.pop(entry_id, None)

        return to_remove

    def merge_similar_entries(self, threshold: Optional[float] = None) -> int:
        """
        Merge entries that are too similar to each other.

        Keeps the entry with the higher score and transfers counters.

        Args:
            threshold: Similarity threshold (uses instance default if not provided)

        Returns:
            Number of entries merged
        """
        if not self.embedder:
            logger.warning("Cannot merge entries without embedder")
            return 0

        threshold = threshold or self.similarity_threshold
        merged_count = 0
        entries_to_check = list(self.playbook.entries.values())
        merged_ids = set()

        for i, entry1 in enumerate(entries_to_check):
            if entry1.id in merged_ids:
                continue

            for entry2 in entries_to_check[i + 1:]:
                if entry2.id in merged_ids:
                    continue

                if entry1.section != entry2.section:
                    continue

                # Get embeddings
                emb1 = self._embedding_cache.get(entry1.id) or self._get_embedding(entry1.content)
                emb2 = self._embedding_cache.get(entry2.id) or self._get_embedding(entry2.content)

                similarity = self._cosine_similarity(emb1, emb2)

                if similarity > threshold:
                    # Keep the one with higher score
                    keeper, remove = (entry1, entry2) if entry1.score() >= entry2.score() else (entry2, entry1)

                    # Transfer counters
                    keeper.helpful_count += remove.helpful_count
                    keeper.harmful_count += remove.harmful_count

                    # Remove the duplicate
                    del self.playbook.entries[remove.id]
                    self._embedding_cache.pop(remove.id, None)
                    merged_ids.add(remove.id)
                    merged_count += 1

                    logger.info(f"Merged {remove.id} into {keeper.id}")

        return merged_count

    def clear_embedding_cache(self) -> None:
        """Clear the embedding cache to free memory."""
        self._embedding_cache.clear()

    def __repr__(self) -> str:
        return (
            f"ACECurator(playbook={self.playbook}, "
            f"similarity_threshold={self.similarity_threshold})"
        )
