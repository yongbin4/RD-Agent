"""Skill matching using embeddings and context filtering."""

from typing import List, Tuple
import numpy as np
import logging

from rdagent.components.skill_learning.skill import Skill

logger = logging.getLogger(__name__)


class SkillMatcher:
    """Match skills to tasks using embedding similarity and context filtering."""

    def __init__(self, embedding_model=None):
        """Initialize with embedding model."""
        self.embedding_model = embedding_model
        self._embedding_cache = {}

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded embedding model: all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not installed, using simple string matching")
                self.embedding_model = "fallback"
        return self.embedding_model

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        model = self._get_embedding_model()
        if model == "fallback":
            # Fallback to simple word overlap if no embedding model
            return np.array([hash(word) % 1000 for word in text.lower().split()[:10]])

        embedding = model.encode(text)
        self._embedding_cache[text] = embedding
        return embedding

    def find_relevant_skills(
        self,
        task_description: str,
        skill_library: List[Skill],
        top_k: int = 5,
        min_success_rate: float = 0.3,
        task_contexts: List[str] = None
    ) -> List[Tuple[Skill, float]]:
        """
        Find top-K relevant skills for task.

        Args:
            task_description: Description of the task
            skill_library: List of available skills
            top_k: Number of skills to return
            min_success_rate: Minimum success rate threshold
            task_contexts: Optional list of context tags for the task

        Returns:
            List of (skill, relevance_score) tuples
        """
        if not skill_library:
            return []

        # Get task embedding
        task_embedding = self._get_embedding(task_description)

        candidates = []
        for skill in skill_library:
            # Filter by success rate
            if skill.success_rate() < min_success_rate:
                continue

            # Context matching (if provided)
            if task_contexts:
                context_match_score = self._context_match_score(task_contexts, skill.applicable_contexts)
                if context_match_score == 0:
                    continue  # No context overlap
            else:
                context_match_score = 0.5  # Neutral if no context provided

            # Compute embedding similarity
            skill_text = f"{skill.name} {skill.description}"
            skill_embedding = self._get_embedding(skill_text)
            similarity = self._cosine_similarity(task_embedding, skill_embedding)

            # Combined score: 40% similarity + 30% success_rate + 30% context_match
            score = (
                0.4 * similarity +
                0.3 * skill.success_rate() +
                0.3 * context_match_score
            )

            candidates.append((skill, score))

        # Sort by score and return top-K
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def _context_match_score(self, task_contexts: List[str], skill_contexts: List[str]) -> float:
        """Calculate context match score (Jaccard similarity)."""
        if not task_contexts or not skill_contexts:
            return 0.0

        task_set = set(c.lower() for c in task_contexts)
        skill_set = set(c.lower() for c in skill_contexts)

        intersection = len(task_set & skill_set)
        union = len(task_set | skill_set)

        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        if len(a) != len(b):
            # Handle different lengths (fallback mode)
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))
