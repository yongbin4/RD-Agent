"""Global Knowledge Base - main interface for skills and SOTA models."""

from typing import List, Optional, TYPE_CHECKING
from pathlib import Path
import logging

from rdagent.components.skill_learning.skill import Skill
from rdagent.components.skill_learning.storage import GlobalKnowledgeStorage
from rdagent.components.skill_learning.matcher import SkillMatcher

if TYPE_CHECKING:
    from rdagent.components.experiment_learning.sota import SOTAModel

logger = logging.getLogger(__name__)


class GlobalKnowledgeBase:
    """Central manager for global knowledge (skills and SOTA models)."""

    def __init__(self, storage_path: Optional[Path] = None, embedding_model=None):
        """
        Initialize global knowledge base.

        Args:
            storage_path: Optional custom storage path (defaults to ~/.rdagent/global_knowledge/)
            embedding_model: Optional custom embedding model
        """
        self.storage = GlobalKnowledgeStorage(storage_path)
        self.matcher = SkillMatcher(embedding_model)
        self.skill_cache = {}  # skill_id -> Skill
        self.sota_cache = {}  # competition -> List[SOTAModel]
        self._skills_loaded = False

    def load_all_skills(self) -> List[Skill]:
        """Load all skills from disk (with caching)."""
        if self._skills_loaded and self.skill_cache:
            return list(self.skill_cache.values())

        skill_ids = self.storage.list_skills()
        logger.info(f"Loading {len(skill_ids)} skills from global library")

        for skill_id in skill_ids:
            if skill_id not in self.skill_cache:
                skill = self.storage.load_skill(skill_id)
                if skill:
                    self.skill_cache[skill.id] = skill

        self._skills_loaded = True
        logger.info(f"✅ Loaded {len(self.skill_cache)} skills successfully")
        return list(self.skill_cache.values())

    def query_skills(
        self,
        task,
        top_k: int = 5,
        task_contexts: List[str] = None
    ) -> List[Skill]:
        """
        Find relevant skills for a task.

        Args:
            task: Task object (or string description)
            top_k: Number of skills to return
            task_contexts: Optional list of context tags

        Returns:
            List of relevant Skill objects
        """
        # Load skills if not already loaded
        all_skills = self.load_all_skills()

        if not all_skills:
            return []

        # Extract task description
        if isinstance(task, str):
            task_description = task
        elif hasattr(task, 'description'):
            task_description = task.description
        elif hasattr(task, 'name'):
            task_description = task.name
        else:
            task_description = str(task)

        # Find relevant skills
        relevant = self.matcher.find_relevant_skills(
            task_description=task_description,
            skill_library=all_skills,
            top_k=top_k,
            task_contexts=task_contexts
        )

        skills = [skill for skill, score in relevant]

        if skills:
            logger.info(f"📚 Found {len(skills)} relevant skills for task")
            for skill in skills:
                logger.debug(f"  - {skill.name} (success rate: {skill.success_rate():.1%})")

        return skills

    def add_or_update_skill(self, skill: Skill):
        """
        Add new skill or update existing similar skill.

        If a similar skill exists (same name or very similar pattern),
        merge them by adding the new example and updating stats.
        """
        # Check if skill with same name exists
        existing_skill = self.skill_cache.get(skill.id)

        if not existing_skill:
            # Check for skills with same name (different IDs)
            for cached_skill in self.skill_cache.values():
                if cached_skill.name == skill.name:
                    existing_skill = cached_skill
                    logger.info(f"Found existing skill with same name: {skill.name}")
                    break

        if existing_skill:
            # Merge with existing skill
            logger.info(f"Updating existing skill: {existing_skill.name}")

            # Add new examples
            existing_skill.examples.extend(skill.examples)

            # Update stats
            existing_skill.success_count += skill.success_count
            existing_skill.attempt_count += skill.attempt_count

            # Merge contexts (deduplicate)
            for context in skill.applicable_contexts:
                if context not in existing_skill.applicable_contexts:
                    existing_skill.applicable_contexts.append(context)

            # Merge source competitions
            for comp in skill.source_competitions:
                if comp not in existing_skill.source_competitions:
                    existing_skill.source_competitions.append(comp)

            # Increment version
            existing_skill.version += 1

            # Save updated skill
            self.storage.save_skill(existing_skill)
            self.skill_cache[existing_skill.id] = existing_skill

        else:
            # Add new skill
            logger.info(f"Adding new skill: {skill.name}")
            self.storage.save_skill(skill)
            self.skill_cache[skill.id] = skill

    def get_sota(self, competition: str, top_k: int = 3) -> List["SOTAModel"]:
        """
        Get top-K SOTA models for a competition.

        Args:
            competition: Competition name
            top_k: Number of SOTA models to return

        Returns:
            List of SOTAModel objects, sorted by rank
        """
        # Check cache
        if competition in self.sota_cache:
            return self.sota_cache[competition][:top_k]

        # Load from storage
        sota_models = self.storage.load_sota(competition, top_k)
        self.sota_cache[competition] = sota_models

        return sota_models

    def save_sota(self, competition: str, sota: "SOTAModel"):
        """
        Save SOTA model for a competition.

        Args:
            competition: Competition name
            sota: SOTAModel object
        """
        self.storage.save_sota(competition, sota)

        # Update cache
        self.sota_cache[competition] = self.storage.load_sota(competition, top_k=3)

    def get_best_score(self, competition: str) -> Optional[float]:
        """
        Get the best score for a competition.

        Args:
            competition: Competition name

        Returns:
            Best score, or None if no SOTA exists
        """
        return self.storage.get_best_score(competition)

    def update_skill_usage(self, skill_id: str, success: bool):
        """
        Update skill statistics after usage.

        Args:
            skill_id: Skill ID
            success: Whether the skill usage was successful
        """
        skill = self.skill_cache.get(skill_id)
        if skill:
            skill.update_stats(success)
            self.storage.save_skill(skill)

    def get_statistics(self) -> dict:
        """Get statistics about the global knowledge base."""
        all_skills = self.load_all_skills()

        # Count competitions
        with open(self.storage.index_path, 'r') as f:
            import json
            index = json.load(f)
            competition_count = len(index.get('competitions', {}))

        return {
            "total_skills": len(all_skills),
            "total_competitions": competition_count,
            "average_skill_success_rate": (
                sum(s.success_rate() for s in all_skills) / len(all_skills)
                if all_skills else 0.0
            ),
            "total_examples": sum(len(s.examples) for s in all_skills),
        }
