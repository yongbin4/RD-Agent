"""Global Knowledge Base - main interface for skills and SOTA models."""

from typing import List, Optional, TYPE_CHECKING
from pathlib import Path
import logging

from rdagent.components.skill_learning.skill import Skill
from rdagent.components.skill_learning.debug_skill import DebugSkill
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
        self.debug_skill_cache = {}  # debug_skill_id -> DebugSkill
        self.sota_cache = {}  # competition -> List[SOTAModel]
        self._skills_loaded = False
        self._debug_skills_loaded = False

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

    def _compare_skills(self, existing: Skill, new: Skill) -> str:
        """Use LLM to decide whether to REPLACE, SKIP, or MERGE skills."""
        from rdagent.oai.llm_utils import APIBackend

        prompt = f"""Compare these two ML skills and decide how to manage them:

EXISTING SKILL:
Name: {existing.name}
Success Rate: {existing.success_rate():.1%} ({existing.success_count}/{existing.attempt_count} attempts)
Competitions: {', '.join(existing.source_competitions[:3])}
Contexts: {', '.join(existing.applicable_contexts[:3])}
Examples: {len(existing.examples)}
Description: {existing.description[:200]}...

NEW SKILL:
Name: {new.name}
Success Rate: {new.success_rate():.1%} ({new.success_count}/{new.attempt_count} attempts)
Competitions: {', '.join(new.source_competitions[:3])}
Contexts: {', '.join(new.applicable_contexts[:3])}
Examples: {len(new.examples)}
Description: {new.description[:200]}...

Decision options:
- REPLACE: New skill is clearly better (higher quality, more applicable, better performance)
- SKIP: New skill is worse or redundant (keep existing, discard new)
- MERGE: Skills are complementary (combine examples and contexts)

Important: Consider that skills from different competitions may use different metrics (Log Loss vs Accuracy vs F1). Don't just compare success rates numerically.

Respond with ONLY ONE WORD: REPLACE, SKIP, or MERGE"""

        try:
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=prompt,
                system_prompt="You are an ML expert. Compare skills intelligently considering different contexts and metrics.",
                json_mode=False
            )
            decision = response.strip().upper()
            # Validate response
            if decision not in ["REPLACE", "SKIP", "MERGE"]:
                logger.warning(f"Invalid LLM decision: {decision}, defaulting to MERGE")
                return "MERGE"
            return decision
        except Exception as e:
            logger.warning(f"LLM skill comparison failed: {e}, defaulting to MERGE")
            return "MERGE"

    def add_or_update_skill(self, skill: Skill):
        """Add skill with LLM-based replace/merge/skip logic."""

        # Find similar skill (by name)
        similar_skill = None
        for cached_skill in self.skill_cache.values():
            if cached_skill.name == skill.name:
                similar_skill = cached_skill
                break

        if similar_skill:
            # Use LLM to decide: REPLACE, SKIP, or MERGE
            decision = self._compare_skills(similar_skill, skill)

            if decision == "REPLACE":
                # Delete old, add new
                logger.info(f"🔄 Replacing skill '{similar_skill.name}' with better version")
                self.storage.delete_skill(similar_skill.id)
                del self.skill_cache[similar_skill.id]
                self.storage.save_skill(skill)
                self.skill_cache[skill.id] = skill

            elif decision == "SKIP":
                # Don't add new skill
                logger.info(f"⏭️  Skipping skill '{skill.name}' (keeping existing)")

            else:  # MERGE
                # Combine examples and contexts
                logger.info(f"🔀 Merging skill '{skill.name}' with existing")

                similar_skill.examples.extend(skill.examples)
                similar_skill.success_count += skill.success_count
                similar_skill.attempt_count += skill.attempt_count

                for context in skill.applicable_contexts:
                    if context not in similar_skill.applicable_contexts:
                        similar_skill.applicable_contexts.append(context)

                for comp in skill.source_competitions:
                    if comp not in similar_skill.source_competitions:
                        similar_skill.source_competitions.append(comp)

                similar_skill.version += 1
                self.storage.save_skill(similar_skill)
                self.skill_cache[similar_skill.id] = similar_skill
        else:
            # Add new skill
            logger.info(f"➕ Adding new skill: {skill.name}")
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

    def load_all_debug_skills(self) -> List[DebugSkill]:
        """Load all debug skills from disk (with caching)."""
        if self._debug_skills_loaded and self.debug_skill_cache:
            return list(self.debug_skill_cache.values())

        skill_ids = self.storage.list_debug_skills()
        logger.info(f"Loading {len(skill_ids)} debug skills from global library")

        for skill_id in skill_ids:
            if skill_id not in self.debug_skill_cache:
                debug_skill = self.storage.load_debug_skill(skill_id)
                if debug_skill:
                    self.debug_skill_cache[debug_skill.id] = debug_skill

        self._debug_skills_loaded = True
        logger.info(f"✅ Loaded {len(self.debug_skill_cache)} debug skills successfully")
        return list(self.debug_skill_cache.values())

    def query_debug_skills(
        self,
        context,
        error_symptoms: str = "",
        top_k: int = 3,
        task_contexts: List[str] = None
    ) -> List[DebugSkill]:
        """
        Find relevant debug skills based on context and error symptoms.

        Args:
            context: Current task/context (or string description)
            error_symptoms: Optional error message or symptom description
            top_k: Number of debug skills to return
            task_contexts: Optional list of context tags

        Returns:
            List of relevant DebugSkill objects
        """
        # Load debug skills if not already loaded
        all_debug_skills = self.load_all_debug_skills()

        if not all_debug_skills:
            return []

        # Build search query from context and symptoms
        if isinstance(context, str):
            query = context
        elif hasattr(context, 'description'):
            query = context.description
        elif hasattr(context, 'name'):
            query = context.name
        else:
            query = str(context)

        # Append error symptoms if provided
        if error_symptoms:
            query = f"{query} {error_symptoms}"

        # Find relevant debug skills (reuse matcher logic)
        relevant = self.matcher.find_relevant_skills(
            task_description=query,
            skill_library=all_debug_skills,
            top_k=top_k,
            task_contexts=task_contexts
        )

        debug_skills = [skill for skill, score in relevant]

        if debug_skills:
            logger.info(f"🔧 Found {len(debug_skills)} relevant debug skills")
            for skill in debug_skills:
                logger.debug(f"  - {skill.name} (fix rate: {skill.fix_success_rate():.1%}, severity: {skill.severity})")

        return debug_skills

    def _compare_debug_skills(self, existing: DebugSkill, new: DebugSkill) -> str:
        """Use LLM to decide whether to REPLACE, SKIP, or MERGE debug skills."""
        from rdagent.oai.llm_utils import APIBackend

        prompt = f"""Compare these two debugging skills:

EXISTING DEBUG SKILL:
Name: {existing.name}
Fix Success Rate: {existing.fix_success_rate():.1%} ({existing.fix_success_count}/{existing.detection_count} detections)
Severity: {existing.severity}
Symptom: {existing.symptom[:150]}...
Solution: {existing.solution[:150]}...

NEW DEBUG SKILL:
Name: {new.name}
Fix Success Rate: {new.fix_success_rate():.1%} ({new.fix_success_count}/{new.detection_count} detections)
Severity: {new.severity}
Symptom: {new.symptom[:150]}...
Solution: {new.solution[:150]}...

Should we REPLACE, SKIP, or MERGE?

Respond with ONLY ONE WORD: REPLACE, SKIP, or MERGE"""

        try:
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=prompt,
                system_prompt="You are an ML expert. Compare debugging patterns intelligently.",
                json_mode=False
            )
            decision = response.strip().upper()
            if decision not in ["REPLACE", "SKIP", "MERGE"]:
                logger.warning(f"Invalid LLM decision: {decision}, defaulting to MERGE")
                return "MERGE"
            return decision
        except Exception as e:
            logger.warning(f"LLM debug skill comparison failed: {e}, defaulting to MERGE")
            return "MERGE"

    def _is_similar_debug_skill_by_embedding(
        self,
        skill1: DebugSkill,
        skill2: DebugSkill,
        threshold: float = 0.85
    ) -> bool:
        """
        Fast similarity check using embeddings before LLM comparison.

        Compares symptom + solution text using embedding similarity.
        This is much faster than LLM comparison and can be used for pre-filtering.

        Args:
            skill1: First debug skill
            skill2: Second debug skill
            threshold: Similarity threshold (0-1), default 0.85

        Returns:
            True if skills are similar (above threshold), False otherwise
        """
        try:
            # Compare symptom + solution embeddings
            text1 = f"{skill1.symptom} {skill1.solution}"
            text2 = f"{skill2.symptom} {skill2.solution}"

            similarity = self.matcher._cosine_similarity(
                self.matcher._get_embedding(text1),
                self.matcher._get_embedding(text2)
            )
            return similarity > threshold
        except Exception as e:
            logger.debug(f"Embedding similarity check failed: {e}")
            # Fall back to name comparison
            return skill1.name == skill2.name

    def add_or_update_debug_skill(self, debug_skill: DebugSkill):
        """Add debug skill with embedding pre-filter and LLM-based replace/merge/skip logic."""

        # Find similar debug skill - first by name, then by embedding similarity
        similar_skill = None

        # First pass: exact name match
        for cached_skill in self.debug_skill_cache.values():
            if cached_skill.name == debug_skill.name:
                similar_skill = cached_skill
                break

        # Second pass: if no name match, check embedding similarity
        if similar_skill is None:
            for cached_skill in self.debug_skill_cache.values():
                if self._is_similar_debug_skill_by_embedding(cached_skill, debug_skill, threshold=0.85):
                    logger.info(f"📊 Found similar debug skill by embedding: '{cached_skill.name}' ≈ '{debug_skill.name}'")
                    similar_skill = cached_skill
                    break

        if similar_skill:
            # Use LLM to decide: REPLACE, SKIP, or MERGE
            decision = self._compare_debug_skills(similar_skill, debug_skill)

            if decision == "REPLACE":
                logger.info(f"🔄 Replacing debug skill '{similar_skill.name}' with better version")
                self.storage.delete_debug_skill(similar_skill.id)
                del self.debug_skill_cache[similar_skill.id]
                self.storage.save_debug_skill(debug_skill)
                self.debug_skill_cache[debug_skill.id] = debug_skill

            elif decision == "SKIP":
                logger.info(f"⏭️  Skipping debug skill '{debug_skill.name}' (keeping existing)")

            else:  # MERGE
                logger.info(f"🔀 Merging debug skill '{debug_skill.name}' with existing")

                similar_skill.examples.extend(debug_skill.examples)
                similar_skill.detection_count += debug_skill.detection_count
                similar_skill.fix_success_count += debug_skill.fix_success_count

                for context in debug_skill.applicable_contexts:
                    if context not in similar_skill.applicable_contexts:
                        similar_skill.applicable_contexts.append(context)

                for comp in debug_skill.source_competitions:
                    if comp not in similar_skill.source_competitions:
                        similar_skill.source_competitions.append(comp)

                similar_skill.version += 1
                self.storage.save_debug_skill(similar_skill)
                self.debug_skill_cache[similar_skill.id] = similar_skill
        else:
            logger.info(f"➕ Adding new debug skill: {debug_skill.name}")
            self.storage.save_debug_skill(debug_skill)
            self.debug_skill_cache[debug_skill.id] = debug_skill

    def get_statistics(self) -> dict:
        """Get statistics about the global knowledge base."""
        all_skills = self.load_all_skills()
        all_debug_skills = self.load_all_debug_skills()

        # Count competitions
        with open(self.storage.index_path, 'r') as f:
            import json
            index = json.load(f)
            competition_count = len(index.get('competitions', {}))

        return {
            "total_skills": len(all_skills),
            "total_debug_skills": len(all_debug_skills),
            "total_competitions": competition_count,
            "average_skill_success_rate": (
                sum(s.success_rate() for s in all_skills) / len(all_skills)
                if all_skills else 0.0
            ),
            "average_debug_skill_fix_rate": (
                sum(s.fix_success_rate() for s in all_debug_skills) / len(all_debug_skills)
                if all_debug_skills else 0.0
            ),
            "total_examples": sum(len(s.examples) for s in all_skills),
            "total_debug_examples": sum(len(s.examples) for s in all_debug_skills),
        }
