"""Storage manager for global knowledge base (skills and SOTA models)."""

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
import json
import logging

from rdagent.components.skill_learning.skill import Skill
from rdagent.components.skill_learning.debug_skill import DebugSkill

if TYPE_CHECKING:
    from rdagent.components.experiment_learning.sota import SOTAModel

logger = logging.getLogger(__name__)


class GlobalKnowledgeStorage:
    """Manages reading/writing to ~/.rdagent/global_knowledge/."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize storage paths."""
        self.base_path = base_path or (Path.home() / ".rdagent" / "global_knowledge")
        self.skills_dir = self.base_path / "skills"
        self.debugging_skills_dir = self.base_path / "debugging_skills"
        self.competitions_dir = self.base_path / "competitions"
        self.index_path = self.base_path / "index.json"
        self.changelog_path = self.base_path / "CHANGELOG.md"

        # Create directories
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.debugging_skills_dir.mkdir(parents=True, exist_ok=True)
        self.competitions_dir.mkdir(parents=True, exist_ok=True)

        # Initialize index if doesn't exist
        if not self.index_path.exists():
            self._init_index()

        logger.info(f"Global knowledge storage initialized at: {self.base_path}")

    def _init_index(self):
        """Initialize the index.json file."""
        index = {
            "version": "1.0",
            "created_at": "",
            "skills": {},  # skill_id -> skill metadata
            "debugging_skills": {},  # debug_skill_id -> debug skill metadata
            "competitions": {},  # competition_name -> metadata
        }
        self.index_path.write_text(json.dumps(index, indent=2))

        # Initialize changelog
        self.changelog_path.write_text(
            "# Global Knowledge Base Changelog\n\n"
            "This file tracks all changes to the global knowledge base.\n\n"
        )

    def save_skill(self, skill: Skill):
        """Save skill to markdown + code files."""
        # Create skill directory
        skill_dir = self.skills_dir / f"skill_{skill.id}_{skill.name}"
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Save README.md
        (skill_dir / "README.md").write_text(skill.to_markdown())

        # Save example code files
        for i, example in enumerate(skill.examples):
            example_file = skill_dir / f"example_{example.competition}.py"
            example_file.write_text(
                f"# Example from {example.competition}\n"
                f"# Score: {example.score:.4f}\n"
                f"# Context: {example.context}\n\n"
                f"{example.code}\n"
            )

        # Save metadata.json
        (skill_dir / "metadata.json").write_text(json.dumps(skill.to_dict(), indent=2))

        # Update index
        self._update_index_skill(skill)

        # Log to changelog
        self._log_changelog(f"Added/Updated skill: {skill.name} (ID: {skill.id})")

        logger.info(f"Saved skill: {skill.name} (ID: {skill.id})")

    def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill from storage."""
        try:
            # Find skill directory
            skill_dirs = list(self.skills_dir.glob(f"skill_{skill_id}_*"))
            if not skill_dirs:
                logger.warning(f"Skill not found for deletion: {skill_id}")
                return False

            # Delete directory
            import shutil
            shutil.rmtree(skill_dirs[0])
            logger.info(f"Deleted skill: {skill_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete skill {skill_id}: {e}")
            return False

    def load_skill(self, skill_id: str) -> Optional[Skill]:
        """Load skill from directory by ID."""
        # Find skill directory
        skill_dirs = list(self.skills_dir.glob(f"skill_{skill_id}_*"))
        if not skill_dirs:
            logger.warning(f"Skill not found: {skill_id}")
            return None

        skill_dir = skill_dirs[0]
        try:
            return Skill.from_directory(skill_dir)
        except Exception as e:
            logger.error(f"Error loading skill from {skill_dir}: {e}")
            return None

    def list_skills(self) -> List[str]:
        """List all skill IDs."""
        skill_ids = []
        for skill_dir in self.skills_dir.glob("skill_*"):
            if skill_dir.is_dir():
                # Extract ID from directory name: skill_{id}_{name}
                parts = skill_dir.name.split("_", 2)
                if len(parts) >= 2:
                    skill_ids.append(parts[1])
        return skill_ids

    def save_debug_skill(self, debug_skill: DebugSkill):
        """Save debug skill to markdown + metadata files."""
        # Create debug skill directory
        skill_dir = self.debugging_skills_dir / f"debug_{debug_skill.id}_{debug_skill.name}"
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Save README.md
        (skill_dir / "README.md").write_text(debug_skill.to_markdown())

        # Save example files (both failed and solution code)
        for i, example in enumerate(debug_skill.examples):
            # Save failed approach
            failed_file = skill_dir / f"example_{example.competition}_failed.py"
            failed_file.write_text(
                f"# Failed approach from {example.competition}\n"
                f"# Symptom: {example.symptom}\n"
                f"# Score before fix: {example.before_score if example.before_score else 'N/A'}\n\n"
                f"{example.failed_code}\n"
            )

            # Save solution
            solution_file = skill_dir / f"example_{example.competition}_solution.py"
            after_score_str = f"{example.after_score:.4f}" if example.after_score is not None else "N/A"
            solution_file.write_text(
                f"# Solution from {example.competition}\n"
                f"# Score after fix: {after_score_str}\n"
                f"# Context: {example.context}\n\n"
                f"{example.solution_code}\n"
            )

        # Save metadata.json
        (skill_dir / "metadata.json").write_text(json.dumps(debug_skill.to_dict(), indent=2))

        # Update index
        self._update_index_debug_skill(debug_skill)

        # Log to changelog
        self._log_changelog(f"Added/Updated debug skill: {debug_skill.name} (ID: {debug_skill.id})")

        logger.info(f"Saved debug skill: {debug_skill.name} (ID: {debug_skill.id})")

    def delete_debug_skill(self, skill_id: str) -> bool:
        """Delete a debug skill from storage."""
        try:
            # Find debug skill directory
            skill_dirs = list(self.debugging_skills_dir.glob(f"debug_{skill_id}_*"))
            if not skill_dirs:
                logger.warning(f"Debug skill not found for deletion: {skill_id}")
                return False

            # Delete directory
            import shutil
            shutil.rmtree(skill_dirs[0])
            logger.info(f"Deleted debug skill: {skill_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete debug skill {skill_id}: {e}")
            return False

    def load_debug_skill(self, skill_id: str) -> Optional[DebugSkill]:
        """Load debug skill from directory by ID."""
        # Find debug skill directory
        skill_dirs = list(self.debugging_skills_dir.glob(f"debug_{skill_id}_*"))
        if not skill_dirs:
            logger.warning(f"Debug skill not found: {skill_id}")
            return None

        skill_dir = skill_dirs[0]
        try:
            return DebugSkill.from_directory(skill_dir)
        except Exception as e:
            logger.error(f"Error loading debug skill from {skill_dir}: {e}")
            return None

    def list_debug_skills(self) -> List[str]:
        """List all debug skill IDs."""
        skill_ids = []
        for skill_dir in self.debugging_skills_dir.glob("debug_*"):
            if skill_dir.is_dir():
                # Extract ID from directory name: debug_{id}_{name}
                parts = skill_dir.name.split("_", 2)
                if len(parts) >= 2:
                    skill_ids.append(parts[1])
        return skill_ids

    def save_sota(self, competition: str, sota: "SOTAModel"):
        """Save SOTA model for a competition."""
        # Create competition directory
        comp_dir = self.competitions_dir / competition
        sota_dir = comp_dir / "sota"
        sota_dir.mkdir(parents=True, exist_ok=True)

        # Set competition name
        sota.competition = competition

        # Save SOTA model
        sota.save_to_directory(sota_dir)

        # Update or create OVERVIEW.md
        self._update_competition_overview(competition)

        # Update index
        self._update_index_competition(competition, sota)

        # Log to changelog
        self._log_changelog(
            f"New SOTA for {competition}: rank {sota.rank}, score {sota.score:.6f}"
        )

        logger.info(f"Saved SOTA for {competition}: rank {sota.rank}, score {sota.score:.6f}")

    def load_sota(self, competition: str, top_k: int = 3) -> List["SOTAModel"]:
        """Load top-K SOTA models for competition."""
        from rdagent.components.experiment_learning.sota import SOTAModel

        comp_dir = self.competitions_dir / competition
        sota_dir = comp_dir / "sota"

        if not sota_dir.exists():
            return []

        # Find all SOTA directories
        sota_models = []
        for model_dir in sota_dir.glob("rank_*"):
            if model_dir.is_dir():
                try:
                    sota = SOTAModel.load_from_directory(model_dir)
                    sota_models.append(sota)
                except Exception as e:
                    logger.error(f"Error loading SOTA from {model_dir}: {e}")

        # Sort by rank and return top-K
        sota_models.sort(key=lambda x: x.rank)
        return sota_models[:top_k]

    def get_best_score(self, competition: str) -> Optional[float]:
        """Get the best score for a competition."""
        sota_models = self.load_sota(competition, top_k=1)
        return sota_models[0].score if sota_models else None

    def _update_index_skill(self, skill: Skill):
        """Update index with skill metadata."""
        with open(self.index_path, "r") as f:
            index = json.load(f)

        index["skills"][skill.id] = {
            "name": skill.name,
            "success_rate": skill.success_rate(),
            "contexts": skill.applicable_contexts,
            "examples_count": len(skill.examples),
        }

        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _update_index_debug_skill(self, debug_skill: DebugSkill):
        """Update index with debug skill metadata."""
        with open(self.index_path, "r") as f:
            index = json.load(f)

        # Ensure debugging_skills key exists
        if "debugging_skills" not in index:
            index["debugging_skills"] = {}

        index["debugging_skills"][debug_skill.id] = {
            "name": debug_skill.name,
            "fix_success_rate": debug_skill.fix_success_rate(),
            "contexts": debug_skill.applicable_contexts,
            "examples_count": len(debug_skill.examples),
            "severity": debug_skill.severity,
        }

        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _update_index_competition(self, competition: str, sota: "SOTAModel"):
        """Update index with competition metadata."""
        with open(self.index_path, "r") as f:
            index = json.load(f)

        if competition not in index["competitions"]:
            index["competitions"][competition] = {
                "best_score": sota.score,
                "models_count": 1,
            }
        else:
            current_best = index["competitions"][competition].get("best_score", 0.0)
            index["competitions"][competition]["best_score"] = max(current_best, sota.score)
            index["competitions"][competition]["models_count"] = (
                index["competitions"][competition].get("models_count", 0) + 1
            )

        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _update_competition_overview(self, competition: str):
        """Create or update OVERVIEW.md for a competition."""
        comp_dir = self.competitions_dir / competition
        overview_path = comp_dir / "OVERVIEW.md"

        # Load SOTA models
        sota_models = self.load_sota(competition, top_k=3)

        if not sota_models:
            return

        # Generate overview
        overview = f"""# Competition: {competition}

**Status**: Active
**Best Score**: {sota_models[0].score:.6f}
**Total Models**: {len(sota_models)}
**Last Updated**: {sota_models[0].timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Top Solutions

"""

        medals = ["🥇", "🥈", "🥉"]
        for i, sota in enumerate(sota_models[:3]):
            medal = medals[i] if i < 3 else f"#{i+1}"
            overview += f"""### {medal} Rank {sota.rank}: Score {sota.score:.6f}
- **Date**: {sota.timestamp.strftime('%Y-%m-%d')}
- **Approach**: {sota.approach_description[:200] if sota.approach_description else 'N/A'}
- **Skills Used**: {', '.join(sota.skills_used) if sota.skills_used else 'None'}
- **Code Files**: {', '.join(sota.code_files.keys())}

"""

        overview_path.write_text(overview)

    def _log_changelog(self, message: str):
        """Append entry to changelog."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"- **{timestamp}**: {message}\n"

        with open(self.changelog_path, "a") as f:
            f.write(entry)

    def update_index(self):
        """Rebuild the entire index from disk."""
        index = {
            "version": "1.0",
            "skills": {},
            "debugging_skills": {},
            "competitions": {},
        }

        # Index all skills
        for skill_id in self.list_skills():
            skill = self.load_skill(skill_id)
            if skill:
                index["skills"][skill.id] = {
                    "name": skill.name,
                    "success_rate": skill.success_rate(),
                    "contexts": skill.applicable_contexts,
                    "examples_count": len(skill.examples),
                }

        # Index all debug skills
        for skill_id in self.list_debug_skills():
            debug_skill = self.load_debug_skill(skill_id)
            if debug_skill:
                index["debugging_skills"][debug_skill.id] = {
                    "name": debug_skill.name,
                    "fix_success_rate": debug_skill.fix_success_rate(),
                    "contexts": debug_skill.applicable_contexts,
                    "examples_count": len(debug_skill.examples),
                    "severity": debug_skill.severity,
                }

        # Index all competitions
        for comp_dir in self.competitions_dir.iterdir():
            if comp_dir.is_dir():
                competition = comp_dir.name
                sota_models = self.load_sota(competition)
                if sota_models:
                    index["competitions"][competition] = {
                        "best_score": max(m.score for m in sota_models),
                        "models_count": len(sota_models),
                    }

        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

        logger.info("Index rebuilt successfully")
