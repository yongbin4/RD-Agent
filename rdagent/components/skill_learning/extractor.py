"""Skill extraction from successful experiments using LLM."""

from typing import List
import json
import logging

from rdagent.components.skill_learning.skill import Skill, SkillExample
from rdagent.oai.llm_utils import APIBackend

logger = logging.getLogger(__name__)


SKILL_EXTRACTION_PROMPT = """Analyze this successful experiment and extract 1-3 reusable patterns (skills).

## Experiment Details
Hypothesis: {hypothesis}
Score: {score}
Competition: {competition}

## Generated Code
{code}

## Task
Extract reusable skills (patterns) that could apply to other similar problems.

For each skill, provide:
1. **name**: Short snake_case identifier (e.g., "missing_value_imputation", "feature_interaction_engineering")
2. **description**: Clear explanation of what it does and when to use it (2-3 sentences)
3. **applicable_contexts**: List of tags describing when this applies (e.g., ["tabular", "missing_values", "classification"])
4. **code_pattern**: The key code snippet (5-20 lines of the most important/reusable part)

Focus on:
- Novel approaches not commonly used
- Effective combinations of techniques
- Domain-specific insights that worked well
- Patterns that could generalize to other problems

Return as JSON array (valid JSON only, no markdown):
[
  {{
    "name": "...",
    "description": "...",
    "applicable_contexts": [...],
    "code_pattern": "..."
  }}
]

Only extract meaningful, generalizable patterns. Skip basic operations everyone knows.
"""


class SkillExtractor:
    """Extract reusable skills from successful experiments using LLM."""

    def __init__(self, llm: APIBackend = None):
        """Initialize with LLM backend."""
        self.llm = llm or APIBackend()

    def extract_from_experiment(
        self,
        experiment,
        feedback,
        competition_context: str = ""
    ) -> List[Skill]:
        """
        Extract skills from a successful experiment.

        Args:
            experiment: The experiment object
            feedback: The feedback object with score and decision
            competition_context: Name/description of the competition

        Returns:
            List of extracted Skill objects
        """
        # Extract hypothesis
        hypothesis = ""
        if hasattr(experiment, "hypothesis"):
            if hasattr(experiment.hypothesis, "hypothesis"):
                hypothesis = str(experiment.hypothesis.hypothesis)
            else:
                hypothesis = str(experiment.hypothesis)

        # Extract score
        score = 0.0
        if hasattr(feedback, "score"):
            score = float(feedback.score)

        # Extract code
        code = self._format_experiment_code(experiment)

        if not code:
            logger.warning("No code found in experiment, skipping skill extraction")
            return []

        # Build prompt
        prompt = SKILL_EXTRACTION_PROMPT.format(
            hypothesis=hypothesis[:500],  # Limit length
            score=score,
            competition=competition_context,
            code=code[:3000],  # Limit code length to avoid token limits
        )

        try:
            # Call LLM
            response = self.llm.build_messages_and_create_chat_completion(
                user_prompt=prompt,
                system_prompt="You are an expert at identifying reusable patterns in data science code.",
                json_mode=True,
            )

            # Parse response
            skills = self._parse_llm_response(response, experiment, feedback, competition_context, code)

            logger.info(f"Extracted {len(skills)} skills from experiment")
            return skills

        except Exception as e:
            logger.error(f"Error extracting skills: {e}")
            return []

    def _format_experiment_code(self, experiment) -> str:
        """Format experiment code for prompt."""
        code_parts = []

        # Try to get code from experiment workspace
        if hasattr(experiment, "experiment_workspace"):
            workspace = experiment.experiment_workspace
            if hasattr(workspace, "file_dict"):
                for filename, code in workspace.file_dict.items():
                    code_parts.append(f"# File: {filename}\n{code}\n")

        # Also try sub_workspace_list
        if hasattr(experiment, "sub_workspace_list"):
            for i, workspace in enumerate(experiment.sub_workspace_list):
                if hasattr(workspace, "file_dict"):
                    for filename, code in workspace.file_dict.items():
                        code_parts.append(f"# Sub-workspace {i} - File: {filename}\n{code}\n")

        return "\n\n".join(code_parts)

    def _parse_llm_response(
        self,
        response: str,
        experiment,
        feedback,
        competition: str,
        full_code: str
    ) -> List[Skill]:
        """Parse LLM JSON response into Skill objects."""
        skills = []

        try:
            # Try to parse JSON
            skill_data_list = json.loads(response)

            if not isinstance(skill_data_list, list):
                logger.warning("LLM response is not a list, trying to wrap it")
                skill_data_list = [skill_data_list]

            for skill_data in skill_data_list:
                try:
                    # Create skill example
                    example = SkillExample(
                        competition=competition,
                        code=skill_data.get("code_pattern", ""),
                        context=skill_data.get("description", "")[:200],
                        score=float(feedback.score) if hasattr(feedback, "score") else 0.0,
                    )

                    # Create skill
                    skill = Skill(
                        name=skill_data.get("name", "unnamed_skill"),
                        description=skill_data.get("description", ""),
                        code_pattern=skill_data.get("code_pattern", ""),
                        applicable_contexts=skill_data.get("applicable_contexts", []),
                        examples=[example],
                        success_count=1,
                        attempt_count=1,
                        source_competitions=[competition],
                        tags=skill_data.get("applicable_contexts", []),
                    )

                    skills.append(skill)

                except Exception as e:
                    logger.error(f"Error creating skill from data: {e}")
                    continue

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                try:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    skill_data_list = json.loads(json_str)
                    # Retry parsing
                    return self._parse_llm_response(json_str, experiment, feedback, competition, full_code)
                except:
                    pass

        return skills
