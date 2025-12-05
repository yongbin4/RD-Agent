"""Skill extraction from successful experiments using LLM."""

from typing import List, Optional, Dict
import json
import logging

from pydantic import BaseModel, Field

from rdagent.components.skill_learning.skill import Skill, SkillExample
from rdagent.oai.llm_utils import APIBackend

logger = logging.getLogger(__name__)


class SkillData(BaseModel):
    """Schema for a single extracted skill."""
    name: str = Field(description="Short snake_case identifier (e.g., 'missing_value_imputation')")
    description: str = Field(description="Clear explanation of what it does and when to use it (2-3 sentences)")
    applicable_contexts: List[str] = Field(description="Tags describing when this applies (e.g., ['tabular', 'classification'])")
    code_pattern: str = Field(description="The key code snippet (5-20 lines of reusable code)")


class SkillExtractionResponse(BaseModel):
    """Schema for skill extraction response."""
    skills: List[SkillData] = Field(description="List of 1-3 extracted skills")


SKILL_EXTRACTION_PROMPT = """Analyze this successful experiment and extract 1-3 reusable patterns (skills).

## Experiment Details
Hypothesis: {hypothesis}
Score: {score}
Competition: {competition}
Context: {context}

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

Return as JSON object with this exact schema (valid JSON only, no markdown):
{{
  "skills": [
    {{
      "name": "...",
      "description": "...",
      "applicable_contexts": [...],
      "code_pattern": "..."
    }}
  ]
}}

Only extract meaningful, generalizable patterns. Skip basic operations everyone knows.
"""


class SkillExtractor:
    """Extract reusable skills from successful experiments using LLM."""

    def __init__(self, llm: APIBackend = None):
        """Initialize with LLM backend."""
        self.llm = llm or APIBackend()
        self.supports_response_schema = self.llm.supports_response_schema()

    def extract_from_experiment(
        self,
        experiment,
        feedback,
        competition_context: str = "",
        score: Optional[float] = None
    ) -> List[Skill]:
        """
        Extract skills from a successful experiment.

        Args:
            experiment: The experiment object
            feedback: The feedback object with score and decision
            competition_context: Name/description of the competition
            score: Optional score (if not provided, will try to extract from feedback)

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

        # Use passed score or fallback to feedback
        if score is None:
            if hasattr(feedback, "score"):
                score = float(feedback.score)
            else:
                score = 0.0

        # Build context information
        context_parts = []

        # Check if this is first successful submission
        if hasattr(feedback, "observations") and feedback.observations:
            obs = str(feedback.observations)
            if "first valid submission" in obs.lower() or "first successful" in obs.lower():
                context_parts.append("This is the first successful submission after multiple failed attempts.")
            context_parts.append(f"Observations: {obs[:200]}")

        # Check if there's a reason for acceptance
        if hasattr(feedback, "reason") and feedback.reason:
            reason_str = str(feedback.reason)[:200]
            context_parts.append(f"Why accepted: {reason_str}")

        context = " ".join(context_parts) if context_parts else "No additional context available."

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
            context=context[:500],  # Limit context length
            code=code[:15000],  # Increased limit to allow complete ML pipelines
        )

        try:
            # Call LLM
            logger.info(f"🤖 Calling LLM for skill extraction...")
            logger.info(f"  Prompt length: {len(prompt)} chars")
            logger.info(f"  Code length (original): {len(code)} chars, Code length (sent): {min(len(code), 15000)} chars")
            logger.info(f"  Hypothesis: {hypothesis[:100]}...")
            logger.info(f"  Context: {context[:100]}...")
            logger.info(f"  Score: {score}")

            # Log first part of prompt for debugging
            logger.debug(f"Prompt preview:\n{prompt[:1000]}...")

            response = self.llm.build_messages_and_create_chat_completion(
                user_prompt=prompt,
                system_prompt="You are an expert at identifying reusable patterns in data science code.",
                response_format=SkillExtractionResponse if self.supports_response_schema else {"type": "json_object"},
                json_target_type=Dict[str, List[Dict[str, str]]] if not self.supports_response_schema else None,
            )

            logger.info(f"✅ LLM responded")
            logger.info(f"  Response length: {len(response)} chars")

            # Log full response for debugging
            logger.debug(f"Full LLM response:\n{response}")
            logger.info(f"  Response preview: {response[:300]}...")

            # Parse response
            skills = self._parse_llm_response(response, experiment, feedback, competition_context, code, score)

            logger.info(f"Extracted {len(skills)} skills from experiment")
            for skill in skills:
                logger.info(f"  - {skill.name}: {len(skill.description)} char description, {len(skill.code_pattern)} char code")
            return skills

        except Exception as e:
            logger.error(f"❌ Error extracting skills: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
        full_code: str,
        score: float = 0.0
    ) -> List[Skill]:
        """Parse LLM JSON response into Skill objects."""
        skills = []

        try:
            # Try to parse JSON
            parsed_response = json.loads(response)

            # Extract skills from the structured response {"skills": [...]}
            if isinstance(parsed_response, dict) and "skills" in parsed_response:
                skill_data_list = parsed_response["skills"]
                logger.info(f"Extracted {len(skill_data_list)} skills from 'skills' key")
            elif isinstance(parsed_response, list):
                skill_data_list = parsed_response  # Fallback for old format
                logger.info(f"Using list format directly ({len(skill_data_list)} skills)")
            else:
                logger.warning(f"Unexpected response format: {type(parsed_response)}, keys: {parsed_response.keys() if isinstance(parsed_response, dict) else 'N/A'}")
                skill_data_list = []

            for skill_data in skill_data_list:
                try:
                    # Create skill example
                    example = SkillExample(
                        competition=competition,
                        code=skill_data.get("code_pattern", ""),
                        context=skill_data.get("description", ""),  # Full description, no truncation
                        score=score,  # Use passed score
                    )

                    # Validate skill has meaningful content
                    description = skill_data.get("description", "")
                    code_pattern = skill_data.get("code_pattern", "")
                    name = skill_data.get("name", "unnamed_skill")

                    if not description or not code_pattern:
                        logger.warning(f"⚠️  Skipping empty skill '{name}' - description: {len(description)} chars, code_pattern: {len(code_pattern)} chars")
                        continue

                    if name == "unnamed_skill":
                        logger.warning(f"⚠️  LLM returned generic name 'unnamed_skill' - might be low quality")

                    # Create skill
                    skill = Skill(
                        name=name,
                        description=description,
                        code_pattern=code_pattern,
                        applicable_contexts=skill_data.get("applicable_contexts", []),
                        examples=[example],
                        success_count=1,
                        attempt_count=1,
                        source_competitions=[competition],
                        tags=skill_data.get("applicable_contexts", []),
                    )

                    skills.append(skill)
                    logger.info(f"✅ Validated skill '{name}': {len(description)} char description, {len(code_pattern)} char code")

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
