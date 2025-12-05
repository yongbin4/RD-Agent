"""Debug skill extraction from failure-to-success patterns using LLM."""

from typing import List, Optional, Tuple
import json
import logging

from rdagent.components.skill_learning.debug_skill import DebugSkill, DebugExample
from rdagent.oai.llm_utils import APIBackend

logger = logging.getLogger(__name__)


DEBUG_SKILL_EXTRACTION_PROMPT = """Analyze this failure-to-success transition and extract a problem-solving pattern (debug skill).

## Context
Competition: {competition}
Task: {task_context}

## Failed Attempt
{failed_info}

## Successful Solution
{success_info}

## Task
Extract a reusable problem-solving pattern that captures:

1. **symptom**: How to recognize this problem
   - Error messages or patterns
   - Metric behavior (e.g., "validation score much worse than training score")
   - Observable signs in output

2. **root_cause**: Why this problem occurs (conceptual explanation)
   - Underlying technical reason
   - Common misconception or mistake
   - Conditions that trigger this issue

3. **failed_approach**: The code pattern that causes this problem
   - Extract 10-30 lines of the problematic code
   - Focus on the core mistake, not surrounding context

4. **solution**: The corrected code pattern
   - Extract 10-30 lines of the fixed code
   - Show the key change that resolves the issue

5. **name**: Short snake_case identifier (e.g., "data_leakage_time_series", "memory_error_large_dataset")

6. **description**: Clear explanation of the problem and solution (2-4 sentences)

7. **applicable_contexts**: List of tags (e.g., ["time_series", "data_leakage", "preprocessing"])

8. **severity**: How serious is this problem? "low", "medium", or "high"

Return as JSON object (valid JSON only, no markdown):
{{
  "name": "...",
  "description": "...",
  "symptom": "...",
  "root_cause": "...",
  "failed_approach": "...",
  "solution": "...",
  "applicable_contexts": [...],
  "severity": "medium"
}}

Focus on extracting patterns that:
- Represent common mistakes others might make
- Have clear symptoms that can be recognized
- Have actionable solutions that can be applied
- Are not trivial (skip syntax errors or typos)

If this transition doesn't reveal a meaningful problem-solving pattern, return an empty object: {{}}
"""


class DebugSkillExtractor:
    """Extract problem-solving patterns from failure-to-success transitions using LLM."""

    def __init__(self, llm: APIBackend = None):
        """Initialize with LLM backend."""
        self.llm = llm or APIBackend()

    def extract_from_transition(
        self,
        failed_experiment,
        success_experiment,
        failed_feedback,
        success_feedback,
        competition_context: str = "",
        before_score: Optional[float] = None,
        after_score: Optional[float] = None
    ) -> Optional[DebugSkill]:
        """
        Extract debug skill from a failure-to-success transition.

        Args:
            failed_experiment: The experiment that failed
            success_experiment: The experiment that succeeded
            failed_feedback: Feedback from the failed experiment
            success_feedback: Feedback from the successful experiment
            competition_context: Name/description of the competition
            before_score: Explicit score from failed experiment (if not provided, tries feedback.score)
            after_score: Explicit score from success experiment (if not provided, tries feedback.score)

        Returns:
            DebugSkill object if meaningful pattern found, None otherwise
        """
        # Extract information from both experiments
        failed_info = self._format_experiment_info(failed_experiment, failed_feedback)
        success_info = self._format_experiment_info(success_experiment, success_feedback)

        if not failed_info or not success_info:
            logger.warning("Insufficient information for debug skill extraction")
            return None

        # Extract task context
        task_context = self._extract_task_context(success_experiment)

        # Build prompt
        prompt = DEBUG_SKILL_EXTRACTION_PROMPT.format(
            competition=competition_context,
            task_context=task_context[:300],
            failed_info=failed_info[:2000],
            success_info=success_info[:2000],
        )

        try:
            # Call LLM
            response = self.llm.build_messages_and_create_chat_completion(
                user_prompt=prompt,
                system_prompt="You are an expert at identifying common problems in data science code and their solutions.",
                json_mode=True,
            )

            # Parse response
            debug_skill = self._parse_llm_response(
                response,
                failed_experiment,
                success_experiment,
                failed_feedback,
                success_feedback,
                competition_context,
                before_score=before_score,
                after_score=after_score
            )

            if debug_skill:
                logger.info(f"Extracted debug skill: {debug_skill.name}")
            else:
                logger.debug("No meaningful debug skill extracted from transition")

            return debug_skill

        except Exception as e:
            logger.error(f"Error extracting debug skill: {e}")
            return None

    def extract_from_error_fix(
        self,
        experiment,
        feedback,
        competition_context: str = ""
    ) -> Optional[DebugSkill]:
        """
        Extract debug skill from an experiment that encountered and fixed an error.

        Args:
            experiment: The experiment object
            feedback: Feedback containing error information
            competition_context: Name/description of the competition

        Returns:
            DebugSkill object if meaningful pattern found, None otherwise
        """
        # Extract error information from feedback
        error_info = self._extract_error_info(feedback)
        if not error_info:
            logger.debug("No error information found in feedback")
            return None

        # Extract the solution (current experiment code)
        solution_info = self._format_experiment_info(experiment, feedback)
        if not solution_info:
            logger.warning("No solution code found in experiment")
            return None

        # Extract task context
        task_context = self._extract_task_context(experiment)

        # Build simplified prompt for error-based extraction
        prompt = f"""Analyze this error and its solution to extract a problem-solving pattern.

## Context
Competition: {competition_context}
Task: {task_context[:300]}

## Error Encountered
{error_info[:1500]}

## Solution That Fixed It
{solution_info[:2000]}

Extract a reusable problem-solving pattern with the same fields as before.
Focus on the error pattern and how it was resolved.

Return JSON object (or empty {{}} if not meaningful):
{{
  "name": "...",
  "description": "...",
  "symptom": "...",
  "root_cause": "...",
  "failed_approach": "...",
  "solution": "...",
  "applicable_contexts": [...],
  "severity": "medium"
}}
"""

        try:
            response = self.llm.build_messages_and_create_chat_completion(
                user_prompt=prompt,
                system_prompt="You are an expert at identifying common problems in data science code and their solutions.",
                json_mode=True,
            )

            # Parse with minimal experiment info
            debug_skill = self._parse_llm_response(
                response,
                None,  # No failed experiment in this case
                experiment,
                feedback,
                feedback,
                competition_context
            )

            if debug_skill:
                logger.info(f"Extracted debug skill from error: {debug_skill.name}")

            return debug_skill

        except Exception as e:
            logger.error(f"Error extracting debug skill from error: {e}")
            return None

    def _format_experiment_info(self, experiment, feedback) -> str:
        """Format experiment information for prompt."""
        parts = []

        # Hypothesis
        if hasattr(experiment, "hypothesis"):
            if hasattr(experiment.hypothesis, "hypothesis"):
                hyp = str(experiment.hypothesis.hypothesis)
            else:
                hyp = str(experiment.hypothesis)
            parts.append(f"Hypothesis: {hyp[:500]}")

        # Score
        if hasattr(feedback, "score"):
            parts.append(f"Score: {feedback.score}")

        # Decision
        if hasattr(feedback, "decision"):
            parts.append(f"Accepted: {feedback.decision}")

        # Feedback/Observations
        if hasattr(feedback, "observations"):
            obs = str(feedback.observations)
            if obs and len(obs) > 10:
                parts.append(f"Observations: {obs[:800]}")

        # Code
        code = self._format_experiment_code(experiment)
        if code:
            parts.append(f"Code:\n{code[:1500]}")

        return "\n\n".join(parts)

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

    def _extract_task_context(self, experiment) -> str:
        """Extract task context from experiment."""
        if hasattr(experiment, "hypothesis"):
            if hasattr(experiment.hypothesis, "hypothesis"):
                return str(experiment.hypothesis.hypothesis)
            return str(experiment.hypothesis)
        return "Data science task"

    def _extract_error_info(self, feedback) -> str:
        """Extract error information from feedback."""
        error_parts = []

        # Check observations for error messages
        if hasattr(feedback, "observations"):
            obs = str(feedback.observations)
            if "error" in obs.lower() or "exception" in obs.lower() or "traceback" in obs.lower():
                error_parts.append(f"Error details: {obs[:1000]}")

        # Check execution results
        if hasattr(feedback, "execution_feedback"):
            error_parts.append(f"Execution feedback: {str(feedback.execution_feedback)[:800]}")

        return "\n".join(error_parts)

    def _is_valid_debug_skill(self, skill_data: dict) -> bool:
        """
        Validate that extracted skill is meaningful and not trivial.

        Rejects:
        - Skills where solution is identical to failed approach
        - Trivial errors (syntax errors, typos, missing imports)
        - Too short solutions (< 20 characters)
        - Skills with no clear fix pattern

        Args:
            skill_data: Dictionary containing extracted skill fields

        Returns:
            True if skill passes quality validation, False otherwise
        """
        solution = skill_data.get('solution', '')
        failed_approach = skill_data.get('failed_approach', '')
        symptom = skill_data.get('symptom', '').lower()
        name = skill_data.get('name', '').lower()
        description = skill_data.get('description', '').lower()

        # Reject if solution is identical to failed approach
        if solution and failed_approach and solution.strip() == failed_approach.strip():
            logger.debug("Rejected: solution identical to failed approach")
            return False

        # Reject trivial errors that aren't worth learning
        trivial_keywords = [
            'syntax error', 'syntaxerror',
            'typo', 'typographical',
            'missing comma', 'missing colon', 'missing parenthesis',
            'indentation error', 'indentationerror',
            'missing import',  # Simple import fixes
            'namenotfounderror',
            'misspelled', 'misspelling',
        ]
        combined_text = f"{symptom} {name} {description}"
        if any(kw in combined_text for kw in trivial_keywords):
            logger.debug(f"Rejected trivial error: {name}")
            return False

        # Require minimum length for meaningful solution
        if len(solution.strip()) < 20:
            logger.debug(f"Rejected: solution too short ({len(solution)} chars)")
            return False

        # Require minimum description length
        if len(skill_data.get('description', '').strip()) < 10:
            logger.debug("Rejected: description too short")
            return False

        # Require symptom to be descriptive
        if len(symptom.strip()) < 10:
            logger.debug("Rejected: symptom too short")
            return False

        return True

    def _parse_llm_response(
        self,
        response: str,
        failed_experiment,
        success_experiment,
        failed_feedback,
        success_feedback,
        competition: str,
        before_score: Optional[float] = None,
        after_score: Optional[float] = None
    ) -> Optional[DebugSkill]:
        """Parse LLM JSON response into DebugSkill object."""
        try:
            # Try to parse JSON
            skill_data = json.loads(response)

            # Check if empty response (no meaningful pattern found)
            if not skill_data or len(skill_data) == 0:
                return None

            # Validate required fields
            required_fields = ["name", "symptom", "solution"]
            if not all(field in skill_data for field in required_fields):
                logger.warning(f"Missing required fields in debug skill: {skill_data.keys()}")
                return None

            # Validate skill quality - reject trivial/invalid patterns
            if not self._is_valid_debug_skill(skill_data):
                logger.debug(f"Rejected trivial/invalid debug skill: {skill_data.get('name', 'unknown')}")
                return None

            # Use passed scores, fall back to feedback.score if not provided
            if before_score is None and failed_feedback and hasattr(failed_feedback, "score"):
                try:
                    before_score = float(failed_feedback.score)
                except:
                    pass

            if after_score is None and hasattr(success_feedback, "score"):
                try:
                    after_score = float(success_feedback.score)
                except:
                    pass
            # Ensure after_score is never None
            if after_score is None:
                after_score = 0.0

            # Get code snippets
            failed_code = skill_data.get("failed_approach", "")
            if not failed_code and failed_experiment:
                failed_code = self._format_experiment_code(failed_experiment)[:500]

            solution_code = skill_data.get("solution", "")

            example = DebugExample(
                competition=competition,
                symptom=skill_data.get("symptom", "")[:500],
                failed_code=failed_code[:1000],
                solution_code=solution_code[:1000],
                context=skill_data.get("description", "")[:300],
                before_score=before_score,
                after_score=after_score,
            )

            # Create debug skill
            debug_skill = DebugSkill(
                name=skill_data.get("name", "unnamed_debug_skill"),
                description=skill_data.get("description", ""),
                symptom=skill_data.get("symptom", ""),
                root_cause=skill_data.get("root_cause", "Unknown cause"),
                failed_approach=skill_data.get("failed_approach", ""),
                solution=skill_data.get("solution", ""),
                applicable_contexts=skill_data.get("applicable_contexts", []),
                examples=[example],
                detection_count=1,
                fix_success_count=1,
                source_competitions=[competition],
                tags=skill_data.get("applicable_contexts", []),
                severity=skill_data.get("severity", "medium"),
            )

            return debug_skill

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                try:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    return self._parse_llm_response(
                        json_str,
                        failed_experiment,
                        success_experiment,
                        failed_feedback,
                        success_feedback,
                        competition
                    )
                except:
                    pass

        except Exception as e:
            logger.error(f"Error creating debug skill from data: {e}")

        return None
